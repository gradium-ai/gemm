//! A per-thread cache of packed left-hand-side panels, keyed on the operand itself.
//!
//! # Why
//!
//! `gemm_basic_generic` packs the lhs into a microkernel-friendly layout on every call and
//! throws the panel away when it returns. That is the right trade when the lhs is an
//! activation, which changes every call, and the wrong one when it is a weight, which does
//! not.
//!
//! It bites hardest when the *other* dimension is small. Packing costs time proportional to
//! the lhs, independent of how many columns of output it is then used for, so with few
//! columns there is nothing to amortise it over. Measured on a Cortex-A76 with the shapes a
//! batch-1 speech model issues -- a weight of 512x1536 up to 512x3584, against 16 columns --
//! packing is 42% to 60% of the whole gemm, and stays roughly constant in absolute terms as
//! the column count grows (542us at 16 columns, 628us at 128 for one such shape).
//!
//! Caching the panel removes that. On the same workload it recovers the gap to XNNPACK, whose
//! `fully_connected` operator wins here for exactly this reason: it packs weights once at
//! operator-create time.
//!
//! # What it assumes
//!
//! The bytes behind a given lhs pointer do not change while the process runs. That holds for
//! inference, where weights are loaded once and then only read. It does *not* hold if a
//! caller mutates a matrix in place and reuses the buffer, so this is off unless
//! `GEMM_PACKED_LHS_CACHE=1`.
//!
//! Two things guard against a wrong hit: the key includes a fingerprint sampled from the
//! operand, so an address freed and reused for different contents misses rather than
//! silently returning a stale panel; and the key includes every parameter the packed layout
//! depends on, notably `kc`, which varies with the shape.
//!
//! Entries are never evicted, so the cache converges on one packed copy of each weight per
//! thread that touched it. `GEMM_PACKED_LHS_CACHE_MB` bounds the total (256 MB by default);
//! past that, callers fall back to packing per call.
//!
//! # Telling a weight from an activation
//!
//! Nothing in a gemm call says whether its lhs is a weight worth caching or an activation
//! that will never be seen again. Guessing wrong is expensive in one direction: caching an
//! activation allocates a panel that is used once, and -- because a caller may widen its
//! packing decision when a panel exists -- can push it onto a slower path than it started on.
//!
//! So an operand is not cached the first time it is seen. The first sighting records a probe:
//! the key only, no allocation. A panel is allocated on the *second* sighting of the same key,
//! which for a weight is the next call and for an activation never comes, since its
//! fingerprint moves. Probes are capped and evicted oldest-first, so a stream of
//! never-repeated operands costs a bounded ring of keys and nothing else.

use alloc::alloc::{alloc, dealloc, Layout};
use alloc::vec::Vec;
use core::any::TypeId;
use core::cell::RefCell;

/// Whether caching is on. Off unless `GEMM_PACKED_LHS_CACHE=1`.
#[cfg(feature = "std")]
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("GEMM_PACKED_LHS_CACHE").as_deref() == Ok("1"))
}

#[cfg(not(feature = "std"))]
pub fn enabled() -> bool {
    false
}

#[cfg(feature = "std")]
fn budget_bytes() -> usize {
    static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *B.get_or_init(|| {
        std::env::var("GEMM_PACKED_LHS_CACHE_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(256)
            .saturating_mul(1024 * 1024)
    })
}

#[cfg(not(feature = "std"))]
fn budget_bytes() -> usize {
    0
}

/// Distinct operands one thread will hold panels for. A model has a few dozen weights; well
/// past that and the lhs is evidently not a weight.
const MAX_ENTRIES: usize = 256;

/// Keys held awaiting a second sighting. Large enough that a weight's probe survives the
/// other operands of one decode step, small enough that never-repeating operands cost little.
const MAX_PROBES: usize = 128;

/// Everything the packed layout depends on, plus enough of the operand to tell it apart from
/// a different one that happens to land at the same address.
#[derive(PartialEq, Eq, Clone, Copy)]
struct Key {
    type_id: TypeId,
    ptr: usize,
    m: usize,
    k: usize,
    lhs_rs: isize,
    lhs_cs: isize,
    kc: usize,
    mr: usize,
    bytes: usize,
    fingerprint: u64,
}

/// An aligned allocation that outlives the call that filled it. Held by raw parts rather than
/// a `Vec` because the microkernel wants `CACHELINE_ALIGN`, which `Vec` will not promise.
struct Buf {
    ptr: *mut u8,
    layout: Layout,
}

impl Drop for Buf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: allocated with exactly this layout in `insert`, and freed once.
            unsafe { dealloc(self.ptr, self.layout) }
        }
    }
}

/// A key seen once (`Probe`), or one seen again and given storage (`Panel`).
enum Slot {
    Probe,
    Panel(Buf),
}

struct Entry {
    key: Key,
    slot: Slot,
    /// Set once every k-chunk has been packed, so a partially filled entry from an
    /// interrupted call is not mistaken for a complete one.
    filled: bool,
}

thread_local! {
    /// A short linear scan beats hashing here: a model touches a few dozen distinct weights,
    /// and the comparison is a handful of words.
    static CACHE: RefCell<Vec<Entry>> = const { RefCell::new(Vec::new()) };
    static USED: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

/// A cached panel: where to write it, and whether it already holds valid data.
pub struct Panel {
    /// Base of the whole buffer, spanning every k-chunk.
    pub ptr: *mut u8,
    /// True if a previous call already packed this operand, so packing can be skipped.
    pub filled: bool,
    /// Points at the entry's own flag, for [`mark_filled`].
    slot: *mut bool,
}

impl Panel {
    /// Record that every k-chunk has now been packed. Call once, after the last chunk.
    pub fn mark_filled(&self) {
        // SAFETY: `slot` points into an entry that is never removed or moved -- entries live
        // behind a stable heap allocation and the vector only ever grows -- and only this
        // thread can reach it.
        unsafe { *self.slot = true }
    }
}

/// Cheap, position-dependent hash of a few elements, so a recycled address with different
/// contents does not hit. Constant work regardless of operand size.
///
/// # Safety
///
/// `lhs` must be valid for reads of an `m` by `k` matrix with the given strides.
unsafe fn fingerprint<T: 'static>(
    lhs: *const T,
    m: usize,
    k: usize,
    lhs_rs: isize,
    lhs_cs: isize,
) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ ((m as u64) << 32) ^ (k as u64);
    let size = core::mem::size_of::<T>();
    for step in 0..8usize {
        let i = (m - 1).min(step * m / 8);
        let j = (k - 1).min(step * k / 8);
        let p = lhs.offset(i as isize * lhs_rs + j as isize * lhs_cs) as *const u8;
        for b in 0..size {
            h ^= *p.add(b) as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

/// Look up, or reserve, a `bytes`-sized panel for this lhs.
///
/// Returns `None` when caching is off, the operand looks unsuitable, or the budget is spent;
/// the caller then packs per call as before.
///
/// # Safety
///
/// `lhs` must be valid for reads of an `m` by `k` matrix with the given strides, and the
/// bytes behind it must not change for the lifetime of the process.
#[allow(clippy::too_many_arguments)]
pub unsafe fn get<T: 'static>(
    lhs: *const T,
    m: usize,
    k: usize,
    lhs_rs: isize,
    lhs_cs: isize,
    kc: usize,
    mr: usize,
    bytes: usize,
    align: usize,
) -> Option<Panel> {
    if !enabled() || bytes == 0 || m == 0 || k == 0 || lhs.is_null() {
        return None;
    }
    let key = Key {
        type_id: TypeId::of::<T>(),
        ptr: lhs as usize,
        m,
        k,
        lhs_rs,
        lhs_cs,
        kc,
        mr,
        bytes,
        fingerprint: fingerprint(lhs, m, k, lhs_rs, lhs_cs),
    };

    CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        match cache.iter().position(|e| e.key == key) {
            Some(i) => {
                // Second sighting or later. Promote a probe to real storage now; a panel is
                // returned as-is.
                if matches!(cache[i].slot, Slot::Probe) {
                    let would_use = USED.with(|u| u.get()).saturating_add(bytes);
                    if would_use > budget_bytes() || panel_count(&cache) >= MAX_ENTRIES {
                        return None;
                    }
                    let layout = Layout::from_size_align(bytes, align.max(1)).ok()?;
                    let ptr = alloc(layout);
                    if ptr.is_null() {
                        return None;
                    }
                    USED.with(|u| u.set(would_use));
                    cache[i].slot = Slot::Panel(Buf { ptr, layout });
                    cache[i].filled = false;
                }
                let e = &mut cache[i];
                let Slot::Panel(buf) = &e.slot else { return None };
                let ptr = buf.ptr;
                Some(Panel { ptr, filled: e.filled, slot: &mut e.filled })
            }
            None => {
                // First sighting: remember the key, allocate nothing, and let this call pack
                // the way it would have without a cache.
                if probe_count(&cache) >= MAX_PROBES {
                    if let Some(i) = cache.iter().position(|e| matches!(e.slot, Slot::Probe)) {
                        cache.remove(i);
                    }
                }
                cache.push(Entry { key, slot: Slot::Probe, filled: false });
                None
            }
        }
    })
}

fn panel_count(cache: &[Entry]) -> usize {
    cache.iter().filter(|e| matches!(e.slot, Slot::Panel(_))).count()
}

fn probe_count(cache: &[Entry]) -> usize {
    cache.iter().filter(|e| matches!(e.slot, Slot::Probe)).count()
}

/// Packed bytes this thread is holding, and how many operands have a panel.
pub fn stats() -> (usize, usize) {
    (USED.with(|u| u.get()), CACHE.with(|c| panel_count(&c.borrow())))
}
