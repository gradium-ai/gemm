//! A per-thread cache of packed lhs panels, keyed on the operand itself.
//!
//! `gemm_basic_generic` repacks the lhs on every call. That is right for an activation and
//! wrong for a weight, and it costs most when the output has few columns, since packing
//! costs the same however many columns it is then used for.
//!
//! It assumes the bytes behind an lhs pointer never change -- true for inference, false for
//! a caller that mutates in place -- so it is off unless `GEMM_PACKED_LHS_CACHE=1` or
//! [`set_enabled`]. Two things guard a hit: the key carries a fingerprint sampled from the
//! contents, so a recycled address misses instead of returning a stale panel, and it carries
//! every parameter the layout depends on, notably `kc`.
//!
//! Nothing in a call says whether its lhs is a weight or an activation, and caching an
//! activation buys a panel used once -- worse, it can widen the caller's packing decision
//! onto a slower path. So storage is withheld until an operand's *second* sighting: the first
//! records a probe, which for a weight pays off on the next call and for an activation never
//! does, since its fingerprint moves.
//!
//! Entries are never evicted; [`clear`] releases a thread's copies, and
//! `GEMM_PACKED_LHS_CACHE_MB` or [`set_budget_mb`] caps the total held process-wide. Past the
//! cap, or past [`MAX_ENTRIES`] operands, callers pack per call.

use alloc::alloc::{alloc, dealloc, Layout};
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::any::TypeId;
use core::cell::Cell;
#[cfg(feature = "std")]
use core::cell::RefCell;
use core::sync::atomic::{AtomicU8, AtomicUsize, Ordering::Relaxed};

const DEFAULT_BUDGET_MB: usize = 256;

/// 0 = off, 1 = on, 2 = not yet read from the environment.
static ENABLED: AtomicU8 = AtomicU8::new(2);
/// `usize::MAX` = not yet read from the environment.
static BUDGET_BYTES: AtomicUsize = AtomicUsize::new(usize::MAX);
/// Packed bytes held across every thread.
static USED: AtomicUsize = AtomicUsize::new(0);

#[cfg(feature = "std")]
fn env_enabled() -> bool {
    std::env::var("GEMM_PACKED_LHS_CACHE").as_deref() == Ok("1")
}

#[cfg(not(feature = "std"))]
fn env_enabled() -> bool {
    false
}

#[cfg(feature = "std")]
fn env_budget_bytes() -> usize {
    std::env::var("GEMM_PACKED_LHS_CACHE_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_BUDGET_MB)
        .saturating_mul(1024 * 1024)
}

#[cfg(not(feature = "std"))]
fn env_budget_bytes() -> usize {
    0
}

/// Whether caching is on. Off unless `GEMM_PACKED_LHS_CACHE=1` or [`set_enabled`].
pub fn enabled() -> bool {
    match ENABLED.load(Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = env_enabled();
            ENABLED.store(on as u8, Relaxed);
            on
        }
    }
}

/// Turn caching on or off, overriding the environment. Enabling it asserts this module's
/// precondition for the whole program, so it belongs to whoever owns `main`.
pub fn set_enabled(on: bool) {
    ENABLED.store(on as u8, Relaxed);
}

/// Ceiling on packed bytes held process-wide.
pub fn budget_bytes() -> usize {
    match BUDGET_BYTES.load(Relaxed) {
        usize::MAX => {
            let bytes = env_budget_bytes();
            BUDGET_BYTES.store(bytes, Relaxed);
            bytes
        }
        bytes => bytes,
    }
}

/// Override the budget. Panels already held are not released; see [`clear`].
pub fn set_budget_mb(mb: usize) {
    BUDGET_BYTES.store(mb.saturating_mul(1024 * 1024), Relaxed);
}

/// Claim `bytes` against the budget, or fail having claimed nothing.
fn reserve(bytes: usize) -> bool {
    let budget = budget_bytes();
    let mut used = USED.load(Relaxed);
    loop {
        let next = used.saturating_add(bytes);
        if next > budget {
            return false;
        }
        match USED.compare_exchange_weak(used, next, Relaxed, Relaxed) {
            Ok(_) => return true,
            Err(actual) => used = actual,
        }
    }
}

/// Distinct operands one thread will hold panels for. A model has a few dozen weights; well
/// past that and the lhs is evidently not a weight.
pub const MAX_ENTRIES: usize = 256;

/// Keys held awaiting a second sighting. Large enough that a weight's probe survives one
/// decode step's other operands, small enough that never-repeating operands cost little.
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

/// Raw parts rather than a `Vec` because the microkernel wants `CACHELINE_ALIGN`, which
/// `Vec` will not promise. Shared by `Rc` with every [`Panel`] handed out for it: a `Panel`
/// outlives the borrow that produced it, and pushing an entry reallocates the entry vector
/// while evicting a probe shifts it, so nothing a `Panel` needs may live *in* that vector.
struct Buf {
    ptr: *mut u8,
    layout: Layout,
    /// Set once every k-chunk is packed, so a partial fill from an interrupted call is not
    /// mistaken for a complete one.
    filled: Cell<bool>,
}

impl Drop for Buf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: allocated with exactly this layout in `get`, and freed once.
            unsafe { dealloc(self.ptr, self.layout) }
            USED.fetch_sub(self.layout.size(), Relaxed);
        }
    }
}

/// A key seen once, or one seen again and given storage.
enum Slot {
    Probe,
    Panel(Rc<Buf>),
}

struct Entry {
    key: Key,
    slot: Slot,
}

#[cfg(feature = "std")]
thread_local! {
    /// A short linear scan beats hashing: a few dozen entries, compared a few words each.
    static CACHE: RefCell<Vec<Entry>> = const { RefCell::new(Vec::new()) };
}

#[cfg(feature = "std")]
fn with_cache<R>(f: impl FnOnce(&mut Vec<Entry>) -> R) -> Option<R> {
    Some(CACHE.with(|cache| f(&mut cache.borrow_mut())))
}

/// Without `std` there is nowhere to keep entries, so the cache is permanently absent.
#[cfg(not(feature = "std"))]
fn with_cache<R>(_: impl FnOnce(&mut Vec<Entry>) -> R) -> Option<R> {
    None
}

/// A cached panel: where to write it, and whether it already holds valid data.
pub struct Panel {
    /// Base of the whole buffer, spanning every k-chunk.
    pub ptr: *mut u8,
    /// A previous call already packed this operand, so packing can be skipped.
    pub filled: bool,
    /// Keeps the storage and its flag alive as long as this handle lives, whatever becomes
    /// of the entry that produced it -- a [`clear`] mid-call included.
    buf: Rc<Buf>,
}

impl Panel {
    /// Record that every k-chunk is now packed. Call once, after the last chunk.
    pub fn mark_filled(&self) {
        self.buf.filled.set(true);
    }
}

/// Position-dependent hash of eight elements, so a recycled address holding different
/// contents misses. Constant work regardless of operand size.
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

/// Look up, or reserve, a `bytes`-sized panel for this lhs. `None` -- caching off, first
/// sighting, or budget spent -- means pack per call as before.
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

    with_cache(|cache| {
        match cache.iter().position(|e| e.key == key) {
            Some(i) => {
                // Seen before: promote a probe to real storage, return a panel as-is.
                if matches!(cache[i].slot, Slot::Probe) {
                    let layout = Layout::from_size_align(bytes, align.max(1)).ok()?;
                    if panel_count(cache) >= MAX_ENTRIES || !reserve(bytes) {
                        return None;
                    }
                    let ptr = alloc(layout);
                    if ptr.is_null() {
                        USED.fetch_sub(bytes, Relaxed);
                        return None;
                    }
                    cache[i].slot = Slot::Panel(Rc::new(Buf {
                        ptr,
                        layout,
                        filled: Cell::new(false),
                    }));
                }
                let Slot::Panel(buf) = &cache[i].slot else {
                    return None;
                };
                Some(Panel {
                    ptr: buf.ptr,
                    filled: buf.filled.get(),
                    buf: buf.clone(),
                })
            }
            None => {
                // First sighting: remember the key and allocate nothing.
                if probe_count(cache) >= MAX_PROBES {
                    if let Some(i) = cache.iter().position(|e| matches!(e.slot, Slot::Probe)) {
                        cache.remove(i);
                    }
                }
                cache.push(Entry {
                    key,
                    slot: Slot::Probe,
                });
                None
            }
        }
    })
    .flatten()
}

fn panel_count(cache: &[Entry]) -> usize {
    cache
        .iter()
        .filter(|e| matches!(e.slot, Slot::Panel(_)))
        .count()
}

fn probe_count(cache: &[Entry]) -> usize {
    cache
        .iter()
        .filter(|e| matches!(e.slot, Slot::Probe))
        .count()
}

/// Packed bytes held process-wide, and operands with a panel on *this* thread.
pub fn stats() -> (usize, usize) {
    (
        USED.load(Relaxed),
        with_cache(|cache| panel_count(cache)).unwrap_or(0),
    )
}

/// Drop this thread's entries, returning their bytes to the budget. A `Panel` already
/// handed to an in-flight call keeps its storage alive until that call returns.
pub fn clear() {
    with_cache(|cache| cache.clear());
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: usize = 64;
    const K: usize = 64;
    const BYTES: usize = 64 * 8 * (M / 8) * 4;

    fn operand(seed: usize) -> Vec<f32> {
        (0..M * K).map(|i| (i as f32) * 0.5 + seed as f32).collect()
    }

    fn getp(v: &[f32]) -> Option<Panel> {
        // SAFETY: `v` is an M-by-K f32 matrix with these strides, and no test mutates one.
        unsafe { get::<f32>(v.as_ptr(), M, K, 1, M as isize, 64, 8, BYTES, 64) }
    }

    fn fresh() {
        set_enabled(true);
        set_budget_mb(1024);
        clear();
    }

    /// One test: the switch and the budget are process-wide, so separate `#[test]`s would
    /// need `--test-threads=1` to stay honest.
    #[test]
    fn cache_mechanics() {
        // Off by default, on a repeat sighting as much as a first.
        set_enabled(false);
        let a = operand(1);
        assert!(getp(&a).is_none());
        assert!(getp(&a).is_none());

        // Storage arrives on the second sighting, so an operand seen once costs nothing.
        fresh();
        let a = operand(2);
        assert!(getp(&a).is_none(), "first sighting is a probe, not a panel");
        let panel = getp(&a).expect("second sighting hands back a panel");
        assert!(!panel.filled, "a fresh panel holds nothing yet");
        panel.mark_filled();
        drop(panel);
        assert!(
            getp(&a).expect("still cached").filled,
            "a later call can skip packing"
        );

        // The flag a `Panel` marks must not live in the entry vector. A handle spans a whole
        // gemm call, and a nested gemm on the same thread -- rayon lets the caller steal one
        // -- pushes entries, reallocating that vector, and evicts probes, shifting it.
        fresh();
        let a = operand(3);
        assert!(getp(&a).is_none());
        let panel = getp(&a).expect("panel");
        let others: Vec<Vec<f32>> = (0..200).map(|s| operand(100 + s)).collect();
        for o in &others {
            let _ = getp(o);
        }
        panel.mark_filled();
        drop(panel);
        assert!(
            getp(&a).expect("still cached").filled,
            "mark_filled() reached the right entry"
        );

        // `clear` gives the bytes back and forgets the operand.
        fresh();
        let a = operand(4);
        assert!(getp(&a).is_none());
        let _ = getp(&a).expect("panel");
        assert_eq!(stats(), (BYTES, 1));
        clear();
        assert_eq!(stats(), (0, 0));
        assert!(getp(&a).is_none(), "a cleared operand is a stranger again");

        // A spent budget declines rather than overshooting it.
        fresh();
        set_budget_mb(0);
        let a = operand(5);
        assert!(getp(&a).is_none());
        assert!(getp(&a).is_none(), "no panel once the budget is spent");
        assert_eq!(stats().0, 0, "a refusal claims nothing");
        set_enabled(false);
    }
}
