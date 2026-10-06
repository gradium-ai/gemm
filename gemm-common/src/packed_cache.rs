//! A per-thread cache of packed lhs panels, keyed on the operand itself.
//!
//! `gemm_basic_generic` repacks the lhs on every call. That is right for an activation and
//! wrong for a weight, and it costs most when the output has few columns, since packing
//! costs the same however many columns it is then used for.
//!
//! It assumes the bytes behind a cached lhs never change. That holds for a weight and not for
//! an operand written in place between calls, such as a fixed-size key/value cache, so there
//! are two ways in:
//!
//! - [`with_constant`]: the caller names the bytes it vouches for, and only an lhs lying wholly
//!   inside them is cached. It needs no switch, and it is the one to use from a library,
//!   which cannot vouch for every operand its users pass.
//! - `GEMM_PACKED_LHS_CACHE=1` or [`set_enabled`]: every lhs is trusted, for a program that
//!   knows it never writes an operand in place.
//!
//! Two things guard a hit: the key carries a fingerprint sampled from the contents, so a
//! recycled address misses instead of returning a stale panel, and it carries every parameter
//! the layout depends on, notably `kc`. The fingerprint is eight elements, so it does not
//! catch a write in place that misses them; that is what the two switches are for.
//!
//! Under [`set_enabled`] nothing in a call says whether its lhs is a weight or an activation,
//! and caching an activation buys a panel used once. Worse, it can widen the caller's packing
//! decision onto a slower path. So storage is withheld until an operand's *second* sighting:
//! the first records a probe, which for a weight pays off on the next call and for an
//! activation never does, since its fingerprint moves. The same rule keeps a declared
//! constant that is used only once from holding a panel.
//!
//! Entries are never evicted; [`clear`] releases a thread's copies, and
//! `GEMM_PACKED_LHS_CACHE_MB` or [`set_budget_mb`] caps the total held process-wide. Past the
//! cap, or past [`MAX_ENTRIES`] operands, callers pack per call, so a budget of 0 turns the
//! cache off altogether.

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

/// Whether every lhs is cached, not only those under [`with_constant`]. Off unless
/// `GEMM_PACKED_LHS_CACHE=1` or [`set_enabled`].
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

/// Turn caching of every lhs on or off, overriding the environment. Enabling it asserts this
/// module's precondition for the whole program, so it belongs to whoever owns `main`.
/// Turning it off leaves [`with_constant`] working.
pub fn set_enabled(on: bool) {
    ENABLED.store(on as u8, Relaxed);
}

/// A span of memory a caller vouches for: see [`with_constant`]. The default is empty.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Constant {
    start: usize,
    end: usize,
}

impl Constant {
    /// The `len` elements of `T` starting at `ptr`.
    pub fn new<T>(ptr: *const T, len: usize) -> Self {
        let start = ptr as usize;
        let end = len
            .checked_mul(core::mem::size_of::<T>())
            .and_then(|bytes| start.checked_add(bytes));
        match end {
            Some(end) => Self { start, end },
            None => Self::default(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    /// Whether every element of an `m` by `k` matrix at `ptr` with these strides lies inside.
    fn covers<T>(&self, ptr: *const T, m: usize, k: usize, rs: isize, cs: isize) -> bool {
        let span = |n: usize, s: isize| isize::try_from(n.checked_sub(1)?).ok()?.checked_mul(s);
        // The first byte of the lowest element and the end of the highest one.
        let bounds = || {
            let (along_m, along_k) = (span(m, rs)?, span(k, cs)?);
            let size = core::mem::size_of::<T>() as isize;
            let lo = along_m.min(0).checked_add(along_k.min(0))?;
            let hi = along_m.max(0).checked_add(along_k.max(0))?.checked_add(1)?;
            let first = (ptr as usize).checked_add_signed(lo.checked_mul(size)?)?;
            let end = (ptr as usize).checked_add_signed(hi.checked_mul(size)?)?;
            Some((first, end))
        };
        match bounds() {
            Some((first, end)) => !self.is_empty() && first >= self.start && end <= self.end,
            None => false,
        }
    }
}

#[cfg(feature = "std")]
thread_local! {
    /// What [`with_constant`] has declared on this thread.
    static CONSTANT: Cell<Constant> = const { Cell::new(Constant { start: 0, end: 0 }) };
}

/// Run `f` with `range` declared constant on this thread: a gemm `f` makes on this thread
/// caches its packed lhs if the lhs lies wholly inside `range`, with or without [`enabled`].
/// Any other lhs, the activation in a weight-times-activation product included, is packed per
/// call as usual. The previous declaration is restored when `f` returns or unwinds.
///
/// The declaration is per thread, so a caller that splits a gemm across threads passes
/// [`constant`] to each.
///
/// The bytes in `range` must not change while a panel packed from them may still be held,
/// which is until the thread exits or calls [`clear`]: a model parameter qualifies, a buffer
/// written in place does not. Bytes freed and later declared again at the same address with
/// other contents are what the fingerprint tells apart, as with [`enabled`].
pub fn with_constant<R>(range: Constant, f: impl FnOnce() -> R) -> R {
    #[cfg(feature = "std")]
    {
        struct Restore(Constant);
        impl Drop for Restore {
            fn drop(&mut self) {
                CONSTANT.with(|c| c.set(self.0));
            }
        }
        let _restore = Restore(CONSTANT.with(|c| c.replace(range)));
        f()
    }
    #[cfg(not(feature = "std"))]
    {
        let _ = range;
        f()
    }
}

/// What [`with_constant`] has declared on this thread, empty outside it.
pub fn constant() -> Constant {
    #[cfg(feature = "std")]
    {
        CONSTANT.with(|c| c.get())
    }
    #[cfg(not(feature = "std"))]
    {
        Constant::default()
    }
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

/// Look up, or reserve, a `bytes`-sized panel for this lhs. `None` means pack per call as
/// before: for an lhs nobody vouches for, on a first sighting, or with the budget spent.
///
/// # Safety
///
/// `lhs` must be valid for reads of an `m` by `k` matrix with the given strides. That its
/// bytes do not change is the precondition of [`set_enabled`] and [`with_constant`], whichever
/// let it through.
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
    if bytes == 0 || m == 0 || k == 0 || lhs.is_null() {
        return None;
    }
    if !enabled() && !constant().covers(lhs, m, k, lhs_rs, lhs_cs) {
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
    const BYTES: usize = 64 * 8 * (M / 8) * 4;

    fn operand(seed: usize) -> Vec<f32> {
        (0..M * M).map(|i| (i as f32) * 0.5 + seed as f32).collect()
    }

    fn getp(v: &[f32]) -> Option<Panel> {
        // SAFETY: `v` is an M-by-M f32 matrix with these strides, and no test mutates one.
        unsafe { get::<f32>(v.as_ptr(), M, M, 1, M as isize, 64, 8, BYTES, 64) }
    }

    /// Second sighting, then mark, then hit.
    fn cycle(v: &[f32]) -> Panel {
        assert!(getp(v).is_none(), "first sighting is a probe, not a panel");
        let panel = getp(v).expect("second sighting hands back a panel");
        assert!(!panel.filled, "a fresh panel holds nothing yet");
        panel
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
        set_enabled(false);
        let a = operand(1);
        assert!(getp(&a).is_none());
        assert!(getp(&a).is_none(), "off by default, on a repeat as much as a first");

        fresh();
        cycle(&a).mark_filled();
        assert!(getp(&a).expect("cached").filled, "a later call can skip packing");

        // The flag a `Panel` marks must not live in the entry vector. A handle spans a whole
        // gemm call, and a nested gemm on the same thread -- rayon lets the caller steal one
        // -- pushes entries, reallocating that vector, and evicts probes, shifting it.
        fresh();
        let panel = cycle(&a);
        let others: Vec<Vec<f32>> = (0..200).map(operand).collect();
        for o in &others {
            let _ = getp(o);
        }
        panel.mark_filled();
        drop(panel);
        assert!(getp(&a).expect("cached").filled, "mark_filled() found the right entry");

        fresh();
        drop(cycle(&a));
        assert_eq!(stats(), (BYTES, 1));
        clear();
        assert_eq!(stats(), (0, 0), "clear returns the bytes");
        assert!(getp(&a).is_none(), "a cleared operand is a stranger again");

        fresh();
        set_budget_mb(0);
        assert!(getp(&a).is_none());
        assert!(getp(&a).is_none(), "no panel once the budget is spent");
        assert_eq!(stats().0, 0, "a refusal claims nothing");

        // With the switch off, a declaration admits what lies inside it and nothing else.
        fresh();
        set_enabled(false);
        let b = operand(2);
        let in_a = Constant::new(a.as_ptr(), a.len());
        with_constant(in_a, || {
            cycle(&a).mark_filled();
            assert!(getp(&a).expect("declared").filled);
            assert!(getp(&b).is_none());
            assert!(getp(&b).is_none(), "an undeclared lhs is never cached");
        });
        assert_eq!(constant(), Constant::default(), "the declaration ends with the closure");
        assert!(getp(&a).is_none(), "and so does the cache's trust in `a`");
        clear();
    }

    #[test]
    fn constant_covers_only_its_own_bytes() {
        let a = operand(1);
        let in_a = Constant::new(a.as_ptr(), a.len());
        let (m, cs) = (M, M as isize);
        assert!(in_a.covers(a.as_ptr(), m, M, 1, cs));
        assert!(in_a.covers(a[M..].as_ptr(), m, M - 1, 1, cs), "a block of columns");
        assert!(!in_a.covers(a[M..].as_ptr(), m, M, 1, cs), "one column past the end");
        assert!(in_a.covers(a[M - 1..].as_ptr(), m, M, -1, cs), "rows walked backwards");
        assert!(!in_a.covers(a[M - 2..].as_ptr(), m, M, -1, cs), "one row before the start");
        assert!(!Constant::default().covers(a.as_ptr(), m, M, 1, cs));

        let in_b = Constant::new(a[M..].as_ptr(), M);
        with_constant(in_a, || {
            with_constant(in_b, || assert_eq!(constant(), in_b));
            assert_eq!(constant(), in_a, "an inner declaration gives way to the outer one");
            let unwound = std::panic::catch_unwind(|| with_constant(in_b, || panic!("unwind")));
            assert!(unwound.is_err());
            assert_eq!(constant(), in_a, "and does so when it unwinds");
        });
    }
}
