//! A byte ceiling on tree-sitter trees held concurrently by parallel walks (#70).
//!
//! ## The term this bounds
//!
//! `find_definitions` and the caller/callee/sibling walks parse a file per walk thread and hold the
//! resulting tree while they read it. A tree is a large multiple of its file's bytes, so peak RSS
//! carried a `walk_threads × tree_size` term that neither `MAX_RETAINED` nor `OFFER_CHUNK` touches.
//! Measured on 60 files of 499 000 B, isolating the parse by re-running the same bytes with the
//! grammar removed:
//!
//! ```text
//!                                 t=1        t=6       t=32     per thread
//! ordinary source (39 B/line)   25.0 MB    95.1 MB   448 MB      ~13 MB
//! dense source    (16 B/line)   48.2 MB   219.9 MB  1090 MB      ~32 MB
//! ```
//!
//! The per-thread figure is flat across a 32x change in thread count, which is what identifies it as
//! one live tree per thread. #19 asked for peak to be "bounded by a configurable ceiling rather than
//! by match count"; #59 delivered the match-count half, and this is the ceiling.
//!
//! ## Why a reservation, and why sized per file
//!
//! A plain semaphore over concurrent parses cannot work. Tree sizes in one repository span three
//! orders of magnitude — a 500-byte file's tree is 23 KB, a 499 000-byte file's is 29 MB — so a
//! permit count set to make the worst case fit throttles the common case to a standstill, and one
//! set for the common case does not bound the worst. Admission has to be sized.
//!
//! ## The estimator, and why lines rather than bytes
//!
//! The size must be known *before* parsing, so it is estimated. Which predictor to use was measured
//! rather than assumed: 95 files across four grammars, exact tree bytes via `ts_set_allocator`
//! counting hooks (tree-sitter allocates through C `malloc`, so a Rust `#[global_allocator]` sees
//! none of it). Taking each predictor's corpus maximum as a conservative constant, over files >= 10
//! KB — smaller ones cannot move a ceiling:
//!
//! ```text
//! predictor   constant         mean over-estimate   worst
//! bytes       58.8 B/byte            4.00x          28.4x
//! lines       1386  B/line           1.93x          12.3x
//! tokens       147  B/token          1.97x          10.2x
//! ```
//!
//! Bytes are the obvious choice and the wrong one: the same 499 000-byte file is 24x its bytes as
//! ordinary source and 59x as dense one-statement-per-line source, because tree size tracks *node*
//! count and nodes track lines far better than they track bytes. Lines and tokens are equivalent on
//! this corpus; lines wins because counting newlines is a SIMD `memchr` pass over bytes already in
//! memory, where tokenising is byte-by-byte.
//!
//! **Over-estimating is the safe direction** and is what the constant is chosen for: it costs
//! parallelism, never correctness, because a rejected parse waits rather than being skipped. The
//! mean ~2x over-estimate means a ceiling of N admits about N/2 of real trees.
//!
//! ## What reserves, and what deliberately does not
//!
//! Every parse that can run concurrently with another *and* whose tree is transient goes through
//! `parse_budgeted`. That is six sites: the two definition walks, `callers`, `callees`, `siblings`,
//! and — less obviously — `lang::outline::get_outline_entries` and `diff::compute_structural_hash`,
//! because `diff` builds its overlays under `par_iter` and calls the former twice per changed file.
//! The diff pair is worth naming because an audit by grepping for `walker` would miss it: it is
//! parallel without being a file walk.
//!
//! Four sites deliberately still call `parse_masked` directly:
//!
//! * `cache::get_or_parse` — retains its tree in `OutlineCache` for the process lifetime, so a
//!   permit held with it would never be released and the budget would fill and stay full. That
//!   retention is #67 and needs its own fix; charging it here would convert a memory leak into a
//!   deadlock-adjacent stall, which is strictly worse.
//! * `read::outline::code`, `read::outline::test_file`, `edit_parse_check` — one file per request,
//!   no concurrency, so a reservation would always be admitted immediately and would bound nothing.
//!
//! `map` also parses per file but its walker is sequential (`for entry in walker.flatten()`), so it
//! is in the second category by construction.

use std::ops::Deref;
use std::sync::{Condvar, Mutex, OnceLock};

use crate::types::Lang;

/// Estimated tree bytes per line of source.
///
/// This has to be an **upper bound**, not a typical value: the ceiling holds only where
/// `estimate >= actual`. The densest files in this repository's own source, over files >= 10 KB:
///
/// ```text
///   1675 B/line   src/mcp/tools/definitions.rs   (405 lines of JSON schema literals)
///   1327 B/line   src/search/callee_query.rs
///   1281 B/line   src/mcp/tools/write.rs
///   1239 B/line   src/classify.rs
///   1228 B/line   src/diff/format.rs
/// ```
///
/// That is a *cluster*, not an outlier, so 2048 is set above the top of it rather than just above
/// the single maximum. `calibrate_tree_bytes_per_line` asserts the bound and prints this list.
///
/// **It was 1536, which did not bound this repository.** The first calibration ran over a corpus
/// gathered with a shell glob that silently did not recurse, so it saw `src/search/*.rs` and
/// `src/*.rs` and never `src/mcp/tools/definitions.rs` — the densest file in the tree. The committed
/// harness found that on its first real run. The lesson worth keeping is about corpora rather than
/// constants: a predictor validated on a subtree is validated on whatever that subtree happens to
/// contain, and the failure is silent in the unsafe direction.
///
/// Raising this admits proportionally fewer concurrent parses, so searches over large-file trees get
/// slower without getting more correct — which is why the default ceiling is re-measured whenever it
/// moves. Lowering it below a grammar's real cost exceeds the ceiling by that ratio: bounded
/// overshoot, not a wrong answer.
const TREE_BYTES_PER_LINE: usize = 2048;

/// Default ceiling on concurrently-held tree bytes.
///
/// Chosen to be **inert at the default thread count and binding above it**, and re-measured whenever
/// `TREE_BYTES_PER_LINE` moves — because the ceiling is denominated in *estimated* bytes, so a more
/// conservative estimator makes the same ceiling bind sooner. Peak working set and wall, five reps
/// per cell (three at 32 threads), fixtures of 60 files of 499 000 B:
///
/// ```text
///                          before          after        wall before   wall after
/// ordinary, default t   94.7-95.2 MB   94.0-95.9 MB    1.04-1.07 s   1.05-1.22 s
/// ordinary, t=32         445-452  MB    216-228  MB    0.79-0.82 s   0.80-0.82 s
/// dense, default t       220-221  MB    220-222  MB    2.80-3.56 s   2.80-3.63 s
/// dense, t=32           1091-1092 MB    264-265  MB    2.42-2.57 s   2.77-2.92 s
/// this repository       20.6-21.2 MB   19.0-22.0 MB    0.08-0.10 s   0.07-0.11 s
/// ```
///
/// **4.1x** at 32 threads on the dense shape for ~15% wall, **2.0x** at 32 threads on ordinary source
/// for none, and genuinely inert at the default — every default-thread cell inside the other's range,
/// on ordinary *and* dense input.
///
/// That last property is what set the number, and it moved: at 256 MB with the corrected estimator
/// the dense shape bound at the default too, costing ~30% wall (3.55-4.39 s against 2.80-3.56 s) to
/// save 64 MB. Swept on the dense fixture to find where that stops:
///
/// ```text
/// ceiling   default-t peak     default-t wall     t=32 peak    t=32 wall
///   256    155.5-156.9 MB     3.55-4.39 s        199-200 MB   3.65-3.70 s
///   320    187.5-189.4 MB     2.99-3.83 s        232    MB    3.05-3.22 s
///   384    221.0-222.1 MB     3.18-3.60 s        264-265 MB   2.77-3.02 s
///   512    220.0-220.4 MB     2.76-3.55 s        328-330 MB   2.51-2.55 s
/// ```
///
/// 384 is the smallest that leaves the default-thread case alone. 512 buys ~10% more wall at 32
/// threads for 64 MB more memory, which is the wrong side of the trade for a bound.
///
/// Override with `TILTH_PARSE_BUDGET_MB`. Three cells say the knob is real rather than decorative,
/// all on the dense fixture at 32 threads:
///
/// ```text
///   0 (disabled)   1090-1092 MB   2.50-2.74 s    reproduces the unbudgeted figures exactly
///  64              101 -102  MB  10.36-10.47 s   tighter ceiling; the cost is the serialisation
///   1              101 -102  MB  10.32-10.44 s   a ceiling 64x smaller than one file's estimate
/// ```
///
/// The `0` row is the control that says the mechanism is off when asked. The `1` row is the one worth
/// reading twice: one file's estimate is ~64 MB against a 1 MB ceiling, and the search still
/// completes with identical output, because `reserve` always admits when nothing else is in flight.
/// It degrades to serial parsing rather than hanging, and lands at the same peak as the 64 MB ceiling
/// because one tree is the floor either way.
const DEFAULT_BUDGET_MB: usize = 384;

/// Process-wide budget. One instance, because the thing being bounded is process peak RSS.
///
/// Per-search would be wrong twice over: `symbol::search` runs two walks concurrently under
/// `rayon::join`, and `grok` runs more, so a per-search budget would permit a multiple of its own
/// ceiling.
pub struct ParseBudget {
    /// `0` means unbounded — see `DEFAULT_BUDGET_MB`.
    ceiling: usize,
    /// Sum of the estimates of every parse currently admitted.
    ///
    /// A `Mutex` rather than an atomic because admission has to test-and-increment against the
    /// ceiling as one step and then sleep on the result; an atomic would need a spin loop, and the
    /// thing being waited for takes tens of milliseconds. One acquisition per parse is nothing
    /// against a parse — this is emphatically not the per-match contention `search::retain` warns
    /// about, since it is one lock per *file*, and each holds it only to add a `usize`.
    in_flight: Mutex<usize>,
    space: Condvar,
}

impl ParseBudget {
    fn from_env() -> Self {
        let ceiling = std::env::var("TILTH_PARSE_BUDGET_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_BUDGET_MB)
            .saturating_mul(1024 * 1024);
        Self {
            ceiling,
            in_flight: Mutex::new(0),
            space: Condvar::new(),
        }
    }

    #[cfg(test)]
    fn with_ceiling(bytes: usize) -> Self {
        Self {
            ceiling: bytes,
            in_flight: Mutex::new(0),
            space: Condvar::new(),
        }
    }

    /// Reserve `estimate` bytes, waiting until they fit.
    ///
    /// **Always admits when nothing else is in flight, however large the estimate.** That is the
    /// property that makes this deadlock-free rather than merely unlikely to deadlock: a single file
    /// whose estimate exceeds the whole ceiling would otherwise wait for space that only it could
    /// release. Every admitted reservation is released by `Permit::drop`, including on panic, so
    /// `in_flight` always returns to zero and some waiter always makes progress.
    ///
    /// The consequence is that the ceiling is soft by up to one file's tree: with nothing in flight,
    /// a 29 MB tree is admitted against any ceiling. Bounding it harder would mean refusing to parse
    /// a file, which changes the answer — and admission must never do that, or which definitions a
    /// search finds would depend on scheduling, the class of defect #8 and #18 removed.
    ///
    /// **This makes #61 slightly worse, and that is the one cost not visible in the measurements.**
    /// `timeout.rs` stops waiting on an expired request but does not cancel its worker, so the
    /// worker keeps running with nothing consuming its result. A worker blocked here lives *longer*
    /// than one that is not, because it now waits for space as well as for its own work — and an
    /// abandoned request is one logical query, so its effective parallelism is low and it is
    /// unusually likely to be the thread that waits. The trade is still right: an abandoned worker
    /// holding a bounded amount of memory for longer beats eight of them holding an unbounded amount,
    /// which is what #61 measures. But if #61 gains cooperative cancellation, the flag has to be
    /// checked *here* as well as in the file callback, or a cancelled worker will sit in this wait
    /// until unrelated work releases it.
    fn reserve(&self, estimate: usize) -> Permit<'_> {
        if self.ceiling == 0 {
            return Permit {
                budget: self,
                estimate: 0,
            };
        }
        let mut in_flight = self
            .in_flight
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // `*in_flight > 0` is the deadlock-freedom guard described above, not an optimisation.
        while *in_flight > 0 && *in_flight + estimate > self.ceiling {
            in_flight = self
                .space
                .wait(in_flight)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        *in_flight += estimate;
        Permit {
            budget: self,
            estimate,
        }
    }

    /// Estimated bytes currently reserved. Report-only; nothing branches on it outside `reserve`.
    #[cfg(test)]
    fn in_flight(&self) -> usize {
        *self
            .in_flight
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// A reservation, released on drop.
struct Permit<'a> {
    budget: &'a ParseBudget,
    estimate: usize,
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        if self.estimate == 0 {
            return;
        }
        let mut in_flight = self
            .budget
            .in_flight
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *in_flight = in_flight.saturating_sub(self.estimate);
        // `notify_all`, not `notify_one`: waiters are waiting for *different* amounts, so the one
        // woken might not fit while another would. Every waiter re-tests its own condition.
        drop(in_flight);
        self.budget.space.notify_all();
    }
}

/// A parsed tree that holds its budget reservation for exactly as long as the tree lives.
///
/// The reservation has to outlive the tree, not the parse: releasing when `parse` returns would
/// leave each walk thread free to hold a tree with nothing accounting for it, which is the whole
/// term being bounded. Pairing them in one struct makes that structural rather than a comment —
/// there is no way to obtain the tree without the permit, and no way to drop the permit early.
///
/// Field order is load-bearing: struct fields drop in declaration order, so `tree` is freed before
/// `_permit` un-charges it. Reversed, the budget would show space that tree-sitter had not yet
/// returned to the allocator.
pub struct BudgetedTree {
    tree: tree_sitter::Tree,
    /// `'static` because the only budget a parse can reserve against is the process-wide one, which
    /// keeps this lifetime out of the public signature.
    _permit: Permit<'static>,
}

impl Deref for BudgetedTree {
    type Target = tree_sitter::Tree;
    fn deref(&self) -> &Self::Target {
        &self.tree
    }
}

fn global() -> &'static ParseBudget {
    static BUDGET: OnceLock<ParseBudget> = OnceLock::new();
    BUDGET.get_or_init(ParseBudget::from_env)
}

/// Estimated tree bytes for `content`, from its line count.
///
/// `memchr` rather than `content.lines().count()`: this runs once per parsed file on the walk's hot
/// path, and counting newlines is the SIMD-friendly half of what `lines()` does. The `+ 1` treats a
/// file with no trailing newline as having a final line, and keeps a one-line file from estimating
/// zero.
fn estimate_bytes(content: &str) -> usize {
    let lines = memchr::memchr_iter(b'\n', content.as_bytes()).count() + 1;
    lines.saturating_mul(TREE_BYTES_PER_LINE)
}

/// `lang::parse_masked`, holding a budget reservation for the tree's lifetime.
///
/// For walk-time parses whose tree is transient. See the module header for what is deliberately not
/// routed through here, and why.
pub fn parse_budgeted(
    content: &str,
    lang: Option<Lang>,
    ts_lang: &tree_sitter::Language,
) -> Option<BudgetedTree> {
    let permit = global().reserve(estimate_bytes(content));
    let tree = super::parse_masked(content, lang, ts_lang)?;
    Some(BudgetedTree {
        tree,
        _permit: permit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// A reservation is charged while held and returned on drop.
    #[test]
    fn a_permit_charges_the_budget_and_releases_it() {
        let b = ParseBudget::with_ceiling(1000);
        assert_eq!(b.in_flight(), 0);
        {
            let _p = b.reserve(400);
            assert_eq!(b.in_flight(), 400);
            {
                let _q = b.reserve(500);
                assert_eq!(b.in_flight(), 900);
            }
            assert_eq!(b.in_flight(), 400);
        }
        assert_eq!(b.in_flight(), 0);
    }

    /// A file whose estimate exceeds the entire ceiling must still parse.
    ///
    /// This is the deadlock-freedom property, and it is the one that would turn a memory bound into
    /// a hang. Without the `in_flight > 0` guard in `reserve` this test blocks forever rather than
    /// failing, which is why it is a test and not a comment.
    #[test]
    fn an_estimate_larger_than_the_ceiling_is_still_admitted() {
        let b = ParseBudget::with_ceiling(1024);
        let p = b.reserve(64 * 1024 * 1024);
        assert_eq!(b.in_flight(), 64 * 1024 * 1024);
        drop(p);
        assert_eq!(b.in_flight(), 0);
    }

    /// `0` means unbounded, and costs nothing while it is.
    #[test]
    fn a_zero_ceiling_disables_accounting() {
        let b = ParseBudget::with_ceiling(0);
        let _p = b.reserve(usize::MAX);
        assert_eq!(b.in_flight(), 0, "a disabled budget must not accumulate");
    }

    /// A reservation that does not fit waits, and proceeds once space is returned.
    ///
    /// Asserts the ordering rather than just the outcome: the waiter must not be admitted until the
    /// holder releases, which is what distinguishes a bound from a counter.
    #[test]
    fn a_reservation_waits_for_space_and_then_proceeds() {
        let b = Arc::new(ParseBudget::with_ceiling(1000));
        let order = Arc::new(AtomicUsize::new(0));

        let held = b.reserve(800);
        assert_eq!(b.in_flight(), 800);

        let (b2, order2) = (Arc::clone(&b), Arc::clone(&order));
        let waiter = std::thread::spawn(move || {
            // 800 + 400 > 1000, and something is in flight, so this must block.
            let _p = b2.reserve(400);
            // 2 if the holder released first, 1 if this was wrongly admitted immediately.
            order2.fetch_add(2, Ordering::SeqCst)
        });

        // Give the waiter time to reach the wait. A sleep is the honest tool here: the property is
        // "it did not proceed", and there is nothing to poll that would not itself be the bug.
        std::thread::sleep(std::time::Duration::from_millis(150));
        assert_eq!(
            order.load(Ordering::SeqCst),
            0,
            "reservation was admitted while the budget was full"
        );

        order.fetch_add(1, Ordering::SeqCst);
        drop(held);
        let seen_before = waiter.join().expect("waiter panicked");
        assert_eq!(
            seen_before, 1,
            "waiter proceeded before the holder released its reservation"
        );
        assert_eq!(b.in_flight(), 0, "budget did not return to zero");
    }

    /// Concurrent reserve/release cannot desynchronise the counter from reality.
    ///
    /// In the shape of `bloom`'s `concurrent_probes_of_one_path_charge_the_budget_once`: the failure
    /// mode for byte accounting under concurrency is a permanently wrong budget rather than a crash,
    /// so the assertion is that it returns exactly to zero after a storm of overlapping permits.
    #[test]
    fn concurrent_reservations_return_the_budget_to_zero() {
        let b = Arc::new(ParseBudget::with_ceiling(1 << 20));
        let mut handles = Vec::new();
        for t in 0..8 {
            let b = Arc::clone(&b);
            handles.push(std::thread::spawn(move || {
                for i in 0..200 {
                    let _p = b.reserve(1 + (t * 31 + i * 17) % 4096);
                }
            }));
        }
        for h in handles {
            h.join().expect("thread panicked");
        }
        assert_eq!(b.in_flight(), 0);
    }

    /// The estimate is a function of line count, not of bytes.
    ///
    /// Two files of the same length and very different line counts must not estimate the same, which
    /// is the entire reason the predictor is lines — see the module header's measured comparison.
    #[test]
    fn the_estimate_tracks_lines_not_bytes() {
        let dense = "a\n".repeat(500); // 1000 bytes, 500 lines
        let sparse = format!("{}\n", "a".repeat(998)); // 999 bytes, 1 line
        assert_eq!(dense.len(), 1000);
        assert_eq!(sparse.len(), 999);
        assert_eq!(estimate_bytes(&dense), 501 * TREE_BYTES_PER_LINE);
        assert_eq!(estimate_bytes(&sparse), 2 * TREE_BYTES_PER_LINE);
        assert!(
            estimate_bytes(&dense) > 100 * estimate_bytes(&sparse),
            "estimator is not discriminating on line count"
        );
    }

    /// An empty file still estimates one line, so nothing reserves zero and slips the accounting.
    #[test]
    fn an_empty_file_estimates_one_line() {
        assert_eq!(estimate_bytes(""), TREE_BYTES_PER_LINE);
    }

    // -----------------------------------------------------------------------
    // Calibration harness for TREE_BYTES_PER_LINE
    // -----------------------------------------------------------------------

    /// Counting hooks for tree-sitter's own allocator, so a tree's bytes can be measured exactly.
    ///
    /// tree-sitter is a C library and allocates through `malloc`, so a Rust `#[global_allocator]`
    /// sees **none** of it — the first attempt at this measurement reported zero for every file.
    /// `ts_set_allocator` is the only way to observe it.
    ///
    /// Each block carries a 16-byte header holding a magic tag and its size, which keeps the
    /// returned pointer 16-aligned (C's `max_align_t`) and makes the swap safe to perform at any
    /// point: a block allocated by the *default* malloc before the swap has no magic, and `ts_free`
    /// leaks it rather than passing a fabricated layout to `dealloc`. Leaking in a `#[ignore]`d
    /// measurement is free; the alternative is undefined behaviour, and it is a real risk because
    /// the lib test binary parses trees in hundreds of other tests that may run first.
    mod alloc_probe {
        use std::alloc::{alloc, dealloc, Layout};
        use std::ffi::c_void;
        use std::sync::atomic::{AtomicUsize, Ordering};

        pub static LIVE: AtomicUsize = AtomicUsize::new(0);

        const HDR: usize = 16;
        const MAGIC: usize = 0x7115_7401_7115_7401;

        fn layout(total: usize) -> Layout {
            Layout::from_size_align(total, HDR).expect("layout")
        }

        unsafe fn tagged_alloc(size: usize) -> *mut c_void {
            let base = alloc(layout(size + HDR));
            if base.is_null() {
                return std::ptr::null_mut();
            }
            base.cast::<usize>().write(MAGIC);
            base.add(8).cast::<usize>().write(size);
            LIVE.fetch_add(size, Ordering::Relaxed);
            base.add(HDR).cast()
        }

        /// `Some(size)` only for blocks this module allocated.
        unsafe fn tagged_size(p: *mut c_void) -> Option<usize> {
            let base = p.cast::<u8>().sub(HDR);
            (base.cast::<usize>().read() == MAGIC).then(|| base.add(8).cast::<usize>().read())
        }

        unsafe fn tagged_free(p: *mut c_void) {
            if p.is_null() {
                return;
            }
            // Foreign block: leak rather than mis-free. See the module doc.
            let Some(size) = tagged_size(p) else { return };
            LIVE.fetch_sub(size, Ordering::Relaxed);
            dealloc(p.cast::<u8>().sub(HDR), layout(size + HDR));
        }

        pub unsafe extern "C" fn ts_malloc(size: usize) -> *mut c_void {
            tagged_alloc(size)
        }

        pub unsafe extern "C" fn ts_calloc(n: usize, size: usize) -> *mut c_void {
            let total = n.saturating_mul(size);
            let p = tagged_alloc(total);
            if !p.is_null() {
                std::ptr::write_bytes(p.cast::<u8>(), 0, total);
            }
            p
        }

        pub unsafe extern "C" fn ts_realloc(p: *mut c_void, size: usize) -> *mut c_void {
            if p.is_null() {
                return tagged_alloc(size);
            }
            let q = tagged_alloc(size);
            if q.is_null() {
                return q;
            }
            match tagged_size(p) {
                Some(old) => {
                    std::ptr::copy_nonoverlapping(p.cast::<u8>(), q.cast::<u8>(), old.min(size));
                    tagged_free(p);
                }
                // Foreign source: its length is unknown, so copying any of it could read past the
                // end. Zeroed is wrong but this path is unreachable in practice — tree-sitter does
                // not realloc across an allocator swap — and it is memory-safe, which mis-copying
                // would not be.
                None => std::ptr::write_bytes(q.cast::<u8>(), 0, size),
            }
            q
        }

        pub unsafe extern "C" fn ts_free(p: *mut c_void) {
            tagged_free(p);
        }
    }

    /// Re-derive `TREE_BYTES_PER_LINE` from a real corpus, and fail if it is no longer an upper
    /// bound.
    ///
    /// `#[ignore]`d and environment-driven, in the shape of `bloom`'s `#[40]` harness and for the
    /// same reason: the question only has an answer over a corpus of real source, and no corpus
    /// broad enough to calibrate a cross-grammar constant can live in this repository. Run it as:
    ///
    /// ```text
    /// TILTH_CALIBRATION_ROOT=<a tree of real source> \
    ///   cargo test --release calibrate_tree_bytes -- --ignored --exact \
    ///   lang::parse_budget::tests::calibrate_tree_bytes_per_line --nocapture
    /// ```
    ///
    /// It exists as a committed test rather than the throwaway crate that first produced these
    /// numbers so that the constant can be re-derived — after a grammar upgrade, or when a new
    /// language is added — and so the *shape* of the measurement is reviewable rather than asserted.
    /// The table in the module header came from exactly this procedure over 95 files and four
    /// grammars.
    ///
    /// **Run it alone.** `--exact` is not decoration: `LIVE` is process-wide, so a tree allocated by
    /// another test running concurrently lands in the same counter and inflates whichever file
    /// happens to be parsing.
    ///
    /// It asserts one thing — that `TREE_BYTES_PER_LINE` still bounds the corpus — because that is
    /// the property the budget's correctness rests on. Everything else it prints.
    #[test]
    #[ignore = "needs a corpus in TILTH_CALIBRATION_ROOT; prints the predictor table and checks the constant still bounds it"]
    fn calibrate_tree_bytes_per_line() {
        let Ok(root) = std::env::var("TILTH_CALIBRATION_ROOT") else {
            eprintln!("set TILTH_CALIBRATION_ROOT to a tree of real source; see this test's doc");
            return;
        };

        // Smaller files cannot move a ceiling, and their trees are dominated by fixed per-parse
        // overhead — a 500-byte file's tree is ~23 KB, which is 46 B/byte and pure noise for a
        // constant meant to bound large ones. The module header's table uses the same floor.
        const MIN_BYTES: usize = 10_000;

        unsafe {
            tree_sitter::set_allocator(
                Some(alloc_probe::ts_malloc),
                Some(alloc_probe::ts_calloc),
                Some(alloc_probe::ts_realloc),
                Some(alloc_probe::ts_free),
            );
        }

        struct Row {
            lang: crate::types::Lang,
            live: usize,
            bytes: usize,
            lines: usize,
            path: String,
        }
        let mut rows: Vec<Row> = Vec::new();

        for entry in ignore::WalkBuilder::new(&root).build().flatten() {
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                continue;
            }
            let path = entry.path();
            let crate::types::FileType::Code(lang) = crate::lang::detect_file_type(path) else {
                continue;
            };
            let Some(ts_lang) = crate::lang::outline::outline_language(lang) else {
                continue;
            };
            let Ok(src) = std::fs::read_to_string(path) else {
                continue;
            };
            if src.len() < MIN_BYTES {
                continue;
            }

            let before = alloc_probe::LIVE.load(Ordering::Relaxed);
            let Some(tree) = super::super::parse_masked(&src, Some(lang), &ts_lang) else {
                continue;
            };
            let live = alloc_probe::LIVE
                .load(Ordering::Relaxed)
                .saturating_sub(before);
            drop(tree);

            rows.push(Row {
                lang,
                live,
                bytes: src.len(),
                lines: src.lines().count().max(1),
                path: path.display().to_string(),
            });
        }

        assert!(
            rows.len() >= 8,
            "only {} files >= {MIN_BYTES} B with a grammar under {root}; too few to calibrate a \
             cross-grammar constant",
            rows.len()
        );

        // Each predictor's corpus maximum is what a conservative constant has to clear; the mean
        // over-estimate is what that costs in admitted parallelism. See the module header.
        let report = |name: &str, of: &dyn Fn(&Row) -> usize| {
            let ratios: Vec<f64> = rows
                .iter()
                .map(|r| r.live as f64 / of(r).max(1) as f64)
                .collect();
            let max = ratios.iter().copied().fold(f64::MIN, f64::max);
            let min = ratios.iter().copied().fold(f64::MAX, f64::min);
            let mean_over = ratios.iter().map(|x| max / x).sum::<f64>() / ratios.len() as f64;
            println!(
                "  {name:6} max={max:9.1} B  spread={:5.1}x  mean over-estimate={mean_over:5.2}x",
                max / min
            );
            max
        };

        let mut langs: Vec<String> = rows.iter().map(|r| format!("{:?}", r.lang)).collect();
        langs.sort_unstable();
        langs.dedup();
        println!(
            "\n{} files >= {MIN_BYTES} B, grammars: {}",
            rows.len(),
            langs.join(", ")
        );
        report("byte", &|r: &Row| r.bytes);
        let per_line = report("line", &|r: &Row| r.lines);
        report("token", &|r: &Row| {
            // The token proxy the module header compares against: each identifier run is one token,
            // each other non-whitespace byte is one.
            let mut n = 0usize;
            let mut in_word = false;
            for b in std::fs::read(&r.path).unwrap_or_default() {
                let word = b.is_ascii_alphanumeric() || b == b'_';
                if word && !in_word {
                    n += 1;
                } else if !word && !b.is_ascii_whitespace() {
                    n += 1;
                }
                in_word = word;
            }
            n
        });

        // The densest few, not just the maximum: one outlier is a fixture question, a cluster is a
        // property of the corpus, and the constant has to be set against the latter.
        rows.sort_by(|a, b| {
            (b.live as f64 / b.lines as f64).total_cmp(&(a.live as f64 / a.lines as f64))
        });
        println!("  densest per line:");
        for r in rows.iter().take(5) {
            println!(
                "    {:6.0} B/line  {:5} lines  {}",
                r.live as f64 / r.lines as f64,
                r.lines,
                r.path
            );
        }
        let worst = &rows[0];

        assert!(
            per_line <= TREE_BYTES_PER_LINE as f64,
            "TREE_BYTES_PER_LINE ({TREE_BYTES_PER_LINE}) no longer bounds this corpus: {} needs \
             {per_line:.0} B/line. The budget under-charges by that ratio, so raise the constant \
             (and re-measure the default ceiling, which was chosen against the old value).",
            worst.path
        );
    }
}
