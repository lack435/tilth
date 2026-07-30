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
/// The corpus maximum over files >= 10 KB was 1386 B/line (`callee_query.rs`, a file of long
/// `match` arms); 1536 rounds that up for margin without inflating the mean over-estimate much
/// beyond the measured 1.93x. See the module header for the full comparison against bytes and
/// tokens, and for why over-estimating is the safe direction.
///
/// If this is ever raised, the ceiling admits proportionally fewer concurrent parses and searches
/// over large-file trees get slower without getting more correct. If it is lowered below a
/// grammar's real cost, the ceiling is exceeded by that ratio — bounded overshoot, not a wrong
/// answer.
const TREE_BYTES_PER_LINE: usize = 1536;

/// Default ceiling on concurrently-held tree bytes.
///
/// 256 MB is chosen to be **inert at the default thread count and binding above it**, and that was
/// measured rather than intended. Peak working set and wall time, five reps per cell, fixtures of 60
/// files of 499 000 B:
///
/// ```text
///                          before          after        wall before   wall after
/// ordinary, default t   94.7-95.6 MB   94.6-95.1 MB    1.02-1.06 s   1.02-1.07 s
/// ordinary, t=32         448-452  MB    193-197  MB    0.78-0.83 s   0.80-0.86 s
/// dense, default t       221-222  MB    188-190  MB    2.74-3.41 s   2.99-3.49 s
/// dense, t=32           1091-1092 MB    231-232  MB    2.43-2.62 s   3.04-3.22 s
/// this repository       20.6-21.2 MB   20.4-22.0 MB    0.08-0.10 s   0.09    s
/// ```
///
/// So: **inert** on ordinary use and on ordinary large files at the default — every figure inside
/// the other's range — a **4.7x** reduction at 32 threads on the dense shape for ~24% wall, and
/// **2.3x** at 32 threads on ordinary source for no wall cost. On the dense shape it binds slightly
/// at the default too, taking ~33 MB off peak with overlapping wall ranges; that was not the intent
/// of "inert" and it is a small win rather than a cost, but the word is doing less work than it
/// looks.
///
/// Override with `TILTH_PARSE_BUDGET_MB`. Three cells say the knob is real rather than decorative,
/// all on the dense fixture at 32 threads:
///
/// ```text
///   0 (disabled)   1091-1092 MB   2.45-2.64 s    reproduces the unbudgeted figures exactly
///  64             101 -102  MB   9.33-9.40 s    tighter ceiling, cost is the serialisation
///   1              101 -102  MB   9.29-9.42 s    a ceiling 48x smaller than one tree
/// ```
///
/// The `0` row is the control that says the mechanism is off when asked. The `1` row is the one
/// worth reading twice: a single file's estimate is ~48 MB against a 1 MB ceiling, and the search
/// still completes with identical output, because `reserve` always admits when nothing else is in
/// flight. It degrades to serial parsing rather than hanging, and lands at the same peak as the 64 MB
/// ceiling because one tree is the floor either way.
const DEFAULT_BUDGET_MB: usize = 256;

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
}
