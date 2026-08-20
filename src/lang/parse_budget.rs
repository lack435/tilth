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
/// `estimate >= actual`. Measured with `examples/calibrate_parse_budget.rs`, which counts exact tree
/// bytes through tree-sitter's own allocator. Across every corpus tried — this repository, a large
/// external C++ tree, and deliberately adversarial shapes — the maximum is **92.5 B per source
/// byte**, so 128 carries ~1.4x margin.
///
/// **Source bytes, and not lines, because only bytes are bounded.** Tree size tracks node count,
/// node count is bounded by token count, and token count is bounded by bytes — so a per-byte
/// estimate has a ceiling that holds by construction. Lines do not: nodes-per-line is
/// bytes-per-line times nodes-per-byte, and bytes-per-line is unbounded.
///
/// ```text
/// shape                                       B/line    B/byte
/// this repository, densest (405 lines)           1675      22.5
/// generated C++                                  1238      13.9
/// ordinary C++ with long lines                   2997      27.3
/// minified JS behind a license banner          470651      53.1
/// one 90 KB line                              4578568      53.2
/// Rust data-as-code, one huge array literal    397322      92.5
/// ```
///
/// **A per-line estimator shipped first, and it bounded nothing.** 1536 B/line, then 2048 after a
/// committed harness found a 1675 B/line file the original corpus had missed. Both were wrong by
/// construction rather than by margin: review of that change measured 6143 B/line on an ordinary
/// hand-written C++ file, and the shapes above reach 4.5 million. Worse than the under-charge, a
/// minified bundle behind a preserved license banner — which passes the minified gate, because the
/// banner puts newlines in the first 2 KB — estimated 92 KB against a real 21 MB tree, leaving the
/// budget **entirely inert** on exactly the input it exists to bound: 765 MB against a 384 MB
/// ceiling, indistinguishable from disabling it.
///
/// The lesson is about the predictor rather than the constant, and it is the second time the corpus
/// question bit: a predictor validated on a corpus is validated on whatever that corpus happened to
/// contain, and both failures were silent and in the unsafe direction. `bytes` is the one candidate
/// whose bound does not depend on the corpus at all.
///
/// Raising this admits proportionally fewer concurrent parses, so searches over large-file trees get
/// slower without getting more correct — which is why the ceiling is re-derived whenever it moves.
/// Lowering it below a grammar's real cost exceeds the ceiling by that ratio.
const TREE_BYTES_PER_SOURCE_BYTE: usize = 128;

/// Maximum size of a file any parsing walk will read before skipping it.
///
/// Every AST gate in search shares this constant: the symbol walks, the caller/callee/deps bloom
/// walk (`search::bloom_walk::MAX_FILE_SIZE`), the outline parse cache (`cache::MAX_PARSE_BYTES`,
/// which also gates `search::scope`), and the in-result outline context. Content search does *not*
/// — it runs no parser and gates on its own `search::content::MAX_SEARCH_FILE_SIZE`.
///
/// Raised from 500 000 to 1 MB so hand-written large source files stay searchable. Real code sits
/// above the old gate — e.g. Unreal's `CharacterMovementComponent.cpp` (536 KB) and
/// `UnrealEngine.cpp` (692 KB) — while nearly everything past 1 MB is generated tables or vendored
/// header dumps, where a parse is pure cost and outline output is noise.
///
/// Coupled to `DEFAULT_BUDGET_MB`: the worst per-file tree estimate is
/// `MAX_PARSE_FILE_SIZE x TREE_BYTES_PER_SOURCE_BYTE`. See there for how the budget responds when
/// this moves.
pub(crate) const MAX_PARSE_FILE_SIZE: u64 = 1_000_000;

/// Default ceiling on concurrently-held tree bytes.
///
/// **Was derived, now deliberately under the derivation.** When the parse gate was 500 000 B the
/// worst per-file estimate was `500_000 x 128 = 64 MB`, and this was the smallest multiple of 64 MiB
/// admitting all `6` of `worker_threads`' worst-case parses at once (`6 x 64 = 384`), so it never
/// bound at the default thread count. Raising the gate to `MAX_PARSE_FILE_SIZE` (1 MB) doubled the
/// worst per-file estimate to `1_000_000 x 128 = 128 MB`, and the budget was **left at 384** rather
/// than moved to `6 x 128 = 768`.
///
/// That is the Option-B trade, and it is safe by construction: `reserve` always admits when nothing
/// is in flight, so no file is ever skipped for want of budget. A walk that parses more than three
/// near-gate-size files at once (`384 / 128 ≈ 3`) serialises the excess parses instead of raising
/// peak RSS — it pays latency, not memory, and only on the rare large-file-dense walk. The
/// 500 000 B era's inert-at-default property is traded for a bound that still holds at 1 MB without
/// doubling peak. The value stays coupled to the thread clamp in `util::worker_threads` and to
/// `MAX_PARSE_FILE_SIZE`; `the_worst_single_file_estimate_bounds_the_ceiling` fails if either
/// drifts.
///
/// Measured with the same binary throughout, `TILTH_PARSE_BUDGET_MB=0` as the unbudgeted reference so
/// nothing but the mechanism differs. Seven reps at the default thread count, five at 32. Fixtures of
/// 60 files: *ordinary* is real source at 39 B/line, *line-dense* is 16 B/line, *long-line* is a
/// minified bundle behind a license banner (41 lines, 387 KB) — the shape the per-line estimator
/// could not see at all.
///
/// ```text
///                        unbudgeted       bounded      wall unbudg   wall bounded
/// ordinary,   default   94.7-95.6 MB   94.5-95.1 MB   1.04-1.07 s   1.14-1.15 s
/// ordinary,   t=32       442-451  MB    110-111  MB   0.90-0.93 s   1.17-1.20 s
/// line-dense, default    220-222  MB    219-222  MB   2.74-3.56 s   3.04-4.13 s
/// line-dense, t=32      1089-1093 MB    263-266  MB   2.72-2.78 s   3.16-3.32 s
/// long-line,  default    266-268  MB    265-269  MB   2.83-2.89 s   2.82-2.89 s
/// long-line,  t=32        707    MB     267-272  MB   2.28-2.35 s   2.63-2.70 s
/// this repository       19.9-22.5 MB   19.9-22.5 MB   0.09-0.12 s   0.09-0.12 s
/// ```
///
/// **4.0x / 4.1x / 2.6x** at 32 threads, for 29% / 18% / 15% wall. At the default thread count every
/// bounded cell sits inside its unbudgeted range on all three shapes and on this repository — which
/// is the property the value was derived for, and the property the previous 256 MB did *not* have:
/// it bound the line-dense shape at the default, taking 33 MB off peak for wall it did not need to
/// spend.
///
/// Override with `TILTH_PARSE_BUDGET_MB`. Three cells say the knob is real rather than decorative,
/// all on the line-dense fixture at 32 threads:
///
/// ```text
///   0 (disabled)   1089-1093 MB   2.72-2.78 s   the reference row above
///  64              101 -102  MB  10.36-10.47 s  tighter ceiling; the cost is the serialisation
///   1              101 -102  MB  10.32-10.44 s  a ceiling 64x smaller than one file's estimate
/// ```
///
/// The `1` row is the one worth reading twice: one file's estimate is 64 MB against a 1 MB ceiling,
/// and the search still completes with identical output, because `reserve` always admits when nothing
/// else is in flight. It degrades to serial parsing rather than hanging, and lands at the same peak as
/// the 64 MB ceiling because one tree is the floor either way.
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
    /// `timeout.rs` stops waiting on an expired request but does not cancel its worker, so the worker
    /// keeps running with nothing consuming its result. A worker parked here lives *longer* than one
    /// that is not, because it now waits for space as well as for its own work.
    ///
    /// The reason it can be parked is worth stating precisely, because the obvious one is wrong: it is
    /// not that an abandoned request has low parallelism. Low self-parallelism means *less* pressure
    /// from itself, and a lone abandoned query never waits at all — `in_flight` reaches zero and the
    /// guard below admits it unconditionally, and the default ceiling is sized to admit all of one
    /// request's threads at once anyway. It waits only when *other, live* requests fill the budget.
    /// So the shape is: live work can park abandoned work, and there is no priority between them.
    ///
    /// The trade is still right — an abandoned worker holding a bounded amount of memory for longer
    /// beats eight of them holding an unbounded amount, which is what #61 estimates. But if #61 gains
    /// cooperative cancellation, checking a flag *here* is necessary and not sufficient: a thread
    /// inside `Condvar::wait` cannot notice a flag until something wakes it, so the canceller has to
    /// `notify_all` or this wait has to become `wait_timeout`. The wait is already bounded — a waiter
    /// exists only while someone holds a permit, and every permit is released by `Permit::drop`,
    /// including on panic — so this is about latency, not liveness.
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

/// Estimated tree bytes for `content`.
///
/// Free: the length is already known, so unlike the line-counting estimator this replaced there is
/// no scan of the file at all on the walk's hot path. `max(1)` keeps an empty file from reserving
/// zero and slipping the accounting.
fn estimate_bytes(content: &str) -> usize {
    content
        .len()
        .max(1)
        .saturating_mul(TREE_BYTES_PER_SOURCE_BYTE)
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

    /// The estimate is a function of length, and of nothing else.
    ///
    /// The predictor this replaced was per *line*, which two files of the same length and different
    /// line counts estimated differently — and that is exactly how it came to under-charge a
    /// one-line 90 KB file by four orders of magnitude. Same length must now mean same estimate,
    /// whatever the shape.
    #[test]
    fn the_estimate_tracks_length_and_not_shape() {
        let dense = "a
"
        .repeat(500); // 1000 bytes, 500 lines
        let one_line = "a".repeat(1000); // 1000 bytes, 1 line
        assert_eq!(dense.len(), one_line.len());
        assert_eq!(
            estimate_bytes(&dense),
            estimate_bytes(&one_line),
            "estimate still depends on line structure, so a long-line file under-charges"
        );
        assert_eq!(estimate_bytes(&dense), 1000 * TREE_BYTES_PER_SOURCE_BYTE);
    }

    /// The worst single-file estimate is the unit the ceiling is measured in, so pin the arithmetic.
    ///
    /// `MAX_PARSE_FILE_SIZE` (1 MB) gates every parsing walk, so no file can estimate above this.
    /// Under the Option-B sizing the default ceiling is deliberately *below* `worker_threads`'
    /// worst-case parses: a large-file-dense walk serialises the excess rather than raising peak RSS
    /// (see `DEFAULT_BUDGET_MB`). What must still hold is that the ceiling admits more than one
    /// worst-case file at once, so such a walk degrades to reduced parallelism, never to strictly
    /// serial parsing. If `MAX_PARSE_FILE_SIZE` or `TREE_BYTES_PER_SOURCE_BYTE` drift, re-derive.
    #[test]
    fn the_worst_single_file_estimate_bounds_the_ceiling() {
        let worst = MAX_PARSE_FILE_SIZE as usize * TREE_BYTES_PER_SOURCE_BYTE;
        assert_eq!(worst, 128_000_000, "worst per-file estimate moved");
        assert!(
            DEFAULT_BUDGET_MB * 1024 * 1024 >= 2 * worst,
            "the default ceiling ({DEFAULT_BUDGET_MB} MB) no longer admits two worst-case parses \
             ({} B), so large-file searches parse strictly serially",
            2 * worst
        );
    }

    /// An empty file still reserves something, so nothing slips the accounting at zero.
    #[test]
    fn an_empty_file_still_reserves() {
        assert_eq!(estimate_bytes(""), TREE_BYTES_PER_SOURCE_BYTE);
    }
}
