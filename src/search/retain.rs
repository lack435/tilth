//! Bounded, deterministic retention for parallel search walks.
//!
//! A walk that completes gives a stable answer, which is why the count-based early quits were
//! removed (#8, #18). But nothing then bounded how many matches were *kept*: every matching line
//! became a `Match` owning a `PathBuf` and the line text, and all of them survived to ranking.
//! Measured on a dense 400-file fixture with a match on every line, ~2.4M matches, over MCP with
//! `expand: 0`:
//!
//! ```text
//!                    peak RSS   wall
//! kind: "symbol"      1154 MB   13.2s
//! kind: "content"       92 MB    4.2s
//! ```
//!
//! Content was bounded in #30 and symbol was not, on identical input — 12.6x the memory, and that
//! is what this module exists to close. The amplifier is worse than the headline: `timeout.rs`
//! detaches a worker on expiry and it keeps allocating, with `MAX_ABANDONED_THREADS = 8` permitting
//! eight of those at once.
//!
//! **The memory win is unconditional; the wall-time win is not.** Same fixture, after:
//!
//! ```text
//!                      peak RSS        wall
//! default threads      85-90 MB     2.4-3.0s   (before: 1161-1164 MB, 10.3-10.7s)
//! TILTH_THREADS=1      22-23 MB    24.4-24.8s  (before: 1110 MB,      19.0-20.2s)
//! ```
//!
//! At default parallelism it is both smaller and faster. Single-threaded it trades ~25% more wall
//! for ~50x less memory, because the per-match scoring the bound needs is no longer hidden behind
//! other threads' I/O. That regime is worth stating rather than burying: the abandoned-worker
//! amplifier above is precisely the case where effective parallelism is low.
//!
//! Two properties every caller needs, and the reason this is shared code rather than copied:
//!
//! * **Determinism.** Selection uses `rank::Scorer::selection_score`, which omits the recency
//!   term, so what survives cannot depend on when the search ran. And the bound decides using
//!   only the candidate's own key — never a shared counter — because a counter read once per file
//!   callback over a parallel walk is exactly the nondeterminism #18 removed.
//! * **Not serialising the walk.** Each file reduces to its own best `cap` off-lock, then merges
//!   under one acquisition. Reducing straight into the shared heap holds the mutex across every
//!   comparison for the file and lets the reject buffer grow to the file's whole match count under
//!   that lock, which serialises the walk behind one dense file. Evicted matches are dropped
//!   *after* the guard releases, because `Match` owns a `PathBuf` and a `String` and dropping
//!   under the lock funnels two deallocations per reject through one mutex.
//!
//! **Retention is not the only term in peak RSS**, and #59 is the part the bound above does not
//! reach: a caller that builds a whole file's `Vec<Match>` before offering it holds
//! `matches_in_that_file` per walk thread, whatever `MAX_RETAINED` is. `FileOffer` closes that.
//! Callers push matches through it as they are found and it flushes every `OFFER_CHUNK`, so nothing
//! accumulates a file's whole match count. What survives is unchanged: the bound decides using only
//! a candidate's own key, so where the chunk boundaries fall cannot affect the retained set — which
//! is exactly why a *streaming* bound is safe here while the count-based early quits #18 removed
//! were not.
//!
//! Measured, three reps per cell, peak working set, `TILTH_THREADS` pinned — the mechanism is per
//! thread, so an unlabelled figure is not reproducible. Every fixture file is 499 000 B (31 187
//! 16-byte lines) so none is skipped by the 500 KB search gate and file size is held constant
//! throughout.
//!
//! **The isolated measurement.** 60 files, a match on every line — 1 871 220 matches — named `.txt`
//! rather than `.rs`. That one change removes tree-sitter from the picture entirely: `detect_file_type`
//! returns non-`Code`, so the definition walk parses nothing, while the usage walk still finds every
//! match. What is left is this module's term and almost nothing else.
//!
//! ```text
//!                  threads=1        threads=6         threads=32
//! before        32.6-33.0 MB    98.7-100.3 MB     452.7-456.1 MB
//! after         19.1-19.3 MB     25.5-26.0 MB      56.6-57.0 MB
//! ratio               1.7x             3.9x               8.0x
//! wall, before  23.3-23.5 s      3.14-3.72 s       1.73-2.06 s
//! wall, after   16.5-16.6 s      1.77-2.31 s       1.65-1.71 s
//! ```
//!
//! The ratio growing with thread count *is* the claim: the term removed is per thread, so the
//! saving has to scale with threads and does, 1.7x → 3.9x → 8.0x. On magnitude the mechanism
//! predicts `threads × 31 187 × ~250 B` — 7.8 / 47 / 250 MB — against 13.7 / 74 / 397 MB observed.
//! That is a consistent **1.6-1.8x** over the naive figure at every thread count, not agreement:
//! `Match` owns a `PathBuf` and a `String`, so each one is three allocations and the allocator's
//! per-allocation overhead is real. The scaling is what was predicted; the constant was not, and
//! quoting the prediction as a match would be the arithmetic working by luck.
//!
//! **The density sweep**, which is what actually establishes that the *match-count* term is gone.
//! Same 60 files, same 499 000 B each, same path depth — so the ten page slots always come from the
//! alphabetically-first file and the renderer's per-shown-file cost is held fixed. Only
//! matches-per-file varies. `.rs`, `kind: "symbol"`:
//!
//! ```text
//! matches/file      threads=6                    threads=32
//!                before        after           before          after
//!         10   205.2-206.1  204.8-221.6   1049.1-1050.7  1049.7-1050.1
//!        100   205.3-208.3  207.2-207.7   1050.8-1052.2  1050.9-1052.3
//!      1 000   209.0-215.9  217.8-218.4   1064.8-1070.2  1065.1-1066.6
//!     10 000   225.3-244.0  220.9-222.7   1180.9-1185.1  1090.5-1092.3
//!     31 187   267.3-290.4  219.5-221.7   1420.7-1433.6  1091.4-1092.5
//! ```
//!
//! Read down the columns. Over a 3 119x increase in density, `before` climbs 378 MB at 32 threads
//! and `after` climbs 42 MB — a 9x reduction in how much peak depends on density, and 5x at 6
//! threads. It does not reach zero, and should not: `after`'s residual slope is the
//! `OFFER_CHUNK × threads` buffer plus the retained heap filling up, ~13 MB predicted against 42
//! observed, the same 1.6-1.8x factor as above.
//!
//! **What the sweep also shows is a floor this change never touches**: ~1050 MB at 32 threads with
//! **600 matches in the entire tree**. That term is density-independent, and the `.txt` table above
//! attributes it — identical bytes at maximum density cost 57 MB when nothing parses them, so
//! ~1034 MB of the 1091 is tree-sitter, one tree per walk thread, ~35 MB for a 31 187-line file
//! (~70x the file's own bytes). It is bounded — `threads × MAX_SEARCH_FILE_SIZE × expansion` — but
//! bounded at roughly a gigabyte at 32 threads, so **#19's acceptance wording, "peak RSS bounded by
//! a configurable ceiling rather than by match count", is only half met by this module**: the "rather
//! than by match count" half is what the sweep demonstrates, and the "configurable ceiling" half
//! belongs to a term that is not in this file.
//!
//! For completeness, the in-situ `.rs` numbers, where the parse term above dominates and therefore
//! flatters nothing:
//!
//! ```text
//!                              threads=1        threads=6        threads=32
//! symbol, dense       before   48.0-48.3 MB   267-290 MB      1421-1434 MB
//!                     after    48.2-48.4 MB   220-222 MB      1091-1093 MB
//! symbol, defs-dense  before   55.6-55.7 MB   289-311 MB
//!                     after    47.6-47.8 MB   180-181 MB
//! content, dense      before   44.5-45.1 MB   91.6-95.1 MB
//!                     after    44.7-44.9 MB   45.1-45.4 MB
//! symbol, spread      before   23.3-23.6 MB   27.1-27.8 MB
//!                     after    23.5-25.0 MB   28.0-29.4 MB
//! wall, dense symbol  before   —              4.29-4.80 s     2.71-2.88 s
//!                     after    —              2.93-3.27 s     2.57-2.71 s
//! ```
//!
//! defs-dense is 60 files whose every 14-byte line is a definition — 2 138 520 definitions, and the
//! same number of usages, since each line both defines and mentions the symbol. It is the fixture
//! where *both* walks are dense and run concurrently under `rayon::join`, which is why it improves
//! most at 6 threads. Because every usage shares a line with a definition, the def/usage dedup
//! removes all of them and the rendered header is `2 138 520 matches (2 138 520 definitions, 0
//! usages)`.
//!
//! It read `2 118 520 usages` when these measurements were taken, and this note used to explain the
//! 20 000 shortfall — exactly `MAX_RETAINED` — as #60, whose overlap subtraction could only see
//! *retained* usages. #71 fixed that by counting the overlap during the walk, so the shortfall is
//! gone. Recorded rather than deleted because this fixture is the shape that made #60 maximal: if
//! the count ever drifts from the definition count again, it is that mechanism regressing.
//! spread is 25 000 two-line files, 50 000 matches, and is here to show the change costs nothing on
//! the shape where there is no per-file term to remove.
//!
//! Wall time improves wherever the term was large and is otherwise unchanged; no cell regressed.
//!
//! Callers keep their own exact `total_found` counters. Those only ever report, so they stay true
//! totals rather than counts of what was retained — the acceptance item #12 asked for.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Mutex;

use crate::search::rank::Scorer;
use crate::types::Match;

/// Default retention ceiling, shared by every bounded search path.
///
/// `20_000` is inherited from `content.rs`, where it was measured and argued (#30); the argument
/// moved here with the code (#62). Selection omits recency, so the retained set must be deep enough
/// that recency can still promote a match onto the page from *within* it. Recency is worth up to 100
/// points, so a match is at risk only when its selection score is within 100 of the score at the
/// cut.
///
/// The bound was 500 first, set from an assumption that recency was small against "scores in the
/// thousands". It is not, and the content path is where that is starkest: a content match scores
/// about **230** in total, because `is_definition` and `exact` are both false for every one of them,
/// which removes two 500-point terms and leaves `scope_proximity` (180 at depth 1) plus the
/// 50-point short-file bonus. Recency is ~43% of that whole score, and 100 points is **five
/// directory levels** of `scope_proximity`.
///
/// The consequence was measured, not theorised: 600 matches at the scope root aged 60 days, plus 300
/// in a freshly-edited directory five levels down, and the fresh directory vanished from the page
/// entirely — 10 of 10 entries before the bound, 0 after, while the header still reported all 900.
/// "Edit a subdirectory, then search for a common token" is an ordinary thing to do.
///
/// That 100-point window is only the whole story because **recency is the only term selection
/// omits**. A first version of the symbol path also passed `None` for `context`, which put
/// `context_proximity` — up to 175 points — outside the window too and widened the residual to 275
/// without saying so. Every caller now passes the real `context`. If another scoring term is ever
/// excluded from `selection_score`, this bound has to be re-argued, not just re-read.
///
/// At ~280 bytes per candidate this is ~5.6 MB, against the ~1.1 GB it replaces on the symbol
/// path, so the memory argument tolerates a bound two orders of magnitude above any display cap.
/// The residual is precise rather than absent: the page can differ from an unbounded search only
/// when more than `MAX_RETAINED` matches sit within 100 points above the dropped one.
///
/// ~5.6 MB is the ceiling for **one** sink. A multi-target query gets one sink per target per walk,
/// so the real ceiling is a multiple of this — see `BoundedRetainSet`, which states it.
pub(crate) const MAX_RETAINED: usize = 20_000;

/// How many matches a `FileOffer` holds before flushing to its sink.
///
/// This is the per-file peak-RSS term #59 filed, and it is the whole reason `FileOffer` exists:
/// before it, a caller built a file's complete `Vec<Match>`, so peak scaled with
/// `max_matches_in_one_file × walk_threads` no matter what `MAX_RETAINED` was.
///
/// Two costs pull in opposite directions:
///
/// * **Memory.** At ~280 bytes per match this is ~280 KB in flight per walk thread, against the
///   ~8 MB a 31 187-match file held before. It is deliberately well under `MAX_RETAINED` so a
///   chunk always takes `offer_batch`'s no-local-heap fast path.
/// * **Lock acquisitions.** One per chunk instead of one per file. Total time *under* the lock is
///   unchanged — it is proportional to the matches merged, not to how they are grouped — so this
///   trades one long hold for `n/1024` short ones, which is the better shape under contention. It
///   is emphatically not one acquisition per match, which is the contention this module's header
///   warns about: a chunk is three orders of magnitude coarser than that.
///
/// **Swept rather than assumed**, on the dense fixture in the module header (60 × 499 000 B, a match
/// on every 16-byte line), three reps per cell, output byte-identical at all three sizes:
///
/// ```text
///                threads=1                       threads=32
///  OFFER_CHUNK   peak            wall            peak              wall
///        128     48.0-48.2 MB    25.9-29.3 s     1076-1077 MB      2.97-3.19 s
///       1024     48.3-48.5 MB    28.3-28.8 s     1091-1093 MB      2.87-2.97 s
///       4096     48.4-48.5 MB    28.5-28.9 s     1141-1143 MB      2.80-3.17 s
/// ```
///
/// Those are an 8× step (128 → 1024) and a 4× one (1024 → 4096), and the peak cost of each is
/// **+15.5 MB (+1.4%)** and **+50 MB (+4.6%)** at 32 threads — not the same size, and the larger one
/// is on the shorter step. Against the naive `threads × Δchunk × 280 B` those steps predict +8.0 and
/// +27.5 MB, so both come in at ~1.8×, the same allocator factor the module header records. So peak
/// does rise with the constant, monotonically and roughly proportionally, but "linear" would be
/// asserting away a reproducible 1.8×.
///
/// Wall time does **not** separate the three: at one thread 128's range is wider than the other two
/// and contains them, and at 32 threads 128 and 1024 share only their endpoint 2.97 s while 4096
/// spans both. Three reps cannot resolve differences that small. That 32-thread column *is* the
/// contention case — 32 walk threads, one shared sink, 60 dense files — so the smallest size tested
/// was not penalised for its 8× acquisition count, which is the result that matters here.
///
/// So the honest summary is narrower than "immaterial": within 128-4096 the wall time is
/// indistinguishable and the memory difference is 1.4-4.6% of a residual this module does not own.
/// **1024 is not a measured win over 128** — 128 was 15.5 MB cheaper at 32 threads and no slower. It
/// is set for the property three reps cannot measure: staying three orders of magnitude coarser than
/// per-match offering, which is the contention shape this module's header warns about. The numbers
/// are recorded so a future reader can move it down on evidence rather than re-running this.
pub(crate) const OFFER_CHUNK: usize = 1024;

/// A match paired with the score that decides whether it survives.
///
/// `Ord` is **inverted** — a lower selection score compares *greater* — so a `BinaryHeap`, which
/// is a max-heap, has the worst retained candidate at its root and can evict in O(log n). Getting
/// this backwards silently retains the worst matches and is not visible in a small fixture.
///
/// Ties fall through to the same key `rank::sort` uses, so two candidates with equal scores are
/// ordered by data they carry rather than by arrival. Without that the heap's eviction choice
/// among equals would depend on walk scheduling, which is the property #18 established and the
/// reason `rank::sort`'s key was extended to a total order.
struct Candidate {
    score: i32,
    m: Match,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // "Greatest" must mean "worst retained", so `peek` is the eviction candidate.
        //
        // Only the score is inverted. `rank::sort` orders score *descending*, so the worst score is
        // the lowest and that has to compare greatest — hence `other.score.cmp(&self.score)`.
        //
        // The tie-breaks are **not** inverted, and this is the subtle half. `rank::sort` orders
        // path, line, `def_range` and text *ascending*, so among equal scores the match that sorts
        // last is the one with the largest path — which is therefore the one to evict, so it must
        // compare greatest, so the comparison runs in the same direction as `rank::sort`.
        //
        // Inverting these too was the first version, and it evicted the best ties instead of the
        // worst. It survived the unit tests because they discriminate on score, where the inversion
        // is correct. It showed up only on a fixture where every match scores the same and ties
        // decide the whole result: retained output no longer matched an unbounded search.
        other
            .score
            .cmp(&self.score)
            .then_with(|| self.m.path.cmp(&other.m.path))
            .then_with(|| self.m.line.cmp(&other.m.line))
            .then_with(|| self.m.def_range.cmp(&other.m.def_range))
            .then_with(|| self.m.text.cmp(&other.m.text))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Exact per-facet tallies over everything a sink was offered.
///
/// Three of the five facets `facets::facet_of` assigns are decided by the match alone —
/// `Implementation` (`is_definition && impl_target.is_some()`), `Definition` (`is_definition`) and
/// `Test` (`is_test_match`). Only the local/cross usage split consults a primary package derived
/// from the whole match set, so only that split is unrecoverable once retention clips.
///
/// Counting the other three here keeps them true. An earlier version derived all five from the
/// retained set and reported "2 tests" on a query that found 25 — the confidently wrong number the
/// renderer's own comment warns is worse than a useless one.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExactTallies {
    pub(crate) definitions: usize,
    pub(crate) implementations: usize,
    pub(crate) tests: usize,
    /// Non-test usages. Their local/cross split is the only part retention can lose.
    pub(crate) usages: usize,
    /// Definition lines that the usage walk's own matcher also matches — the exact number of
    /// usages `symbol::assemble`'s def/usage dedup will remove.
    ///
    /// **Not a match count, and deliberately outside `total()`.** Every other field counts
    /// matches this sink was offered; this one counts a property of them, so folding it in would
    /// break the "tallies account for every offered match" assertion in `finish`. Only the
    /// definition walk populates it — the usage walk cannot see a definition — so it is 0 on a
    /// usage sink and reading it from one is meaningless rather than merely zero.
    ///
    /// Counted during the walk because retention is what makes it unobservable afterwards: the
    /// overlap over the *retained* usages omits every collision the bound clipped, and
    /// `total_found` then over-reports by exactly that many (#60).
    pub(crate) usages_on_definition_lines: usize,
    /// The subset of `usages_on_definition_lines` whose definition line sits in the `Test` facet.
    ///
    /// `facets::facet_of` routes a usage in a test file to `Test`, so a removed collision has to
    /// come off *that* count and not off a usage bucket. Without the split, `totals.tests` reports a
    /// match `total_found` has already subtracted and the facets sum past the header.
    pub(crate) usages_on_test_definition_lines: usize,
}

impl ExactTallies {
    /// Matches offered, which is what `total_found` is built from.
    ///
    /// `usages_on_definition_lines` is excluded — see its doc comment.
    pub(crate) fn total(self) -> usize {
        self.definitions + self.implementations + self.tests + self.usages
    }
}

/// Shared bounded sink for a parallel walk.
pub(crate) struct BoundedRetain {
    heap: Mutex<BinaryHeap<Candidate>>,
    cap: usize,
    /// Exact count of matches offered, whether retained or evicted.
    ///
    /// The sink is the only thing that sees every match, so it is the right place to keep the true
    /// total. Without it, callers derive their reported counts from `into_matches().len()` and a
    /// bounded search silently under-reports — 2.4M matches announced as 20k. #19 asks for totals
    /// that stay true or are explicitly labelled clamped, and a header that quietly shrinks is the
    /// failure mode it names.
    ///
    /// Report-only and `Relaxed`: nothing reads it to decide what to retain, and it is read after
    /// the walk has joined.
    offered: AtomicUsize,
    /// Exact per-facet tallies, same discipline as `offered`. See `ExactTallies`.
    t_defs: AtomicUsize,
    t_impls: AtomicUsize,
    t_tests: AtomicUsize,
    t_usages: AtomicUsize,
    /// See `ExactTallies::usages_on_definition_lines`. Fed by `add_usages_on_definition_lines`
    /// rather than derived from the matches, because deciding it needs the line each definition
    /// sits on and only the definition walk has that.
    t_usage_collisions: AtomicUsize,
    /// See `ExactTallies::usages_on_test_definition_lines`.
    t_usage_collisions_in_tests: AtomicUsize,
}

impl BoundedRetain {
    pub(crate) fn new(cap: usize) -> Self {
        Self {
            heap: Mutex::new(BinaryHeap::new()),
            cap,
            offered: AtomicUsize::new(0),
            t_defs: AtomicUsize::new(0),
            t_impls: AtomicUsize::new(0),
            t_tests: AtomicUsize::new(0),
            t_usages: AtomicUsize::new(0),
            t_usage_collisions: AtomicUsize::new(0),
            t_usage_collisions_in_tests: AtomicUsize::new(0),
        }
    }

    /// Add one file's count of definition lines a usage match also lands on, and how many of those
    /// lines belong to the `Test` facet.
    ///
    /// Separate from the offer path because the count is not derivable from the matches: it needs
    /// the line each definition sits on. Report-only and `Relaxed`, exactly like `offered` —
    /// nothing reads either counter to decide what to retain, and both are read after the walk has
    /// joined, so they need no joint atomicity.
    pub(crate) fn add_usages_on_definition_lines(&self, n: usize, in_tests: usize) {
        debug_assert!(
            in_tests <= n,
            "test subset ({in_tests}) exceeds the total ({n})"
        );
        if n > 0 {
            self.t_usage_collisions
                .fetch_add(n, AtomicOrdering::Relaxed);
        }
        if in_tests > 0 {
            self.t_usage_collisions_in_tests
                .fetch_add(in_tests, AtomicOrdering::Relaxed);
        }
    }

    /// Offer an explicit batch of matches. **Tests only.**
    ///
    /// Every real caller pushes through `FileOffer` (#59) — taking a whole batch means the caller
    /// has already built the per-file `Vec<Match>` that is the peak-RSS term. What the tests need
    /// that `FileOffer` cannot give them is control over *where* the batch boundaries fall, which
    /// is the input to `grouping_matches_into_offers_cannot_change_what_is_retained`.
    #[cfg(test)]
    pub(crate) fn offer_file(&self, mut file_matches: Vec<Match>, scorer: &mut Scorer<'_>) {
        self.offer_batch(&mut file_matches, scorer);
    }

    /// Score a batch off-lock, then merge it under one acquisition.
    ///
    /// `scorer` is per-thread — `Scorer` caches package roots and is `&mut` — so scoring happens
    /// off-lock by construction.
    ///
    /// Drains rather than consumes, which is what lets `FileOffer` keep one allocation for a file
    /// however many chunks it flushes.
    fn offer_batch(&self, file_matches: &mut Vec<Match>, scorer: &mut Scorer<'_>) {
        if file_matches.is_empty() {
            return;
        }
        // Counted before reduction, so these are true totals and not what survived.
        self.offered
            .fetch_add(file_matches.len(), AtomicOrdering::Relaxed);
        let (mut d, mut i, mut t, mut u) = (0, 0, 0, 0);
        for m in file_matches.iter() {
            if m.is_definition && m.impl_target.is_some() {
                i += 1;
            } else if m.is_definition {
                d += 1;
            } else if crate::search::facets::is_test_match_for_totals(m) {
                t += 1;
            } else {
                u += 1;
            }
        }
        // One `fetch_add` per facet per file, not per match: the tally is folded locally first.
        self.t_defs.fetch_add(d, AtomicOrdering::Relaxed);
        self.t_impls.fetch_add(i, AtomicOrdering::Relaxed);
        self.t_tests.fetch_add(t, AtomicOrdering::Relaxed);
        self.t_usages.fetch_add(u, AtomicOrdering::Relaxed);

        // Score off-lock, and reduce off-lock only when there is something to reduce.
        //
        // The local heap uses the same `cap` as the shared one, so for any file with `cap` matches
        // or fewer it reduces *nothing* — it heapifies the file's matches and hands all of them
        // straight to the merge loop. Since `cap` is 20_000 and real files hold a handful of
        // matches, that was the case essentially always: pure overhead on every file. Skipping it
        // there is ~17% of single-threaded wall on a dense fixture, with byte-identical output.
        //
        // **The heap arm is now unreachable in a release build.** Every production caller offers
        // through `FileOffer`, whose batches are at most `OFFER_CHUNK` (1024), and every production
        // sink is built with `cap = MAX_RETAINED` (20 000) — so `scored.len() <= cap` always holds
        // and this is always the fast arm. It is kept, rather than deleted with the `cap`
        // comparison, because it is what makes `offer_batch` correct for a batch of any size: the
        // `#[cfg(test)] offer_file` relies on it, and so would any future caller whose per-file
        // match count is bounded by something other than a chunk. Without it one pathological batch
        // would hand its entire match count to the merge loop and hold the mutex for all of it.
        // `retention_is_bounded_by_the_cap` and `the_best_candidates_survive_not_the_worst` cover
        // it, so it is unshipped rather than untested.
        let scored: Vec<Candidate> = file_matches
            .drain(..)
            .map(|m| Candidate {
                score: scorer.selection_score(&m),
                m,
            })
            .collect();
        let local: Vec<Candidate> = if scored.len() <= self.cap {
            scored
        } else {
            let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(self.cap + 1);
            for cand in scored {
                if heap.len() < self.cap {
                    heap.push(cand);
                } else if heap.peek().is_some_and(|worst| cand < *worst) {
                    // Peek before pushing so a doomed candidate is never sifted in and back out.
                    heap.pop();
                    heap.push(cand);
                }
            }
            heap.into_vec()
        };

        // One acquisition, bounded by `cap` rather than by the file's match count.
        let mut evicted: Vec<Candidate> = Vec::new();
        {
            let mut heap = self
                .heap
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for cand in local {
                if heap.len() < self.cap {
                    heap.push(cand);
                } else if heap.peek().is_some_and(|worst| cand < *worst) {
                    if let Some(out) = heap.pop() {
                        evicted.push(out);
                    }
                    heap.push(cand);
                } else {
                    evicted.push(cand);
                }
            }
        }
        // Freed after the guard drops, not under it.
        drop(evicted);
    }

    /// Exact per-facet tallies over everything offered. See `ExactTallies`.
    pub(crate) fn tallies(&self) -> ExactTallies {
        ExactTallies {
            definitions: self.t_defs.load(AtomicOrdering::Relaxed),
            implementations: self.t_impls.load(AtomicOrdering::Relaxed),
            tests: self.t_tests.load(AtomicOrdering::Relaxed),
            usages: self.t_usages.load(AtomicOrdering::Relaxed),
            usages_on_definition_lines: self.t_usage_collisions.load(AtomicOrdering::Relaxed),
            usages_on_test_definition_lines: self
                .t_usage_collisions_in_tests
                .load(AtomicOrdering::Relaxed),
        }
    }

    /// Exact number of matches offered, independent of the cap.
    pub(crate) fn offered(&self) -> usize {
        self.offered.load(AtomicOrdering::Relaxed)
    }

    /// The retained matches plus the exact offered count, so a caller cannot accidentally report
    /// `len()` as the total.
    pub(crate) fn finish(self) -> (Vec<Match>, ExactTallies) {
        let tallies = self.tallies();
        debug_assert_eq!(
            tallies.total(),
            self.offered(),
            "facet tallies must account for every offered match"
        );
        (self.into_matches(), tallies)
    }

    /// Consume the sink and return the retained matches.
    ///
    /// Order is unspecified — `rank::sort` is a total order over these, so the caller's output does
    /// not depend on it. That independence is what the bound buys, and it is asserted in `rank`'s
    /// `sort_is_order_independent_for_matches_tied_on_path_and_line`.
    pub(crate) fn into_matches(self) -> Vec<Match> {
        self.heap
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .into_vec()
            .into_iter()
            .map(|c| c.m)
            .collect()
    }
}

/// A per-file staging buffer that flushes to a sink every `OFFER_CHUNK` matches.
///
/// This is the fix for #59. `offer_file` bounds what is *retained*; it cannot bound what exists at
/// once, because by the time it is called the caller has already built the file's whole
/// `Vec<Match>`. Push through this instead and the in-flight term is `OFFER_CHUNK` per walk thread
/// rather than the densest file's match count.
///
/// **Chunking cannot change which matches are retained**, and that is worth stating because a
/// per-file flush looks like the count-based early quits #18 removed. It is not one: nothing here
/// decides whether to keep walking or which matches to admit. `BoundedRetain` admits on a
/// candidate's own key, and `offer_batch`'s local reduction only ever discards candidates outside
/// the top `cap` of their own batch — which cannot be in the top `cap` of the union — so the shared
/// heap ends as the top `cap` of everything offered, however it was grouped. The removed quits
/// consulted a shared counter, which is what made them depend on scheduling.
///
/// Two things that claim would over-state if left unqualified:
///
/// * The retained *set* is grouping-independent; the heap's internal **order** is not.
///   `into_matches` hands back `into_vec()`, i.e. heap layout, which does depend on how the batches
///   fell. That is why `into_matches`'s doc declares its order unspecified and every caller re-sorts
///   through `rank::sort`, whose key is total.
/// * `Candidate::cmp` is a total order only up to candidates equal on all five levels, and the merge
///   admits on strict `<`. So among *fully* key-equal candidates, which one survives does follow
///   arrival, and therefore grouping. Reaching that needs two definitions at the same span with the
///   same `def_weight` and different `def_name` — which `SameSpanDedupe` deliberately keeps both of.
///   `rank::sort` sets the standard for this case and it applies here unchanged: two matches
///   agreeing on all of score, path, line, span and text are indistinguishable to a reader, so which
///   one is shown is not observable.
///
/// The exact counters see every match under any grouping, because they are incremented per offer
/// with the batch length and the batches partition what was pushed.
///
/// `finish` is not optional. Dropping a `FileOffer` with a partial buffer would silently discard
/// those matches *and* their contribution to the exact totals, so the buffer is asserted empty on
/// drop in debug builds.
pub(crate) struct FileOffer<'a> {
    sink: &'a BoundedRetain,
    buf: Vec<Match>,
}

impl<'a> FileOffer<'a> {
    pub(crate) fn new(sink: &'a BoundedRetain) -> Self {
        // No capacity reserved up front: most files in a walk contribute nothing, and a walk over a
        // large tree calls this once per file.
        Self {
            sink,
            buf: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, m: Match, scorer: &mut Scorer<'_>) {
        self.buf.push(m);
        if self.buf.len() >= OFFER_CHUNK {
            self.sink.offer_batch(&mut self.buf, scorer);
        }
        // The bound this type exists to provide, asserted rather than assumed.
        //
        // Removing the flush above is the one mutation that reverts #59 entirely, and no
        // equivalence test can catch it: chunking is *designed* to be invisible in the output, so a
        // test comparing retained sets or exact totals passes either way. That is not a gap in the
        // tests, it is a property of what they assert — the mechanism has to be observed directly,
        // and this is where. In debug builds it turns every test that streams more than
        // `OFFER_CHUNK` matches from one file into a detector, which is why the fixtures in
        // `search::tests` are sized to do that; `push_flushes_at_the_chunk_boundary` pins it
        // directly.
        debug_assert!(
            self.buf.len() < OFFER_CHUNK,
            "FileOffer buffered {} matches without flushing; the per-file bound is gone",
            self.buf.len()
        );
    }

    /// Flush whatever is left. Consumes `self` so a partial buffer cannot be forgotten.
    pub(crate) fn finish(mut self, scorer: &mut Scorer<'_>) {
        self.sink.offer_batch(&mut self.buf, scorer);
    }
}

impl Drop for FileOffer<'_> {
    fn drop(&mut self) {
        // Unreachable as written — `finish` consumes `self` and drains, so the only way here with a
        // non-empty buffer is a future `finish` that stops draining. Kept as a guard on that.
        //
        // The `panicking()` check is load-bearing, not defensive noise. A panic anywhere between
        // `new` and `finish` — including the `push` assert above — unwinds through this `Drop` with
        // the buffer still full, so without the check the assert fires *during unwinding*, and a
        // panic while unwinding aborts. That was not hypothetical: deleting `push`'s flush turned
        // the whole test binary into `STATUS_STACK_BUFFER_OVERRUN` with no named failure, which
        // hides the diagnosis behind a crash. Skipping the check while panicking lets the original
        // panic be the one reported.
        debug_assert!(
            self.buf.is_empty() || std::thread::panicking(),
            "FileOffer dropped with {} unflushed matches; call finish()",
            self.buf.len()
        );
    }
}

/// One bounded sink per target, for the multi-query walks.
///
/// Each target gets its own cap and its own lock. The single-`Mutex` version this replaces took one
/// acquisition per file for *all* queries, which was deliberate: every bucket then received the
/// file's matches as one contiguous block, and contiguity was what made ties deterministic. With
/// `rank::sort`'s key now total that is no longer needed, and per-bucket locks are strictly better
/// under contention — a dense file holds one target's lock instead of everyone's.
///
/// **The retention ceiling is per bucket, so a multi-target query's is a multiple of
/// `MAX_RETAINED`.** Stated here because it is easy to read `MAX_RETAINED`'s ~5.6 MB as the ceiling
/// for a whole search and it is not: symbol search runs two walks, each with one bucket per target,
/// so a 5-target comma query can retain 10 × 20 000 candidates — ~56 MB, ten times the figure
/// `MAX_RETAINED` budgets. That is deliberate rather than an oversight (#59 asked for it fixed or
/// documented honestly, and this is the honest half). A per-query-set cap would have to be shared
/// across buckets, which reintroduces the single lock this type exists to remove, and dividing the
/// cap by the target count would shrink each target's retained depth below the 100-point recency
/// window `MAX_RETAINED` is derived from — the residual it bounds is per target, because ranking is.
/// The multiplier is bounded and small: targets are capped at 5 by the tool schema.
pub(crate) struct BoundedRetainSet {
    buckets: Vec<BoundedRetain>,
}

impl BoundedRetainSet {
    pub(crate) fn new(targets: usize, cap: usize) -> Self {
        Self {
            buckets: (0..targets).map(|_| BoundedRetain::new(cap)).collect(),
        }
    }

    /// Target `i`'s sink, for callers that stream through a `FileOffer` per target.
    ///
    /// `None` is unreachable — `buckets.len() == queries.len()` and both callers derive `i` from
    /// that same list — so the `debug_assert` is the real report. Returning `None` rather than
    /// panicking in release is deliberate: a mismatch would be a bug in the caller's own query
    /// list, not a reason to abort a walk that is returning correct results for every other
    /// target.
    pub(crate) fn bucket(&self, i: usize) -> Option<&BoundedRetain> {
        debug_assert!(i < self.buckets.len(), "target index {i} out of range");
        self.buckets.get(i)
    }

    /// `BoundedRetain::add_usages_on_definition_lines` for target `i`. Out-of-range `i` is ignored
    /// rather than panicking, for the same reason `bucket` returns an `Option`: a mismatch is a bug
    /// in the caller, not a reason to abort a walk returning correct results for every other target.
    pub(crate) fn add_usages_on_definition_lines(&self, i: usize, n: usize, in_tests: usize) {
        debug_assert!(i < self.buckets.len(), "target index {i} out of range");
        if let Some(b) = self.buckets.get(i) {
            b.add_usages_on_definition_lines(n, in_tests);
        }
    }

    /// Per-target retained matches paired with each target's exact offered count.
    pub(crate) fn finish(self) -> Vec<(Vec<Match>, ExactTallies)> {
        self.buckets
            .into_iter()
            .map(BoundedRetain::finish)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::time::SystemTime;

    fn m(path: &str, line: u32) -> Match {
        Match {
            path: PathBuf::from(path),
            line,
            text: format!("hit at {line}"),
            is_definition: false,
            exact: true,
            file_lines: 100,
            mtime: SystemTime::UNIX_EPOCH,
            def_range: None,
            def_name: None,
            def_weight: 0,
            impl_target: None,
        }
    }

    /// Identity of a retained set, order-insensitive and covering every tie-break level.
    fn key_of(v: Vec<Match>) -> Vec<(PathBuf, u32, Option<(u32, u32)>, String)> {
        let mut k: Vec<_> = v
            .into_iter()
            .map(|x| (x.path, x.line, x.def_range, x.text))
            .collect();
        k.sort();
        k
    }

    fn scorer<'a>(scope: &'a Path) -> Scorer<'a> {
        Scorer::new("hit", scope, None)
    }

    /// The whole point: retention is capped however many matches arrive.
    #[test]
    fn retention_is_bounded_by_the_cap() {
        let scope = Path::new(".");
        let sink = BoundedRetain::new(10);
        let mut sc = scorer(scope);
        for f in 0..20 {
            let batch: Vec<Match> = (0..500).map(|i| m(&format!("f{f}.rs"), i)).collect();
            sink.offer_file(batch, &mut sc);
        }
        assert_eq!(sink.into_matches().len(), 10);
    }

    /// Feeding the same files in a different order must retain the same set.
    ///
    /// This is the property that lets a parallel walk use this at all: thread scheduling decides
    /// arrival order, so if the retained set depended on it the answer would vary run to run —
    /// the nondeterminism #18 removed. Fails if `Candidate`'s tie-break is dropped back to score
    /// alone, because then eviction among equal scores follows insertion.
    #[test]
    fn retained_set_does_not_depend_on_arrival_order() {
        let scope = Path::new(".");
        let files: Vec<Vec<Match>> = (0..8)
            .map(|f| (0..50).map(|i| m(&format!("f{f}.rs"), i)).collect())
            .collect();

        let forward = BoundedRetain::new(37);
        let mut sc = scorer(scope);
        for batch in files.clone() {
            forward.offer_file(batch, &mut sc);
        }

        let reverse = BoundedRetain::new(37);
        let mut sc = scorer(scope);
        for batch in files.into_iter().rev() {
            reverse.offer_file(batch, &mut sc);
        }

        let key = |v: Vec<Match>| {
            let mut k: Vec<(PathBuf, u32)> = v.into_iter().map(|m| (m.path, m.line)).collect();
            k.sort();
            k
        };
        assert_eq!(
            key(forward.into_matches()),
            key(reverse.into_matches()),
            "retained set depends on arrival order, so a parallel walk would vary run to run"
        );
    }

    /// The heap must keep the *best* candidates, not the worst — the inverted `Ord` is easy to get
    /// backwards and a small fixture will not show it.
    #[test]
    fn the_best_candidates_survive_not_the_worst() {
        let scope = Path::new("src");
        let sink = BoundedRetain::new(3);
        let mut sc = scorer(scope);

        // Deeper paths score lower on `scope_proximity`, so the shallow file should win.
        let mut batch = vec![
            m("src/near.rs", 1),
            m("src/near.rs", 2),
            m("src/near.rs", 3),
        ];
        batch.extend((0..20).map(|i| m("src/a/b/c/d/e/far.rs", i)));
        let mut scores: Vec<i32> = batch.iter().map(|x| sc.selection_score(x)).collect();
        scores.sort_unstable();
        // Guard the fixture itself: if every candidate scores the same there is nothing to rank.
        assert_ne!(
            scores.first(),
            scores.last(),
            "fixture does not discriminate; it cannot test selection"
        );

        sink.offer_file(batch, &mut sc);
        let kept = sink.into_matches();
        assert_eq!(kept.len(), 3);
        assert!(
            kept.iter().all(|k| k.path.ends_with("near.rs")),
            "kept the low-scoring candidates: {:?}",
            kept.iter().map(|k| k.path.clone()).collect::<Vec<_>>()
        );
    }

    /// When candidates score the same, the **tie-break direction** decides everything, and it has to
    /// agree with `rank::sort` at *every* level.
    ///
    /// One mixed fixture does not establish this. The first version of `Candidate::cmp` inverted the
    /// tie-breaks along with the score, keeping the worst ties instead of the best; the first version
    /// of this test then pinned only `line`, and a mixed fixture pinned only `path` and `line` —
    /// because a level is exercised only when the retention cut falls *between two candidates that
    /// tie on every level above it*. Inverting `def_range` or `text` broke nothing.
    ///
    /// So each level gets its own fixture in which only that level varies, with a cap that cuts
    /// through the middle of the tied run. Asserted against `rank::sort` on the same input, because
    /// the property is "retention keeps what an unbounded search would have shown" — the reference has
    /// to be the real ranker, not a hardcoded list.
    #[test]
    fn among_equal_scores_every_tie_break_level_agrees_with_the_ranker() {
        let scope = Path::new(".");

        // Each entry varies exactly one level and holds the others fixed.
        let fixtures: Vec<(&str, Vec<Match>)> = vec![
            (
                "path",
                (0..12).map(|f| m(&format!("src/f{f:02}.rs"), 1)).collect(),
            ),
            ("line", (0..12).map(|l| m("src/one.rs", l)).collect()),
            (
                "def_range",
                (0..12)
                    .map(|i| {
                        let mut x = m("src/one.rs", 1);
                        x.def_range = Some((1, i));
                        x
                    })
                    .collect(),
            ),
            (
                "text",
                (0..12)
                    .map(|i| {
                        let mut x = m("src/one.rs", 1);
                        x.def_range = Some((1, 9));
                        x.text = format!("variant {i:02}");
                        x
                    })
                    .collect(),
            ),
        ];

        for (level, all) in fixtures {
            let mut sc = scorer(scope);
            let mut distinct: Vec<i32> = all.iter().map(|x| sc.selection_score(x)).collect();
            distinct.sort_unstable();
            distinct.dedup();
            assert_eq!(
                distinct.len(),
                1,
                "{level} fixture does not hold score constant, so it cannot test a tie-break"
            );

            // Cuts through the middle of the tied run, so this level decides the boundary.
            let cap = 5;
            let sink = BoundedRetain::new(cap);
            let mut sc = scorer(scope);
            sink.offer_file(all.clone(), &mut sc);
            let kept = key_of(sink.into_matches());

            let mut reference = all;
            crate::search::rank::sort(&mut reference, "hit", scope, None);
            reference.truncate(cap);
            let want = key_of(reference);

            assert_eq!(
                kept, want,
                "retention disagrees with `rank::sort` when the `{level}` tie-break decides"
            );
        }
    }

    #[test]
    fn empty_offer_is_a_no_op() {
        let scope = Path::new(".");
        let sink = BoundedRetain::new(5);
        let mut sc = scorer(scope);
        sink.offer_file(Vec::new(), &mut sc);
        assert!(sink.into_matches().is_empty());
    }

    /// The retained set must not depend on how a file's matches are grouped into offers.
    ///
    /// This is the property that makes `FileOffer` safe (#59). Chunking is a memory decision, and it
    /// would be a correctness bug if it were also a selection decision — the distinction between
    /// this and the count-based early quits #18 removed rests entirely on it. So every partition of
    /// the same matches must retain the same set *and* report the same exact totals: one batch, the
    /// real `FileOffer` chunking, singletons, and uneven runs that straddle no boundary neatly.
    ///
    /// Fails if `BoundedRetain` ever consults anything but a candidate's own key — a per-offer
    /// budget, a fill fraction, an arrival counter — because each of those makes the answer a
    /// function of where the flushes fell.
    #[test]
    fn grouping_matches_into_offers_cannot_change_what_is_retained() {
        let scope = Path::new("src");
        let cap = 50;

        // Mixed depths so scores discriminate, and enough matches that the cap really clips.
        //
        // The count also exceeds `OFFER_CHUNK`, so arm 1 below crosses real flush boundaries and
        // ends on a partial tail. At 400 it did not: with a chunk of 1024 every match sat in the
        // buffer until `finish`, so the arm compared one batch against one batch and would have
        // passed with `push`'s flush deleted.
        let all: Vec<Match> = (0..OFFER_CHUNK * 2 + 7)
            .map(|i| {
                let depth = i % 4;
                let dir = "a/".repeat(depth);
                m(&format!("src/{dir}f{i:05}.rs"), i as u32 % 7)
            })
            .collect();
        assert!(
            all.len() > cap,
            "fixture must exceed the cap or nothing clips"
        );
        assert!(
            all.len() > OFFER_CHUNK,
            "fixture must exceed OFFER_CHUNK or arm 1 never reaches a flush"
        );

        // Reference: the whole file in one offer, which is what every caller did before #59.
        let one = BoundedRetain::new(cap);
        let mut sc = scorer(scope);
        one.offer_file(all.clone(), &mut sc);
        let want_key = key_of(one.into_matches());

        // Arm 1: the real thing — pushed one at a time through `FileOffer`, so the flushes fall
        // wherever `OFFER_CHUNK` puts them.
        let streamed = BoundedRetain::new(cap);
        let mut sc = scorer(scope);
        let mut offer = FileOffer::new(&streamed);
        for x in all.clone() {
            offer.push(x, &mut sc);
        }
        offer.finish(&mut sc);
        assert_eq!(
            key_of(streamed.into_matches()),
            want_key,
            "FileOffer retained a different set from a single batch"
        );

        // Arms 2..n: explicit partitions, including ones that do not divide the input evenly.
        for group in [1usize, 3, 7, 50, 128, 399] {
            let sink = BoundedRetain::new(cap);
            let mut sc = scorer(scope);
            for chunk in all.chunks(group) {
                sink.offer_file(chunk.to_vec(), &mut sc);
            }
            assert_eq!(
                sink.offered(),
                all.len(),
                "offered count wrong at group size {group}"
            );
            assert_eq!(
                sink.tallies().total(),
                all.len(),
                "facet tallies lost matches at group size {group}"
            );
            assert_eq!(
                key_of(sink.into_matches()),
                want_key,
                "retained set depends on offer grouping at group size {group}"
            );
        }
    }

    /// `push` must flush at the chunk boundary — the mechanism, not its harmlessness.
    ///
    /// Everything else about `FileOffer` is asserted through equivalence: same retained set, same
    /// exact totals, byte-identical output. None of that can catch deleting the flush from `push`,
    /// because chunking is *designed* to leave all three unchanged — an equivalence test passes
    /// most loudly when the thing it compares has been removed. So the flush has to be observed
    /// happening, and `offered()` is the observation: it counts what has reached the sink, so
    /// reading it mid-file distinguishes "flushed at 1024" from "still buffering the whole file".
    ///
    /// `offered()` takes `&self`, so it is readable while `offer` holds its borrow of the sink.
    #[test]
    fn push_flushes_at_the_chunk_boundary() {
        let scope = Path::new("src");
        let sink = BoundedRetain::new(MAX_RETAINED);
        let mut sc = scorer(scope);
        let mut offer = FileOffer::new(&sink);

        for i in 0..OFFER_CHUNK - 1 {
            offer.push(m("src/one.rs", i as u32), &mut sc);
        }
        assert_eq!(
            sink.offered(),
            0,
            "flushed early — the buffer should still be filling one match short of the boundary"
        );

        offer.push(m("src/one.rs", (OFFER_CHUNK - 1) as u32), &mut sc);
        assert_eq!(
            sink.offered(),
            OFFER_CHUNK,
            "push did not flush at the chunk boundary, so nothing bounds the per-file buffer"
        );

        // And it keeps flushing rather than only doing so once.
        for i in 0..OFFER_CHUNK {
            offer.push(m("src/one.rs", (OFFER_CHUNK + i) as u32), &mut sc);
        }
        assert_eq!(
            sink.offered(),
            OFFER_CHUNK * 2,
            "the second chunk did not flush"
        );

        offer.finish(&mut sc);
    }

    /// `finish` must flush a tail shorter than `OFFER_CHUNK`.
    ///
    /// The failure this guards is silent in both directions: the tail matches vanish from the page
    /// *and* from the exact totals, so the header agrees with a result that is missing them.
    #[test]
    fn finish_flushes_a_partial_chunk() {
        let scope = Path::new("src");
        let sink = BoundedRetain::new(MAX_RETAINED);
        let mut sc = scorer(scope);
        let n = OFFER_CHUNK + 7;
        let mut offer = FileOffer::new(&sink);
        for i in 0..n {
            offer.push(m("src/one.rs", i as u32), &mut sc);
        }
        offer.finish(&mut sc);
        assert_eq!(sink.offered(), n, "tail was not counted");
        assert_eq!(sink.into_matches().len(), n, "tail was not retained");
    }
}
