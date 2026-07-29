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
/// `20_000` is inherited from `content.rs`, where it was measured and argued: selection omits
/// recency, so the retained set must be deep enough that recency can still promote a match onto
/// the page from *within* it. Recency is worth up to 100 points, so a match is at risk only when
/// its selection score is within 100 of the score at the cut. An earlier 500 was far too small and
/// deleted a freshly-edited subdirectory from results entirely.
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
pub(crate) const MAX_RETAINED: usize = 20_000;

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
}

impl ExactTallies {
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
        }
    }

    /// Offer one file's matches.
    ///
    /// `scorer` is per-thread — `Scorer` caches package roots and is `&mut` — so scoring happens
    /// off-lock by construction.
    pub(crate) fn offer_file(&self, file_matches: Vec<Match>, scorer: &mut Scorer<'_>) {
        if file_matches.is_empty() {
            return;
        }
        // Counted before reduction, so these are true totals and not what survived.
        self.offered
            .fetch_add(file_matches.len(), AtomicOrdering::Relaxed);
        let (mut d, mut i, mut t, mut u) = (0, 0, 0, 0);
        for m in &file_matches {
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
        // The heap still earns its place above the cap: without it one pathological file would hand
        // its entire match count to the merge loop and hold the mutex for all of it.
        let scored: Vec<Candidate> = file_matches
            .into_iter()
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

    /// Consume the sink and return the retained matches.
    ///
    /// Order is unspecified — `rank::sort` is a total order over these, so the caller's output
    /// does not depend on it. That independence is what the bound buys and is asserted in
    /// `rank`'s `sort_is_order_independent_for_matches_tied_on_path_and_line`.
    /// Exact per-facet tallies over everything offered.
    pub(crate) fn tallies(&self) -> ExactTallies {
        ExactTallies {
            definitions: self.t_defs.load(AtomicOrdering::Relaxed),
            implementations: self.t_impls.load(AtomicOrdering::Relaxed),
            tests: self.t_tests.load(AtomicOrdering::Relaxed),
            usages: self.t_usages.load(AtomicOrdering::Relaxed),
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

/// One bounded sink per target, for the multi-query walks.
///
/// Each target gets its own cap and its own lock. The single-`Mutex` version this replaces took one
/// acquisition per file for *all* queries, which was deliberate: every bucket then received the
/// file's matches as one contiguous block, and contiguity was what made ties deterministic. With
/// `rank::sort`'s key now total that is no longer needed, and per-bucket locks are strictly better
/// under contention — a dense file holds one target's lock instead of everyone's.
pub(crate) struct BoundedRetainSet {
    buckets: Vec<BoundedRetain>,
}

impl BoundedRetainSet {
    pub(crate) fn new(targets: usize, cap: usize) -> Self {
        Self {
            buckets: (0..targets).map(|_| BoundedRetain::new(cap)).collect(),
        }
    }

    /// Offer one file's matches for target `i`.
    ///
    /// Out-of-range `i` is ignored rather than panicking: the callers derive `i` from their own
    /// query list, so a mismatch is a bug in this file, not a reason to abort a walk that is
    /// otherwise returning correct results for every other target.
    pub(crate) fn offer_file(&self, i: usize, file_matches: Vec<Match>, scorer: &mut Scorer<'_>) {
        debug_assert!(i < self.buckets.len(), "target index {i} out of range");
        if let Some(b) = self.buckets.get(i) {
            b.offer_file(file_matches, scorer);
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
}
