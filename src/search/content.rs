use std::collections::BinaryHeap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use super::file_metadata;

use crate::error::TilthError;
use crate::search::rank;
use crate::types::{FacetTotals, Match, SearchResult};
use grep_regex::RegexMatcher;
use grep_searcher::sinks::UTF8;
use grep_searcher::Searcher;

const MAX_MATCHES: usize = 10;
const FULL_MAX_MATCHES: usize = 100;
const MAX_SEARCH_FILE_SIZE: u64 = 500_000;

/// Candidates kept for ranking. Everything past this is counted and dropped.
///
/// Well above `FULL_MAX_MATCHES` (100) on purpose: the display cap decides what is *shown*,
/// this decides what is *available to rank*, and the gap is the room recency has to reorder
/// the page. See `rank::Scorer` for why selection deliberately ignores recency and what the
/// residual risk is.
const MAX_RETAINED: usize = 500;

// This walk used to stop once a shared `AtomicUsize` crossed `EARLY_QUIT_THRESHOLD`
// (30, or 300 under `--full`), which made content search **non-deterministic** for the
// same reason as the symbol and caller walks: parallel walk, counter read once per file
// callback, many matches addable per file. Measured on a 176k-file C++ tree: three
// distinct renderings in six identical runs.
//
// It hid better here than on the symbol path, because the header reported the display
// cap rather than a total, so the instability was only visible by diffing full output.
// Both halves of that are fixed: the walk completes, and `facet_totals` below is
// computed on the pre-cap set so the rendered `shown/total` labels are true totals.
//
// Cost of completing the walk, measured over MCP `tilth_search` with `kind: "content"`
// and `expand: 0` on that tree. Three reps each:
//
//   query                            bounded                       walk completes
//   literal, 137 matches             0.59-0.79s, 3 of 3 distinct   2.23-2.41s / 27 MB
//   literal, 34290 matches           0.043s,     3 of 3 distinct   4.56-4.74s / 34 MB
//
// Note what the bounded column bought on the second row: 43ms for an answer that was
// different every single time it was asked. An agent that asks the same question twice
// and gets two different answers cannot reason about either. The honest cost is that
// a search for something common goes from instant to ~4.6s, because "how many are
// there" cannot be answered without looking.
//
// Neither row was the worst case, and for a while the worst case was unbounded. 34290
// matches over 176k files is only 0.2 matches per file; the per-file size gate bounds bytes
// rather than matches, and `is_minified_by_content` only inspects the first 2 KB, so a 499 KB
// file of short lines is searched in full and can contribute ~150k matches by itself. On a
// deliberately dense 400-file, 49 MB fixture — a match on every line, ~2.4M matches — that
// reached 420-462 MB resident.
//
// Retention is now bounded at `MAX_RETAINED` candidates, selected by rank rather than by
// arrival, while the *counts* stay exact via two atomics. So the reported totals are still
// true totals, which is the half of this that a naive cap gets wrong.
//
// The three things that make a bound safe here are worth naming together, because each was a
// bug on some other path in this codebase first:
//
//   * The counters only ever report. Nothing reads them to decide whether to keep walking —
//     a count that gates a parallel walk cannot be deterministic.
//   * Selection uses `rank::Scorer`, which omits the recency term, so the wall clock cannot
//     decide which matches exist. A clock deciding content is the `overview::hot_files` bug.
//   * The selection key is a total order (score, then path, then line), so a truncation can
//     never be resolved by thread arrival order. That is the `tilth_files` bug.

/// Content search using ripgrep crates. Literal by default, regex if `is_regex`.
pub fn search(
    pattern: &str,
    scope: &Path,
    is_regex: bool,
    context: Option<&Path>,
    glob: Option<&str>,
    full: bool,
) -> Result<SearchResult, TilthError> {
    let max_matches = if full { FULL_MAX_MATCHES } else { MAX_MATCHES };
    let matcher = if is_regex {
        RegexMatcher::new(pattern)
    } else {
        RegexMatcher::new(&regex_syntax::escape(pattern))
    }
    .map_err(|e| TilthError::InvalidQuery {
        query: pattern.to_string(),
        reason: e.to_string(),
    })?;

    // Max-heap ordered so its top is the *worst* retained candidate — see `Candidate`.
    let matches: Mutex<BinaryHeap<Candidate>> = Mutex::new(BinaryHeap::new());
    let total_found = AtomicUsize::new(0);
    let test_matches = AtomicUsize::new(0);

    let walker = super::walker(scope, glob)?;

    walker.run(|| {
        let matcher = &matcher;
        let matches = &matches;
        let total_found = &total_found;
        let test_matches = &test_matches;
        // One scorer per worker thread. It memoises package-root lookups, and omitting the
        // recency term makes it independent of when it runs — so two threads scoring the
        // same match always agree, and so do two runs.
        let mut scorer = rank::Scorer::new(pattern, scope, context);

        Box::new(move |entry| {
            let Ok(entry) = entry else {
                return ignore::WalkState::Continue;
            };

            if !entry.file_type().is_some_and(|ft| ft.is_file()) {
                return ignore::WalkState::Continue;
            }

            let path = entry.path();

            // Skip files that look minified by filename — `.min.js`, `app-min.css`.
            if path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(crate::lang::detection::is_minified_by_name)
            {
                return ignore::WalkState::Continue;
            }

            // Skip oversized files — tree-sitter and ripgrep shouldn't spend time on minified bundles
            let file_size = match std::fs::metadata(path) {
                Ok(meta) => {
                    if meta.len() > MAX_SEARCH_FILE_SIZE {
                        return ignore::WalkState::Continue;
                    }
                    meta.len()
                }
                Err(_) => 0,
            };

            // Read the file once. Use `search_slice` instead of `search_path`
            // so the minified-check (when triggered) and the actual search
            // share a single kernel read — no double I/O, no TOCTOU window
            // between the heuristic and the search.
            let Ok(bytes) = std::fs::read(path) else {
                return ignore::WalkState::Continue;
            };

            // Catch unmarked minified bundles in the 100KB–500KB range.
            if file_size >= crate::lang::detection::MINIFIED_CHECK_THRESHOLD
                && crate::lang::detection::is_minified_by_content(&bytes)
            {
                return ignore::WalkState::Continue;
            }

            let (file_lines, mtime) = file_metadata(path);

            let mut file_matches = Vec::new();
            let mut searcher = Searcher::new();

            let _ = searcher.search_slice(
                matcher,
                &bytes,
                UTF8(|line_num, line| {
                    file_matches.push(Match {
                        path: path.to_path_buf(),
                        line: line_num as u32,
                        text: line.trim_end().to_string(),
                        is_definition: false,
                        exact: false,
                        file_lines,
                        mtime,
                        def_range: None,
                        def_name: None,
                        def_weight: 0,
                        impl_target: None,
                    });
                    Ok(true)
                }),
            );

            if !file_matches.is_empty() {
                // Counts first, and they are exact regardless of what is retained below.
                // Every content match is a usage, so the only facet split that can occur is
                // test-vs-other, decidable per match from its path and text. See the note
                // on `facet_totals` after the walk for why nothing else is reachable.
                let tests_here = file_matches
                    .iter()
                    .filter(|m| super::facets::is_test_match_for_totals(m))
                    .count();
                total_found.fetch_add(file_matches.len(), Ordering::Relaxed);
                test_matches.fetch_add(tests_here, Ordering::Relaxed);

                // Score outside the lock. `Scorer` omits the recency term, so which matches
                // are retained cannot depend on the wall clock.
                let scored: Vec<Candidate> = file_matches
                    .into_iter()
                    .map(|m| Candidate {
                        score: scorer.selection_score(&m),
                        m,
                    })
                    .collect();

                // Reduce *inside* the walk, which is the whole point. A first version of
                // this collected every scored match and reduced afterwards; output was
                // correct and it was faster, but peak memory did not move at all — the peak
                // is reached before ranking ever starts, so a bound applied after the walk
                // bounds nothing.
                let mut evicted: Vec<Candidate> = Vec::new();
                {
                    let mut heap = matches
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    for cand in scored {
                        if heap.len() < MAX_RETAINED {
                            heap.push(cand);
                        } else if heap.peek().is_some_and(|worst| cand < *worst) {
                            // Better than the worst kept: swap it in. Peeking first avoids
                            // sifting a doomed candidate in and back out, and keeps the
                            // rejected one out of the heap entirely.
                            if let Some(out) = heap.pop() {
                                evicted.push(out);
                            }
                            heap.push(cand);
                        } else {
                            evicted.push(cand);
                        }
                    }
                }
                // Freed after the guard drops. `Match` owns a `PathBuf` and a `String`, so
                // dropping under the lock would serialise two deallocations per rejected
                // match through one mutex — the mistake found in `glob::search`'s review.
                drop(evicted);
            }

            ignore::WalkState::Continue
        })
    });

    let heap = matches
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    // True totals, read once after `walker.run` has joined every thread. These are counters
    // that only ever *report* — nothing above reads them to decide whether to keep walking,
    // which is the distinction that made the old match-count cutoffs non-deterministic.
    let total = total_found.load(Ordering::Relaxed);
    let tests = test_matches.load(Ordering::Relaxed);

    // Best-first by the selection key. `rank::sort` then re-orders with recency included;
    // this ordering exists so the input to it is a function of the tree, not of arrival.
    let mut all_matches: Vec<Match> = heap.into_sorted_vec().into_iter().map(|c| c.m).collect();

    rank::sort(&mut all_matches, pattern, scope, context);

    // Per-facet totals. Content search used to return `FacetTotals::default()`, i.e. all
    // zeros, which made `count_label` print a bare `10` and suppressed every hidden-count
    // tail — a query with 34290 matches rendered exactly like one with 10.
    //
    // These are computed from the counters rather than from the retained set, and they are
    // still *exact*, because content search can only ever populate two of the five buckets.
    // Every match has `is_definition: false`, so `facets::primary_package` finds no primary
    // definition, `is_same_package` short-circuits to false, and every non-test match lands
    // in `usages_cross`. There is nothing for a bound to make approximate here.
    let facet_totals = FacetTotals {
        definitions: 0,
        implementations: 0,
        tests,
        usages_local: 0,
        usages_cross: total - tests,
    };

    all_matches.truncate(max_matches);

    Ok(SearchResult {
        query: pattern.to_string(),
        scope: scope.to_path_buf(),
        matches: all_matches,
        total_found: total,
        definitions: 0,
        usages: total,
        facet_totals,
    })
}

/// A match plus its time-independent selection score, ordered so that **greater means
/// worse**.
///
/// That inversion is what lets a `BinaryHeap` — a max-heap — hold the *best* `MAX_RETAINED`
/// candidates: its top is the worst kept, so it is the one to evict. The key is
/// `(score desc, path asc, line asc)`, the same chain `rank::sort` uses, which makes it a
/// total order over distinct matches: a truncation can therefore never be resolved by the
/// order the walk's threads happened to arrive in.
struct Candidate {
    score: i32,
    m: Match,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Lower score is worse, hence greater. Then larger path, then larger line.
        other
            .score
            .cmp(&self.score)
            .then_with(|| self.m.path.cmp(&other.m.path))
            .then_with(|| self.m.line.cmp(&other.m.line))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for Candidate {}
