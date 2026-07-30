use std::path::Path;

use super::file_metadata;
use super::retain::{BoundedRetain, FileOffer, MAX_RETAINED};

use crate::error::TilthError;
use crate::search::rank;
use crate::types::{FacetTotals, Match, SearchResult};
use grep_regex::RegexMatcher;
use grep_searcher::sinks::UTF8;
use grep_searcher::Searcher;

const MAX_MATCHES: usize = 10;
const FULL_MAX_MATCHES: usize = 100;
const MAX_SEARCH_FILE_SIZE: u64 = 500_000;

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
// arrival, while the *counts* stay exact. So the reported totals are still true totals, which
// is the half of this that a naive cap gets wrong.
//
// That bound, its heap, its inverted `Ord` and its exact counters all live in `search::retain`
// now, shared with the symbol path (#62). This file carried a second copy for a while, with its
// own `Candidate` whose key was three levels where `retain`'s is five — harmless on this input,
// where every match has `is_definition: false` and a unique `(path, line)`, but an invariant
// stated in two places and enforced in one. It has already been the source of one bug on the
// other copy (the inverted `Ord` shipped backwards, keeping the worst ties), and a fix to either
// had to be applied to both. `retain::Candidate` is now the only implementation; the three
// properties that make a bound safe here are argued there, at the type that enforces them.
//
// What this file still owns is the *facet* mapping, and it is exact rather than approximate:
// every content match has `is_definition: false`, so `facets::primary_package` finds no primary
// definition, `is_same_package` short-circuits to false, and every non-test match lands in
// `usages_cross`. Two of the five buckets are reachable, and `retain`'s tallies count both.

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

    let sink = BoundedRetain::new(MAX_RETAINED);

    let walker = super::walker(scope, glob)?;

    walker.run(|| {
        let matcher = &matcher;
        let sink = &sink;
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

            // Matches go straight into the sink as they are found, `OFFER_CHUNK` at a time,
            // rather than accumulating this file's whole `Vec<Match>` first. That per-file
            // term was #59: it is multiplied by the walk's thread count and is independent
            // of `MAX_RETAINED`, so it reached 422 MB on a dense fixture while retention was
            // doing exactly what it promised. Where the chunk boundaries fall cannot change
            // which matches survive — see `FileOffer`.
            let mut offer = FileOffer::new(sink);
            let mut searcher = Searcher::new();

            let _ = searcher.search_slice(
                matcher,
                &bytes,
                UTF8(|line_num, line| {
                    offer.push(
                        Match {
                            path: path.to_path_buf(),
                            line: line_num as u32,
                            text: crate::types::match_text(line),
                            is_definition: false,
                            exact: false,
                            file_lines,
                            mtime,
                            def_range: None,
                            def_name: None,
                            def_weight: 0,
                            impl_target: None,
                        },
                        &mut scorer,
                    );
                    Ok(true)
                }),
            );
            offer.finish(&mut scorer);

            ignore::WalkState::Continue
        })
    });

    // True totals, read once after `walker.run` has joined every thread. The sink is the only
    // thing that sees every match, so it is where they are kept; they only ever *report* —
    // nothing above reads them to decide whether to keep walking, which is the distinction
    // that made the old match-count cutoffs non-deterministic.
    let (mut all_matches, tallies) = sink.finish();
    let total = tallies.total();
    let tests = tallies.tests;
    // The invariant this file used to argue in prose, now asserted where it is relied on: a
    // content match is never a definition, so only the test/usage split is reachable and the
    // facet mapping below is exact rather than approximate.
    debug_assert_eq!(
        (tallies.definitions, tallies.implementations),
        (0, 0),
        "content search produced a definition match; the facet mapping below assumes it cannot"
    );

    // `into_matches` returns the retained set in no particular order. That costs nothing:
    // `rank::sort`'s key is a total order over these matches, so the page is a function of the
    // tree rather than of the order the walk's threads arrived in.
    rank::sort(&mut all_matches, pattern, scope, context);

    // Per-facet totals. Content search used to return `FacetTotals::default()`, i.e. all
    // zeros, which made `count_label` print a bare `10` and suppressed every hidden-count
    // tail — a query with 34290 matches rendered exactly like one with 10.
    //
    // These come from the sink's exact tallies rather than from the retained set, and they are
    // exact rather than approximate for the reason asserted above: only two of the five
    // buckets are reachable, and the sink counts both.
    let facet_totals = FacetTotals {
        definitions: 0,
        implementations: 0,
        tests,
        usages_local: 0,
        usages_cross: tallies.usages,
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
