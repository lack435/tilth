use std::path::Path;
use std::sync::Mutex;

use super::file_metadata;

use crate::error::TilthError;
use crate::search::rank;
use crate::types::{Match, SearchResult};
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
// Neither row is the worst case, and the worst case is NOT bounded. 34290 matches over
// 176k files is 0.2 matches per file; nothing here caps the *total* retained. The size
// gate below is per file, and `is_minified_by_content` only rejects files with under two
// newlines in their first 2 KB, so a 499 KB file of short lines is searched in full and
// can contribute ~150k matches by itself. Measured on a deliberately dense 400-file,
// 49 MB fixture (a match on every line): 0.31s / 33 MB bounded, 36s / 457 MB complete.
// A `kind: "content"` query for something like `return` on a large tree is that shape,
// and it is an invited query for a tool that offers itself as a grep replacement.
// Bounding retained matches without reintroducing the nondeterminism is tracked
// separately — it needs a per-file or ranked-streaming cap, not a shared counter.

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

    let matches: Mutex<Vec<Match>> = Mutex::new(Vec::new());

    let walker = super::walker(scope, glob)?;

    walker.run(|| {
        let matcher = &matcher;
        let matches = &matches;

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
                let mut all = matches
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                // One lock, one contiguous block per file — see the determinism note at the
                // top of this file. Extending per match would break tie-ordering.
                all.extend(file_matches);
            }

            ignore::WalkState::Continue
        })
    });

    let mut all_matches = matches
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let total = all_matches.len();

    rank::sort(&mut all_matches, pattern, scope, context);

    // Per-facet totals on the *pre-cap* set. Content search used to return
    // `FacetTotals::default()`, i.e. all zeros, which made `count_label` print a bare
    // `10` and suppressed every hidden-count tail — so a query with 34290 matches
    // rendered exactly like one with 10. The counts are now true totals.
    let facet_totals = super::facets::facet_totals(&all_matches, scope);

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
