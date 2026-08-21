use std::path::Path;

use super::file_metadata;
use super::retain::{BoundedRetain, FileOffer, MAX_RETAINED};

use crate::error::TilthError;
use crate::search::rank;
use crate::types::{CaseMode, FacetTotals, Match, SearchResult};
use grep_regex::RegexMatcherBuilder;
use grep_searcher::sinks::UTF8;
use grep_searcher::Searcher;

const MAX_MATCHES: usize = 10;
const FULL_MAX_MATCHES: usize = 100;
/// Content search runs no parser (ripgrep internals), so this is *not* the AST parse gate
/// (`lang::parse_budget::MAX_PARSE_FILE_SIZE`) and carries none of its budget coupling — it only
/// bounds bytes scanned per file. Kept at 1 MB to match the parse gate so both surfaces reach the
/// same large source files; retention past that is bounded separately by `MAX_RETAINED`.
const MAX_SEARCH_FILE_SIZE: u64 = 1_000_000;

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
///
/// `case` selects the matcher's case behaviour. It is applied identically to the literal and regex
/// paths: `Smart` inspects the (escaped or raw) pattern's literal characters, so `[A-Z]` in a regex
/// correctly forces sensitivity and an inline `(?i)` flag still overrides the matcher default for
/// the span it scopes.
pub fn search(
    pattern: &str,
    scope: &Path,
    is_regex: bool,
    context: Option<&Path>,
    glob: Option<&str>,
    full: bool,
    case: CaseMode,
) -> Result<SearchResult, TilthError> {
    let max_matches = if full { FULL_MAX_MATCHES } else { MAX_MATCHES };
    // Literal queries are regex-escaped so metacharacters match verbatim; regex queries pass
    // through untouched. Case is then a matcher-builder setting rather than anything baked into
    // the pattern, so the two paths share one build site.
    let escaped;
    let effective_pattern = if is_regex {
        pattern
    } else {
        escaped = regex_syntax::escape(pattern);
        escaped.as_str()
    };
    let mut builder = RegexMatcherBuilder::new();
    match case {
        CaseMode::Sensitive => {}
        CaseMode::Insensitive => {
            builder.case_insensitive(true);
        }
        // Only one arm ever sets a flag, so `case_smart` and `case_insensitive` are never both
        // enabled here. (grep-regex does accept both at once, letting `case_insensitive` win — we
        // simply never rely on that precedence.)
        CaseMode::Smart => {
            builder.case_smart(true);
        }
    }
    let matcher = builder
        .build(effective_pattern)
        .map_err(|e| TilthError::InvalidQuery {
            query: pattern.to_string(),
            reason: e.to_string(),
        })?;

    let sink = BoundedRetain::new(MAX_RETAINED);

    let walker = super::walker(scope, glob)?;

    super::run_walk(walker, || {
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

            // Catch unmarked minified bundles in the 100KB–1MB range.
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

#[cfg(test)]
mod tests {
    use super::search;
    use crate::types::CaseMode;

    /// One file, three lines, deliberately varied in case so a case decision is
    /// observable in the match count.
    fn fixture() -> tempfile::TempDir {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("notes.txt"),
            "ALERT THRESHOLD marker\n\
             alert threshold here\n\
             Mixed Alert text\n",
        )
        .unwrap();
        tmp
    }

    fn count(pattern: &str, is_regex: bool, case: CaseMode) -> usize {
        let tmp = fixture();
        search(pattern, tmp.path(), is_regex, None, None, false, case)
            .unwrap()
            .total_found
    }

    /// Sensitive: an all-lowercase literal matches only the lowercase line.
    /// This is the pre-2026 behaviour — the class of failure reported in #138.
    #[test]
    fn sensitive_matches_exact_case_only() {
        assert_eq!(count("alert threshold", false, CaseMode::Sensitive), 1);
    }

    /// Smart on an all-lowercase query behaves case-insensitively: it also
    /// matches the uppercase line. Mutating the `Smart` arm back to sensitive
    /// (or dropping the `case` plumbing) drops this to 1.
    #[test]
    fn smart_all_lowercase_is_case_insensitive() {
        assert_eq!(count("alert threshold", false, CaseMode::Smart), 2);
    }

    /// Smart on a query that carries an uppercase letter stays case-sensitive.
    /// `"ALERT threshold"` matches neither `"ALERT THRESHOLD"` (differs on
    /// THRESHOLD) nor `"alert threshold"` (differs on ALERT), so a correct
    /// smart-case gives zero. A smart arm wrongly wired to `case_insensitive`
    /// would return 2.
    #[test]
    fn smart_with_uppercase_stays_sensitive() {
        assert_eq!(count("ALERT threshold", false, CaseMode::Smart), 0);
    }

    /// Insensitive forces case-folding even when the query carries uppercase —
    /// the behaviour that distinguishes it from `Smart` on the same input
    /// (`smart_with_uppercase_stays_sensitive` gets 0 for this very query).
    #[test]
    fn insensitive_ignores_case_even_with_uppercase_query() {
        assert_eq!(count("ALERT threshold", false, CaseMode::Insensitive), 2);
    }

    /// An inline `(?i)` flag is honoured on the regex path regardless of the
    /// requested `CaseMode` — the flag scopes case-insensitivity in the pattern
    /// itself. This is issue #138's requested feature (3). Passing `Sensitive`
    /// proves the flag, not the mode, is what folds case here.
    #[test]
    fn regex_inline_case_flag_is_honoured() {
        assert_eq!(count("(?i)alert threshold", true, CaseMode::Sensitive), 2);
    }

    /// The sharp interaction: `(?i)` followed by an *uppercase* literal under
    /// `Smart`. `case_smart` sees the uppercase `A` and would set the matcher
    /// default to sensitive, but the inline flag must still win and fold case.
    /// All three fixture lines contain "alert"/"ALERT"/"Alert", so a correct
    /// result is 3; if smart-case sensitivity wrongly overrode the flag, only
    /// the `"Mixed Alert text"` line would match and this would be 1.
    #[test]
    fn regex_inline_flag_overrides_smart_case_on_uppercase() {
        assert_eq!(count("(?i)Alert", true, CaseMode::Smart), 3);
    }

    /// The same `(?i)` text on the *literal* path is escaped, so it matches the
    /// literal characters `(?i)…` — which no line contains. Guards against the
    /// literal path accidentally interpreting regex metacharacters.
    #[test]
    fn literal_path_escapes_inline_flag() {
        assert_eq!(count("(?i)alert threshold", false, CaseMode::Sensitive), 0);
    }
}
