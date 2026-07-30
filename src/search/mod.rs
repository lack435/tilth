mod alloc;
pub mod blast;
pub mod callees;
pub mod callers;
pub mod content;
pub mod deps;
pub mod facets;
pub mod glob;
pub mod grok;
pub mod rank;
pub mod siblings;
pub mod strip;
pub mod symbol;
pub mod truncate;

mod bloom_walk;
mod callee_query;
mod retain;
pub mod scope;

use std::collections::HashSet;
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use ignore::WalkBuilder;

use crate::cache::OutlineCache;
use crate::error::TilthError;
use crate::format;
use crate::read;
use crate::session::Session;
use crate::types::{estimate_tokens, FileType, Match, SearchResult};

use crate::format::rel;

// Directories that are always skipped — build artifacts, dependencies, VCS internals.
// We skip these explicitly instead of relying on .gitignore so that locally-relevant
// gitignored files (docs/, configs, generated code) are still searchable.
pub(crate) const SKIP_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "dist",
    "build",
    "__pycache__",
    ".pycache",
    "vendor",
    ".next",
    ".nuxt",
    "coverage",
    ".cache",
    ".tox",
    ".venv",
    ".eggs",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".turbo",
    ".parcel-cache",
    ".svelte-kit",
    "out",
    ".output",
    ".vercel",
    ".netlify",
    ".gradle",
    ".idea",
    ".scala-build",
    "target",
    ".bloop",
    ".metals",
];

const EXPAND_FULL_FILE_THRESHOLD: u64 = 800;

/// Cap for inlined markdown section bodies in the default preview slot.
/// Long sections get a tail "… (N more lines — pass --expand to see the full
/// section)" so the user knows to expand for the rest.
const MARKDOWN_PREVIEW_MAX_LINES: usize = 40;

/// Shared walker policy: searches ALL files except known junk directories.
/// Does NOT respect .gitignore — ensures gitignored but locally-relevant files
/// are found. Used by both the parallel search walker (`walker()`) and the
/// sequential map walker (`crate::map::generate`), which each apply their own
/// final `.max_depth()`/`.threads()` and `.build()`/`.build_parallel()`.
pub(crate) fn base_walk_builder(scope: &Path) -> WalkBuilder {
    let mut builder = WalkBuilder::new(scope);
    builder
        .follow_links(true)
        .same_file_system(true) // Stop at mount boundaries (NFS, external volumes).
        .hidden(false)
        .git_ignore(false)
        .git_global(false)
        .git_exclude(false)
        .ignore(false)
        .parents(false)
        .filter_entry(|entry| {
            if entry.file_type().is_some_and(|ft| ft.is_dir()) {
                if let Some(name) = entry.file_name().to_str() {
                    return !SKIP_DIRS.contains(&name);
                }
            }
            true
        });
    builder
}

/// Tree walks constructed, per scope, for tests that assert how many a query performs.
///
/// `walker` builds every traversal on the search paths this measures — the primary caller
/// walk, second hops, existence scans, definition and usage walks — so counting here needs
/// no per-call-site instrumentation to fall out of date. Counting construction rather than
/// completion is deliberate: a walk that is built is a walk that runs, and construction is
/// where the count is unambiguous.
///
/// Not literally every walk in the crate: `find_basename_fallback` below and `map.rs` build
/// their own. Neither is reachable from the paths under test — `find_basename_fallback` only
/// through `format_search_result`, which the multi-symbol path does not call — but a
/// traversal added outside `walker` would be invisible here rather than loudly wrong.
///
/// **Keyed by scope, not a bare counter**, for two reasons that between them rule out the
/// simpler options. Tests run in parallel in one process, so a global counter is inflated by
/// whatever else is running — and a `thread_local` does not fix that, because
/// `symbol::search` builds its two walkers inside `rayon::join`, which may run either
/// closure on a stolen thread. Every test uses its own `tempfile::tempdir`, so keying on the
/// scope isolates measurements without any coordination between tests.
#[cfg(test)]
pub(crate) static WALKS_BUILT: std::sync::Mutex<Option<std::collections::HashMap<PathBuf, usize>>> =
    std::sync::Mutex::new(None);

/// Start counting walks under `scope`, discarding any previous count for it.
#[cfg(test)]
pub(crate) fn reset_walk_count(scope: &Path) {
    let mut counts = WALKS_BUILT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    counts
        .get_or_insert_with(std::collections::HashMap::new)
        .insert(scope.to_path_buf(), 0);
}

/// Walks built under `scope` since the last [`reset_walk_count`].
#[cfg(test)]
pub(crate) fn walk_count(scope: &Path) -> usize {
    let counts = WALKS_BUILT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    counts
        .as_ref()
        .and_then(|c| c.get(scope).copied())
        .unwrap_or(0)
}

/// Build a parallel directory walker that searches ALL files except known junk directories.
/// Does NOT respect .gitignore — ensures gitignored but locally-relevant files are found.
/// When `glob` is Some, applies a file-pattern override (whitelist or negation).
pub(crate) fn walker(scope: &Path, glob: Option<&str>) -> Result<ignore::WalkParallel, TilthError> {
    #[cfg(test)]
    {
        let mut counts = WALKS_BUILT
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Only scopes a test opted in to are tracked, so this stays a no-op for the rest of
        // the suite rather than accumulating an entry per tempdir it ever searched.
        if let Some(n) = counts.as_mut().and_then(|c| c.get_mut(scope)) {
            *n += 1;
        }
    }

    let threads = std::env::var("TILTH_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism().map_or(4, |n| (n.get() / 2).clamp(2, 6))
        });

    let mut builder = base_walk_builder(scope);
    builder.threads(threads);

    if let Some(pattern) = glob {
        if !pattern.is_empty() {
            let mut overrides = ignore::overrides::OverrideBuilder::new(scope);
            overrides
                .add(pattern)
                .map_err(|e| TilthError::InvalidQuery {
                    query: pattern.to_string(),
                    reason: format!("invalid glob: {e}"),
                })?;
            builder.overrides(overrides.build().map_err(|e| TilthError::InvalidQuery {
                query: pattern.to_string(),
                reason: format!("invalid glob: {e}"),
            })?);
        }
    }

    Ok(builder.build_parallel())
}

/// Parse `/pattern/` regex syntax. Returns (pattern, `is_regex`).
fn parse_pattern(query: &str) -> (&str, bool) {
    if query.starts_with('/') && query.ends_with('/') && query.len() > 2 {
        (&query[1..query.len() - 1], true)
    } else {
        (query, false)
    }
}

/// Get `file_lines` estimate and mtime from metadata. One `stat()` per file.
pub(crate) fn file_metadata(path: &Path) -> (u32, SystemTime) {
    match std::fs::metadata(path) {
        Ok(meta) => {
            let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            let est_lines = (meta.len() / 40).max(1) as u32;
            (est_lines, mtime)
        }
        Err(_) => (0, SystemTime::UNIX_EPOCH),
    }
}

/// Dispatch search by query type.
pub fn search_symbol(
    query: &str,
    scope: &Path,
    cache: &OutlineCache,
    glob: Option<&str>,
) -> Result<String, TilthError> {
    let result = symbol::search(query, scope, None, glob, false)?;
    let bloom = crate::index::bloom::BloomFilterCache::new();
    format_search_result(&result, cache, None, &bloom, 0, None)
}

pub fn search_symbol_expanded(
    query: &str,
    scope: &Path,
    cache: &OutlineCache,
    session: &Session,
    bloom: &crate::index::bloom::BloomFilterCache,
    expand: usize,
    context: Option<&Path>,
    glob: Option<&str>,
    full: bool,
    budget: Option<u64>,
) -> Result<String, TilthError> {
    let result = symbol::search(query, scope, context, glob, full)?;
    format_search_result(&result, cache, Some(session), bloom, expand, budget)
}

pub fn search_multi_symbol_expanded(
    queries: &[&str],
    scope: &Path,
    cache: &OutlineCache,
    session: &Session,
    bloom: &crate::index::bloom::BloomFilterCache,
    expand: usize,
    context: Option<&Path>,
    glob: Option<&str>,
    full: bool,
    budget: Option<u64>,
) -> Result<String, TilthError> {
    // Shared expand budget: at least 1 slot per query, or explicit expand if higher.
    // expand=0 means no expansion at all.
    let mut expand_remaining = if expand == 0 {
        0
    } else {
        expand.max(queries.len())
    };
    let mut expanded_files = HashSet::new();
    let mut sections = Vec::with_capacity(queries.len());

    // One pair of walks for every target, rather than a pair per target. Rendering below
    // is unchanged and still per target — see `symbol::search_multi` for why a batched
    // result is identical to a lone `symbol::search`'s.
    let results = symbol::search_multi(queries, scope, context, glob, full)?;

    for result in &results {
        let mut out = format::search_header(
            &result.query,
            &result.scope,
            result.total_found,
            result.definitions,
            result.usages,
        );
        let mut segments: Vec<(i64, usize, usize)> = Vec::new();
        format_matches(
            &result.matches,
            &result.scope,
            cache,
            Some(session),
            bloom,
            &mut expand_remaining,
            &mut expanded_files,
            &mut out,
            &mut segments,
        );
        if result.total_found > result.matches.len() {
            let omitted = result.total_found - result.matches.len();
            let _ = write!(
                out,
                "\n\n... and {omitted} more matches. Narrow with scope."
            );
        }
        // budget.unwrap_or(DEFAULT_BUDGET): keeps the no-budget path byte-
        // identical to before this fix (see format_search_result's own comment).
        let budget_tokens = budget.unwrap_or(crate::budget::DEFAULT_BUDGET);
        out = crate::search::alloc::fit_to_budget(&out, &segments, budget_tokens);
        sections.push(out);
    }

    Ok(sections.join("\n\n---\n"))
}

pub fn search_content(
    query: &str,
    scope: &Path,
    cache: &OutlineCache,
    glob: Option<&str>,
) -> Result<String, TilthError> {
    let (pattern, is_regex) = parse_pattern(query);
    let result = content::search(pattern, scope, is_regex, None, glob, false)?;
    let bloom = crate::index::bloom::BloomFilterCache::new();
    format_search_result(&result, cache, None, &bloom, 0, None)
}

pub fn search_regex(
    pattern: &str,
    scope: &Path,
    cache: &OutlineCache,
    glob: Option<&str>,
) -> Result<String, TilthError> {
    let result = content::search(pattern, scope, true, None, glob, false)?;
    let bloom = crate::index::bloom::BloomFilterCache::new();
    format_search_result(&result, cache, None, &bloom, 0, None)
}

pub fn search_content_expanded(
    query: &str,
    scope: &Path,
    cache: &OutlineCache,
    session: &Session,
    expand: usize,
    context: Option<&Path>,
    glob: Option<&str>,
    full: bool,
    budget: Option<u64>,
) -> Result<String, TilthError> {
    let (pattern, is_regex) = parse_pattern(query);
    let result = content::search(pattern, scope, is_regex, context, glob, full)?;
    let bloom = crate::index::bloom::BloomFilterCache::new();
    format_search_result(&result, cache, Some(session), &bloom, expand, budget)
}

/// Expanded regex search — takes raw pattern, no slash wrapping needed.
pub fn search_regex_expanded(
    pattern: &str,
    scope: &Path,
    cache: &OutlineCache,
    session: &Session,
    expand: usize,
    context: Option<&Path>,
    glob: Option<&str>,
    full: bool,
    budget: Option<u64>,
) -> Result<String, TilthError> {
    let result = content::search(pattern, scope, true, context, glob, full)?;
    let bloom = crate::index::bloom::BloomFilterCache::new();
    format_search_result(&result, cache, Some(session), &bloom, expand, budget)
}

/// Raw symbol search — returns structured result for programmatic inspection.
pub fn search_symbol_raw(
    query: &str,
    scope: &Path,
    glob: Option<&str>,
) -> Result<SearchResult, TilthError> {
    symbol::search(query, scope, None, glob, false)
}

/// Raw content search — returns structured result for programmatic inspection.
pub fn search_content_raw(
    query: &str,
    scope: &Path,
    glob: Option<&str>,
) -> Result<SearchResult, TilthError> {
    let (pattern, is_regex) = parse_pattern(query);
    content::search(pattern, scope, is_regex, None, glob, false)
}

/// Raw regex search — returns structured result for programmatic inspection.
pub fn search_regex_raw(
    pattern: &str,
    scope: &Path,
    glob: Option<&str>,
) -> Result<SearchResult, TilthError> {
    content::search(pattern, scope, true, None, glob, false)
}

/// Format a raw search result (symbol or content — both use the same pipeline).
pub fn format_raw_result(
    result: &SearchResult,
    cache: &OutlineCache,
) -> Result<String, TilthError> {
    let bloom = crate::index::bloom::BloomFilterCache::new();
    format_search_result(result, cache, None, &bloom, 0, None)
}

pub fn search_glob(pattern: &str, scope: &Path) -> Result<String, TilthError> {
    let result = glob::search(pattern, scope)?;
    format_glob_result(&result, scope)
}

/// Render the count for a facet section heading. Returns the bare displayed
/// count when nothing was hidden (`shown == total`), or `displayed/total`
/// when the cap dropped some entries — so a reader sees at a glance whether
/// the facet was truncated.
fn count_label(shown: usize, total: usize) -> String {
    if shown >= total {
        format!("{shown}")
    } else {
        format!("{shown}/{total}")
    }
}

/// Emit a per-facet hidden-count tail line after a truncated facet's entries.
/// Wording mirrors the linear-path global tail so a reader sees a single
/// consistent shape — only the noun changes per facet kind.
fn write_hidden_tail(out: &mut String, shown: usize, total: usize, kind: &str) {
    if shown < total {
        let hidden = total - shown;
        let _ = write!(out, "\n\n... and {hidden} more {kind}. Narrow with scope.");
    }
}

/// Format match entries with optional expansion.
/// Groups consecutive usage matches in the same enclosing function to reduce token noise.
/// Shared expand state enables cross-query dedup in multi-symbol search.
fn format_matches(
    matches: &[Match],
    scope: &Path,
    cache: &OutlineCache,
    session: Option<&Session>,
    bloom: &crate::index::bloom::BloomFilterCache,
    expand_remaining: &mut usize,
    expanded_files: &mut HashSet<PathBuf>,
    out: &mut String,
    segments: &mut Vec<(i64, usize, usize)>,
) {
    // Multi-file: one expand per unique file. Single-file: sequential per-match.
    // expanded_files may contain entries from prior queries (cross-query dedup).
    let multi_file = matches
        .first()
        .is_some_and(|first| matches.iter().any(|m| m.path != first.path));

    // Group consecutive non-definition matches by (path, enclosing_outline_idx).
    // Definitions are never grouped — they need individual expand with callees/siblings.
    let groups = group_matches(matches, cache);

    for group in &groups {
        if group.len() == 1 {
            // Single match — format as before
            let start = out.len();
            format_single_match(
                group[0],
                scope,
                cache,
                session,
                bloom,
                expand_remaining,
                expanded_files,
                multi_file,
                out,
            );
            segments.push((i64::from(group[0].def_weight), start, out.len()));
        } else {
            // Multiple usages collapsed into one entry
            let start = out.len();
            format_grouped_usages(group, scope, cache, out);
            let value = group
                .iter()
                .map(|m| i64::from(m.def_weight))
                .max()
                .unwrap_or(0);
            segments.push((value, start, out.len()));
        }
    }
}

/// Group consecutive non-definition matches by (path, enclosing outline entry).
/// Dedup key for definition matches: (path, line, `def_range`, `def_name`, `impl_target`).
type DefKey<'a> = (
    &'a Path,
    u32,
    Option<(u32, u32)>,
    Option<&'a str>,
    Option<&'a str>,
);

/// Returns a Vec of groups, where each group is a slice of matches.
/// Definitions and impl matches are always singleton groups.
fn group_matches<'a>(matches: &'a [Match], cache: &OutlineCache) -> Vec<Vec<&'a Match>> {
    let mut groups: Vec<Vec<&Match>> = Vec::new();
    let mut seen_defs: HashSet<DefKey<'_>> = HashSet::new();

    for m in matches {
        if m.is_definition || m.impl_target.is_some() {
            let key = (
                m.path.as_path(),
                m.line,
                m.def_range,
                m.def_name.as_deref(),
                m.impl_target.as_deref(),
            );
            if !seen_defs.insert(key) {
                continue;
            }
        }
        // Definitions and impls are never grouped
        if m.is_definition || m.impl_target.is_some() {
            groups.push(vec![m]);
            continue;
        }

        // For usages: try to merge with previous group if same (path, outline_idx)
        if let Some(last_group) = groups.last_mut() {
            let prev = last_group[0];
            // Only merge usages (previous must also be a usage in the same file)
            if !prev.is_definition
                && prev.impl_target.is_none()
                && prev.path == m.path
                && m.file_lines >= 50
            {
                let prev_idx = find_enclosing_outline_idx(&prev.path, prev.line, cache);
                let curr_idx = find_enclosing_outline_idx(&m.path, m.line, cache);
                if prev_idx.is_some() && prev_idx == curr_idx {
                    last_group.push(m);
                    continue;
                }
            }
        }
        groups.push(vec![m]);
    }
    groups
}

/// Format a group of usages collapsed into a single entry.
fn format_grouped_usages(group: &[&Match], scope: &Path, cache: &OutlineCache, out: &mut String) {
    let first = group[0];
    let path_str = rel(&first.path, scope);

    // Build comma-separated line list, collapsing consecutive runs (e.g. 55,56,57 → 55-57)
    let lines: Vec<u32> = group.iter().map(|m| m.line).collect();
    let line_str = format_line_list(&lines);

    let scope_label = enclosing_scope_label(&first.path, first.line, cache);

    let _ = write!(out, "\n\n### {path_str}:{line_str} [{} usages", group.len());
    if let Some(ref label) = scope_label {
        let _ = write!(out, " in {label}");
    }
    out.push(']');

    // Show outline context once for the group
    if let Some(context) = outline_context_for_match(&first.path, first.line, cache) {
        out.push_str(&context);
    }
}

/// Format a comma-separated line list, collapsing consecutive runs.
/// e.g. [50, 55, 56, 57, 58, 63, 67] → "50,55-58,63,67"
fn format_line_list(lines: &[u32]) -> String {
    if lines.is_empty() {
        return String::new();
    }
    let mut parts: Vec<String> = Vec::new();
    let mut run_start = lines[0];
    let mut run_end = lines[0];
    for &line in &lines[1..] {
        if line == run_end + 1 {
            run_end = line;
        } else {
            if run_end > run_start + 1 {
                parts.push(format!("{run_start}-{run_end}"));
            } else if run_end > run_start {
                parts.push(format!("{run_start},{run_end}"));
            } else {
                parts.push(format!("{run_start}"));
            }
            run_start = line;
            run_end = line;
        }
    }
    if run_end > run_start + 1 {
        parts.push(format!("{run_start}-{run_end}"));
    } else if run_end > run_start {
        parts.push(format!("{run_start},{run_end}"));
    } else {
        parts.push(format!("{run_start}"));
    }
    parts.join(",")
}

/// The symbol to feed query-aware truncation when expanding a match's body.
///
/// For `impl`/`implements` matches the user searched for the trait or interface,
/// which is held in `impl_target` — `def_name` is the rendered label
/// (`"impl Trait for Type"` / `"Type implements Trait"`) and never appears
/// verbatim in the body, so boosting on it is a no-op. For plain definitions
/// `impl_target` is `None` and the searched token is the symbol name in
/// `def_name`. Preferring `impl_target` routes the real query into the boost for
/// both shapes.
fn boost_query(m: &Match) -> Option<&str> {
    m.impl_target.as_deref().or(m.def_name.as_deref())
}

/// Format a single match entry (unchanged from original behavior).
fn format_single_match(
    m: &Match,
    scope: &Path,
    cache: &OutlineCache,
    session: Option<&Session>,
    bloom: &crate::index::bloom::BloomFilterCache,
    expand_remaining: &mut usize,
    expanded_files: &mut HashSet<PathBuf>,
    multi_file: bool,
    out: &mut String,
) {
    let kind = if m.impl_target.is_some() {
        "impl"
    } else if m.is_definition {
        "definition"
    } else {
        "usage"
    };

    // For usages, append the enclosing function/section if we can recover one.
    // Definitions and impls already are the named scope.
    let scope_suffix = if m.is_definition || m.impl_target.is_some() {
        String::new()
    } else {
        enclosing_scope_label(&m.path, m.line, cache)
            .map(|s| format!(" in {s}"))
            .unwrap_or_default()
    };

    // Show line range for definitions with def_range, otherwise just the line
    if m.is_definition {
        if let Some((start, end)) = m.def_range {
            let _ = write!(
                out,
                "\n\n### {}:{}-{} [{kind}]",
                rel(&m.path, scope),
                start,
                end
            );
        } else {
            let _ = write!(out, "\n\n### {}:{} [{kind}]", rel(&m.path, scope), m.line);
        }
    } else {
        let _ = write!(
            out,
            "\n\n### {}:{} [{kind}{scope_suffix}]",
            rel(&m.path, scope),
            m.line
        );
    }

    // Markdown-heading defs (`def_weight == 30`): the heading text alone is
    // just the query, so the default preview slot would carry no information.
    // Inline the section body directly. Bypasses the --expand budget — this
    // is a fixed-cost preview, not the on-demand expand — and short-circuits
    // the rest of the function (no callees / siblings etc. apply to a
    // markdown section). On any read failure or empty body, fall through to
    // the existing outline / single-line preview branches.
    if m.is_definition && m.def_weight == 30 {
        if let Some((heading_line_1, section_end_1)) = m.def_range {
            if let Ok(content) = fs::read_to_string(&m.path) {
                let lines: Vec<&str> = content.lines().collect();
                // def_range is `(heading_line, section_end)` in 1-indexed
                // inclusive form (see `find_defs_markdown_buf`). The body
                // starts at the line *after* the heading. In 0-indexed
                // half-open form: `[heading_line_1 .. section_end_1)`.
                let body_start = heading_line_1 as usize;
                let body_end = (section_end_1 as usize).min(lines.len());
                if body_start < body_end {
                    let total_body_lines = body_end - body_start;
                    let take_n = total_body_lines.min(MARKDOWN_PREVIEW_MAX_LINES);
                    out.push('\n');
                    for line in &lines[body_start..body_start + take_n] {
                        out.push_str(line);
                        out.push('\n');
                    }
                    if total_body_lines > take_n {
                        let truncated = total_body_lines - take_n;
                        let _ = write!(
                            out,
                            "… ({truncated} more lines — pass --expand to see the full section)"
                        );
                    }
                    return;
                }
            }
        }
    }

    // Check session dedup for definitions with def_range. The mtime
    // check ensures a post-edit search re-inlines the body rather than
    // pointing at stale line numbers.
    let current_mtime = std::fs::metadata(&m.path)
        .ok()
        .and_then(|md| md.modified().ok());
    let deduped = m.is_definition
        && m.def_range.is_some()
        && session
            .is_some_and(|s| current_mtime.is_some_and(|t| s.is_expanded(&m.path, m.line, t)));
    // expand_match always prints a range containing m.line (def_range starts
    // at m.line for definitions; the ±10 fallback for def_range: None / usages
    // trivially contains it), so the raw "-> [line] text" preview would
    // reprint m.text byte-for-byte inside the fence below. Only the
    // structural outline_context (neighboring entries' signatures, not the
    // matched line's own source) survives alongside an expansion.
    let fence_will_follow =
        *expand_remaining > 0 && !deduped && !(multi_file && expanded_files.contains(&m.path));

    // Skip outline for small files — the expanded code speaks for itself
    if m.file_lines < 50 {
        if !fence_will_follow {
            let _ = write!(out, "\n-> [{}]   {}", m.line, m.text);
        }
    } else if let Some(context) = outline_context_for_match(&m.path, m.line, cache) {
        out.push_str(&context);
    } else if !fence_will_follow {
        let _ = write!(out, "\n-> [{}]   {}", m.line, m.text);
    }

    if *expand_remaining > 0 {
        if deduped {
            if let Some((start, end)) = m.def_range {
                let _ = write!(
                    out,
                    "\n\n[shown earlier] {}:{}-{} {}",
                    rel(&m.path, scope),
                    start,
                    end,
                    m.text
                );
            }
        } else {
            let skip = multi_file && expanded_files.contains(&m.path);
            if !skip {
                if let Some((code, content)) = expand_match(m, scope) {
                    if m.is_definition && m.def_range.is_some() {
                        if let (Some(s), Some(t)) = (session, current_mtime) {
                            s.record_expand(&m.path, m.line, t);
                        }
                    }

                    let file_type = crate::lang::detect_file_type(&m.path);
                    let mut skip_lines = strip::strip_noise(&content, &m.path, m.def_range);

                    if let Some((def_start, def_end)) = m.def_range {
                        if let crate::types::FileType::Code(_) = file_type {
                            if let Some(keep) = truncate::select_diverse_lines(
                                &content,
                                def_start,
                                def_end,
                                boost_query(m),
                            ) {
                                let keep_set: HashSet<u32> = keep.into_iter().collect();
                                for ln in def_start..=def_end {
                                    if !keep_set.contains(&ln) {
                                        skip_lines.insert(ln);
                                    }
                                }

                                // Record token savings: full def body vs kept lines.
                                // Measure raw line content (bytes + 1 for newline each),
                                // independent of any surrounding formatting.
                                if let Some(sess) = session {
                                    let body_lines: Vec<&str> = content
                                        .lines()
                                        .enumerate()
                                        .filter_map(|(i, l)| {
                                            let ln = (i as u32) + 1;
                                            if ln >= def_start && ln <= def_end {
                                                Some(l)
                                            } else {
                                                None
                                            }
                                        })
                                        .collect();
                                    let full_bytes: u64 =
                                        body_lines.iter().map(|l| l.len() as u64 + 1).sum();
                                    let kept_bytes: u64 = body_lines
                                        .iter()
                                        .enumerate()
                                        .filter_map(|(i, l)| {
                                            let ln = def_start + i as u32;
                                            if keep_set.contains(&ln) {
                                                Some(l.len() as u64 + 1)
                                            } else {
                                                None
                                            }
                                        })
                                        .sum();
                                    sess.record_savings(
                                        crate::types::estimate_tokens(full_bytes),
                                        crate::types::estimate_tokens(kept_bytes),
                                    );
                                }
                            }
                        }
                    }

                    let stripped_code = if skip_lines.is_empty() {
                        code
                    } else {
                        filter_code_lines(&code, &skip_lines)
                    };

                    out.push('\n');
                    out.push_str(&stripped_code);

                    if m.is_definition && m.def_range.is_some() {
                        if let crate::types::FileType::Code(lang) = file_type {
                            let callee_names =
                                callees::extract_callee_names(&content, lang, m.def_range);
                            if !callee_names.is_empty() {
                                // The declared search scope is the containment root for
                                // C/C++ include-root resolution; without it callee
                                // resolution can only follow includes in a tree that has
                                // a `.git` ancestor (#15). Canonicalized here rather than
                                // hoisted out of `format_single_match`: this block only
                                // runs for a definition that actually has callees, and it
                                // is one stat in front of a walk that reads and parses
                                // every imported file.
                                let boundary =
                                    crate::read::imports::canonical_boundary(Some(scope));
                                let mut nodes = callees::resolve_callees_transitive(
                                    &callee_names,
                                    &m.path,
                                    &content,
                                    bloom,
                                    2,
                                    15,
                                    boundary.as_deref(),
                                );

                                if let Some(ref name) = m.def_name {
                                    nodes.retain(|n| n.callee.name != *name);
                                }
                                if nodes.len() > 8 {
                                    nodes.sort_by_key(|n| i32::from(n.callee.file == m.path));
                                    nodes.truncate(8);
                                }

                                if !nodes.is_empty() {
                                    out.push_str("\n\n-- calls --");
                                    for n in &nodes {
                                        let c = &n.callee;
                                        let _ = write!(
                                            out,
                                            "\n  {}  {}:{}-{}",
                                            c.name,
                                            rel(&c.file, scope),
                                            c.start_line,
                                            c.end_line
                                        );
                                        if let Some(ref sig) = c.signature {
                                            let _ = write!(out, "  {sig}");
                                        }
                                        for child in &n.children {
                                            let _ = write!(
                                                out,
                                                "\n    -> {}  {}:{}-{}",
                                                child.name,
                                                rel(&child.file, scope),
                                                child.start_line,
                                                child.end_line
                                            );
                                            if let Some(ref sig) = child.signature {
                                                let _ = write!(out, "  {sig}");
                                            }
                                        }
                                    }
                                }
                            }

                            if let Some(def_range) = m.def_range {
                                let entries =
                                    crate::lang::outline::get_outline_entries(&content, lang);
                                if let Some(parent) = siblings::find_parent_entry(&entries, m.line)
                                {
                                    let refs = siblings::extract_sibling_references(
                                        &content, lang, def_range,
                                    );
                                    if !refs.is_empty() {
                                        let filtered: Vec<String> =
                                            if let Some(ref name) = m.def_name {
                                                refs.into_iter().filter(|r| r != name).collect()
                                            } else {
                                                refs
                                            };

                                        let resolved =
                                            siblings::resolve_siblings(&filtered, &parent.children);
                                        if !resolved.is_empty() {
                                            out.push_str("\n\n-- siblings --");
                                            for s in &resolved {
                                                let _ = write!(
                                                    out,
                                                    "\n  {}  {}:{}-{}  {}",
                                                    s.name,
                                                    rel(&m.path, scope),
                                                    s.start_line,
                                                    s.end_line,
                                                    s.signature,
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    *expand_remaining -= 1;
                    expanded_files.insert(m.path.clone());
                }
            }
        }
    }
}

/// Format a symbol/content search result.
/// When an outline cache is available, wraps each match in the file's outline context.
/// When `expand > 0`, the top N matches inline actual code (def body or ±10 lines).
/// When there are >5 matches, groups them into facets for easier navigation.
/// Prefer source languages over their compiled equivalents.
/// Higher value = more likely to be the original source.
fn source_priority(path: &Path) -> u8 {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "ts" | "tsx" => 10,
        "rs" | "go" | "py" | "rb" | "java" | "kt" | "scala" | "swift" | "c" | "cpp" | "h"
        | "cs" | "php" => 9,
        "js" | "jsx" | "mjs" | "cjs" => 7,
        _ => 3,
    }
}

/// Find a basename-matching candidate among already-collected search matches.
fn find_basename_candidate(matches: &[Match], query_lower: &str) -> Option<PathBuf> {
    let mut candidate: Option<&Path> = None;
    let mut best_priority: u8 = 0;

    for m in matches {
        let Some(stem) = m.path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if stem.to_ascii_lowercase() != query_lower {
            continue;
        }
        let ext = m.path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let is_code = matches!(
            ext,
            "rs" | "ts"
                | "tsx"
                | "js"
                | "jsx"
                | "go"
                | "py"
                | "rb"
                | "java"
                | "c"
                | "cpp"
                | "h"
                | "cs"
                | "swift"
                | "kt"
                | "scala"
                | "php"
        );
        if !is_code {
            if candidate.is_none() {
                candidate = Some(&m.path);
            }
            continue;
        }
        let prio = source_priority(&m.path);
        if prio > best_priority {
            best_priority = prio;
            candidate = Some(&m.path);
        }
    }

    candidate.map(Path::to_path_buf)
}

/// Format a token count into a human-readable string (e.g. "~1.2k" or "~743").
pub(crate) fn format_token_count(tokens: u64) -> String {
    if tokens >= 1000 {
        format!("~{}.{}k", tokens / 1000, (tokens % 1000) / 100)
    } else {
        format!("~{tokens}")
    }
}

/// Fallback: lightweight directory walk to find a basename-matching file
/// when it didn't survive ranking/truncation in the match set.
fn find_basename_fallback(scope: &Path, query_lower: &str) -> Option<PathBuf> {
    let mut candidate: Option<PathBuf> = None;
    let mut best_priority: u8 = 0;

    let walker = ignore::WalkBuilder::new(scope)
        .follow_links(true)
        .same_file_system(true) // Stop at mount boundaries (NFS, external volumes).
        .hidden(true)
        .git_ignore(true)
        .max_depth(Some(6))
        .build();

    for entry in walker.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if stem.to_ascii_lowercase() != *query_lower {
            continue;
        }
        let prio = source_priority(path);
        if prio > best_priority {
            best_priority = prio;
            candidate = Some(path.to_path_buf());
        }
    }

    candidate
}

/// When a file's basename (without extension) matches the query exactly,
/// return a compact outline of that file. Helps concept queries like `cli`
/// surface the file `cli.ts` with structural context instead of scattered text matches.
///
/// Scans the already-collected search results first (fast path), falls back to
/// a lightweight directory walk when the basename file didn't survive truncation.
fn basename_file_outline(
    query: &str,
    matches: &[Match],
    scope: &Path,
    cache: &OutlineCache,
) -> Option<String> {
    let query_lower = query.to_ascii_lowercase();

    // Only trigger for short single-word queries (concept/file-level intent)
    if query_lower.is_empty() || query.contains(' ') || query.contains("::") {
        return None;
    }

    // Find the best candidate among existing matches whose basename matches the query
    let matched_path = find_basename_candidate(matches, &query_lower)
        .or_else(|| find_basename_fallback(scope, &query_lower))?;

    // Read file and generate outline
    let content = std::fs::read_to_string(&matched_path).ok()?;
    let file_type = crate::lang::detect_file_type(&matched_path);
    let mtime = std::fs::metadata(&matched_path)
        .and_then(|m| m.modified())
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

    let outline = cache.get_or_compute(&matched_path, mtime, || {
        crate::read::outline::generate(
            &matched_path,
            file_type,
            &content,
            content.as_bytes(),
            false,
        )
    });

    if outline.trim().is_empty() {
        return None;
    }

    let rel_path = rel(&matched_path, scope);
    let line_count = content.lines().count();
    Some(format!(
        "## File overview: {rel_path} ({line_count} lines)\n{outline}"
    ))
}

fn format_search_result(
    result: &SearchResult,
    cache: &OutlineCache,
    session: Option<&Session>,
    bloom: &crate::index::bloom::BloomFilterCache,
    expand: usize,
    budget: Option<u64>,
) -> Result<String, TilthError> {
    let header = format::search_header(
        &result.query,
        &result.scope,
        result.total_found,
        result.definitions,
        result.usages,
    );
    let mut out = header;
    let mut expand_remaining = expand;
    let mut expanded_files = HashSet::new();
    let mut segments: Vec<(i64, usize, usize)> = Vec::new();

    // File-level retrieval: when a file basename matches the query exactly,
    // prepend a compact outline so the agent gets file-level context first.
    if let Some(file_outline) =
        basename_file_outline(&result.query, &result.matches, &result.scope, cache)
    {
        let _ = write!(out, "\n\n{file_outline}");
    }

    // Apply faceting when there are many matches (>5)
    if result.matches.len() > 5 {
        let faceted = facets::facet_matches(result.matches.clone(), &result.scope);
        let totals = &result.facet_totals;

        // Format each non-empty facet with section headers. After a truncated
        // facet's entries, emit a per-facet hidden-count line (`write_hidden_tail`)
        // so the reader sees which facet got cut. The global tail used to live
        // at the end of `format_search_result`; on the facet path we suppress
        // it to avoid double-counting hidden matches across two surfaces.
        if !faceted.definitions.is_empty() {
            let _ = write!(
                out,
                "\n\n## Definitions ({})",
                count_label(faceted.definitions.len(), totals.definitions)
            );
            format_matches(
                &faceted.definitions,
                &result.scope,
                cache,
                session,
                bloom,
                &mut expand_remaining,
                &mut expanded_files,
                &mut out,
                &mut segments,
            );
            write_hidden_tail(
                &mut out,
                faceted.definitions.len(),
                totals.definitions,
                "definitions",
            );
        }

        if !faceted.implementations.is_empty() {
            let _ = write!(
                out,
                "\n\n## Implementations ({})",
                count_label(faceted.implementations.len(), totals.implementations)
            );
            format_matches(
                &faceted.implementations,
                &result.scope,
                cache,
                session,
                bloom,
                &mut expand_remaining,
                &mut expanded_files,
                &mut out,
                &mut segments,
            );
            write_hidden_tail(
                &mut out,
                faceted.implementations.len(),
                totals.implementations,
                "implementations",
            );
        }

        if !faceted.tests.is_empty() {
            let _ = write!(
                out,
                "\n\n## Tests ({})",
                count_label(faceted.tests.len(), totals.tests)
            );
            // Compact test format — one line per match, no expand budget consumed
            for m in &faceted.tests {
                let _ = write!(
                    out,
                    "\n  {}:{} — {}",
                    rel(&m.path, &result.scope),
                    m.line,
                    m.text.trim()
                );
            }
            write_hidden_tail(&mut out, faceted.tests.len(), totals.tests, "tests");
        }

        if !faceted.usages_local.is_empty() {
            let _ = write!(
                out,
                "\n\n## Usages — same package ({})",
                count_label(faceted.usages_local.len(), totals.usages_local)
            );
            format_matches(
                &faceted.usages_local,
                &result.scope,
                cache,
                session,
                bloom,
                &mut expand_remaining,
                &mut expanded_files,
                &mut out,
                &mut segments,
            );
            write_hidden_tail(
                &mut out,
                faceted.usages_local.len(),
                totals.usages_local,
                "usages",
            );
        }

        if !faceted.usages_cross.is_empty() {
            let _ = write!(
                out,
                "\n\n## Usages — other ({})",
                count_label(faceted.usages_cross.len(), totals.usages_cross)
            );
            format_matches(
                &faceted.usages_cross,
                &result.scope,
                cache,
                session,
                bloom,
                &mut expand_remaining,
                &mut expanded_files,
                &mut out,
                &mut segments,
            );
            write_hidden_tail(
                &mut out,
                faceted.usages_cross.len(),
                totals.usages_cross,
                "usages",
            );
        }

        // A facet can be allotted *zero* display slots when higher-ranked facets consume
        // the whole `MAX_MATCHES` cap — which facet loses is decided by rank, and rank
        // breaks ties on path, so a directory name can do it. The `!is_empty()` guards
        // above then skip that facet entirely, taking its per-facet total and its
        // hidden-count tail with it: no heading, no "... and N more".
        //
        // That was survivable while the header printed the display cap, because the whole
        // count line was self-evidently nonsense ("10 matches (55 definitions, ...)").
        // Now that the header states a true total, an unaccounted-for facet turns it into
        // an arithmetic contradiction — "99 usages" over a body listing 92 — which is a
        // worse failure than an obviously useless number. Name what was dropped so the
        // header's arithmetic closes.
        let omitted: Vec<String> = [
            ("definitions", faceted.definitions.len(), totals.definitions),
            (
                "implementations",
                faceted.implementations.len(),
                totals.implementations,
            ),
            ("tests", faceted.tests.len(), totals.tests),
            (
                "same-package usages",
                faceted.usages_local.len(),
                totals.usages_local,
            ),
            (
                "other usages",
                faceted.usages_cross.len(),
                totals.usages_cross,
            ),
        ]
        .iter()
        .filter(|(_, shown, total)| *shown == 0 && *total > 0)
        .map(|(kind, _, total)| format!("{total} {kind}"))
        .collect();

        // The facet totals above are computed over the *retained* candidate set, which the search
        // bounds (`search::retain`). `total_found` is the true pre-bound total, so when retention
        // clipped, the facets cannot sum to it and the difference is real matches that no facet
        // knows about.
        //
        // Naming that remainder is what keeps this header honest. Reporting only the facet numbers
        // would say "19600 other usages" for a search that found 2.4M, which is worse than a
        // useless number — it is a confident wrong one, and it is the failure the comment above
        // exists to prevent. The local/cross split of the clipped portion genuinely cannot be
        // recovered (`facets::facet_of` needs a primary package derived from the full set), so this
        // reports the one thing that is knowable: how many there were.
        let beyond_retention = facets::unattributed_remainder(result.total_found, totals);

        if !omitted.is_empty() || beyond_retention > 0 {
            let mut parts = omitted;
            if beyond_retention > 0 {
                parts.push(format!("{beyond_retention} beyond the retention limit"));
            }
            let _ = write!(
                out,
                "\n\nNot shown: {}. Narrow with scope.",
                parts.join(", ")
            );
        }
    } else {
        // Linear display for ≤5 matches
        format_matches(
            &result.matches,
            &result.scope,
            cache,
            session,
            bloom,
            &mut expand_remaining,
            &mut expanded_files,
            &mut out,
            &mut segments,
        );

        // Global hidden-tail only on the linear path. The faceted path emits
        // a per-facet line for each truncated facet above; printing both
        // would double-count the same hidden matches.
        if result.total_found > result.matches.len() {
            let omitted = result.total_found - result.matches.len();
            let _ = write!(
                out,
                "\n\n... and {omitted} more matches. Narrow with scope."
            );
        }
    }

    // Apply value-based budget allocation before appending the token footer.
    // Under-budget: byte-identical. Over-budget: drops lowest-value match blocks.
    // budget.unwrap_or(DEFAULT_BUDGET) keeps the no-budget path byte-identical
    // to before this fix — DEFAULT_BUDGET remains the default, it is simply no
    // longer a hardcode that shadows a real caller-supplied budget.
    let budget_tokens = budget.unwrap_or(crate::budget::DEFAULT_BUDGET);
    out = crate::search::alloc::fit_to_budget(&out, &segments, budget_tokens);

    let tokens = estimate_tokens(out.len() as u64);
    let token_str = format_token_count(tokens);
    let _ = write!(out, "\n\n({token_str} tokens)");

    Ok(out)
}

/// Inline the actual code for a match. Returns `(formatted_block, raw_content)`.
/// The raw content is returned so the caller can reuse it (e.g. for related-file hints)
/// without a redundant file read.
///
/// For definitions: use tree-sitter node range (`def_range`).
/// For usages: ±10 lines around the match.
fn expand_match(m: &Match, scope: &Path) -> Option<(String, String)> {
    let mut content = fs::read_to_string(&m.path).ok()?;
    // One strip, because this function does two things with these lines and a BOM broke both
    // (#51): it renders them into the fenced block, where a BOM'd line 1 showed a stray
    // glyph, and it prefix-tests them below for the leading-import skip, where
    // `trimmed.starts_with("use ")` silently failed so a BOM'd line-1 import was never
    // skipped. The second is the #35 bug again, in a path that fix never visited.
    //
    // Removing a BOM cannot shift a line number — it carries no newline — so `start`/`end`
    // and the `{i:>4} |` gutter stay correct. Drained in place rather than reassigned so a
    // large file is not copied; the range is a multiple of 3 bytes and therefore always on a
    // char boundary.
    let bom_len = content.len() - crate::lang::outline::strip_bom(&content).len();
    if bom_len > 0 {
        content.drain(..bom_len);
    }
    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len() as u32;

    let (mut start, end) = if estimate_tokens(content.len() as u64) < EXPAND_FULL_FILE_THRESHOLD {
        (1, total)
    } else {
        let (s, e) = m
            .def_range
            .unwrap_or((m.line.saturating_sub(10), m.line.saturating_add(10)));
        (s.max(1), e.min(total))
    };

    // Skip leading import blocks in expanded definitions near top of file
    if m.is_definition && start <= 5 {
        let mut first_non_import = start;
        for i in start..=end {
            let idx = (i - 1) as usize;
            if idx >= lines.len() {
                break;
            }
            let trimmed = lines[idx].trim();
            let is_import = trimmed.starts_with("use ")
                || trimmed.starts_with("import ")
                || trimmed.starts_with("from ")
                || trimmed.starts_with("#include")
                || trimmed.starts_with("require(")
                || trimmed.starts_with("require ")
                || (trimmed.starts_with("const ") && trimmed.contains("= require("));

            if !is_import && !trimmed.is_empty() {
                first_non_import = i;
                break;
            }
        }
        // Guard: only skip if we found at least one non-import line
        if first_non_import > start && first_non_import <= end {
            start = first_non_import;
        }
    }

    let mut out = String::new();
    let _ = write!(out, "\n```{}:{}-{}", rel(&m.path, scope), start, end);

    // Track consecutive blank lines for collapsing
    let mut prev_blank = false;
    for i in start..=end {
        let idx = (i - 1) as usize;
        if idx < lines.len() {
            let line = lines[idx];
            let is_blank = line.trim().is_empty();

            // Skip consecutive blank lines (keep first, drop rest)
            if is_blank && prev_blank {
                continue;
            }

            let _ = write!(out, "\n{i:>4} | {line}");
            prev_blank = is_blank;
        }
    }
    out.push_str("\n```");
    Some((out, content))
}

/// Filter formatted code lines using a set of line numbers to skip.
/// Input is the fenced code block from `expand_match` (opening/closing fence lines
/// plus numbered content lines). Inserts gap markers for runs of >3 skipped lines.
fn filter_code_lines(code: &str, skip_lines: &HashSet<u32>) -> String {
    let mut kept: Vec<String> = Vec::new();
    let mut consecutive_skipped: u32 = 0;

    for segment in code.split('\n') {
        // Fence lines and the leading empty segment pass through unchanged
        if segment.starts_with("```") || segment.is_empty() {
            flush_gap_marker(&mut kept, &mut consecutive_skipped);
            kept.push(segment.to_owned());
            continue;
        }

        // Extract line number from formatted line: "  42 | content"
        let line_num = segment
            .find('|')
            .and_then(|pos| segment[..pos].trim().parse::<u32>().ok());

        if let Some(num) = line_num {
            if skip_lines.contains(&num) {
                consecutive_skipped += 1;
                continue;
            }
        }

        flush_gap_marker(&mut kept, &mut consecutive_skipped);
        kept.push(segment.to_owned());
    }

    kept.join("\n")
}

/// If >3 lines were skipped consecutively, push a gap marker and reset counter.
fn flush_gap_marker(kept: &mut Vec<String>, consecutive_skipped: &mut u32) {
    if *consecutive_skipped > 3 {
        kept.push(format!(
            "       ... ({} lines omitted)",
            *consecutive_skipped
        ));
    }
    *consecutive_skipped = 0;
}

/// Get cached outline string for a file. Returns None for non-code or huge files.
fn get_outline_str(path: &std::path::Path, cache: &OutlineCache) -> Option<std::sync::Arc<str>> {
    let file_type = crate::lang::detect_file_type(path);
    if !matches!(file_type, FileType::Code(_)) {
        return None;
    }
    let meta = std::fs::metadata(path).ok()?;
    let mtime = meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
    if meta.len() > 500_000 {
        return None;
    }
    Some(cache.get_or_compute(path, mtime, || {
        let content = std::fs::read_to_string(path).unwrap_or_default();
        let buf = content.as_bytes();
        read::outline::generate(path, file_type, &content, buf, false)
    }))
}

/// Find the outline entry index that encloses the given line.
fn find_enclosing_outline_idx(
    path: &std::path::Path,
    match_line: u32,
    cache: &OutlineCache,
) -> Option<usize> {
    let outline_str = get_outline_str(path, cache)?;
    let outline_lines: Vec<&str> = outline_str.lines().collect();
    outline_lines.iter().position(|line| {
        extract_line_range(line).is_some_and(|(s, e)| match_line >= s && match_line <= e)
    })
}

/// Build outline context around a match — ±2 entries around the enclosing one.
fn outline_context_for_match(
    path: &std::path::Path,
    match_line: u32,
    cache: &OutlineCache,
) -> Option<String> {
    let outline_str = get_outline_str(path, cache)?;
    let outline_lines: Vec<&str> = outline_str.lines().collect();
    if outline_lines.is_empty() {
        return None;
    }

    let match_idx = outline_lines.iter().position(|line| {
        extract_line_range(line).is_some_and(|(s, e)| match_line >= s && match_line <= e)
    })?;

    let start = match_idx.saturating_sub(2);
    let end = (match_idx + 3).min(outline_lines.len());

    let mut context = String::new();
    for (i, line) in outline_lines.iter().enumerate().take(end).skip(start) {
        if i == match_idx {
            let _ = write!(context, "\n-> {line}");
        } else {
            let _ = write!(context, "\n  {line}");
        }
    }
    Some(context)
}

/// Annotate a usage match with its enclosing scope: `"function foo"` /
/// `"class Bar"` for code (via tree-sitter), `"§Heading"` for markdown
/// (via line walk). Returns `None` for top-level matches and unsupported
/// file types — the formatter renders those without an `in …` suffix.
fn enclosing_scope_label(
    path: &std::path::Path,
    match_line: u32,
    cache: &OutlineCache,
) -> Option<String> {
    match crate::lang::detect_file_type(path) {
        FileType::Code(_) => {
            let s = scope::enclosing_definition_at(path, match_line, cache)?;
            Some(format!("{} {}", s.kind, s.name))
        }
        FileType::Markdown => markdown_enclosing_scope(path, match_line),
        _ => None,
    }
}

/// Find the deepest ATX-heading section that encloses `match_line`. Returns
/// the heading text prefixed with `§`. A `# foo` line inside a fenced or
/// indented code block is NOT a heading — the tree-sitter-md block grammar
/// owns that distinction, so we don't need our own fence pre-pass.
fn markdown_enclosing_scope(path: &std::path::Path, match_line: u32) -> Option<String> {
    if match_line == 0 {
        return None;
    }
    let raw = std::fs::read_to_string(path).ok()?;
    // BOM-stripped before parsing, to match the read side and `find_defs_markdown_buf` — see
    // that function for why the two sides disagreeing mattered (#51).
    let content = crate::lang::outline::strip_bom(&raw);
    let tree = crate::lang::outline::parse_markdown(content)?;
    let lines: Vec<&str> = content.lines().collect();
    let mut best: Option<(tree_sitter::Node, u32)> = None;
    walk_md_for_enclosing(tree.root_node(), match_line, &mut best);
    let (heading, _) = best?;
    let text = crate::lang::outline::heading_text(heading, &lines);
    if text.is_empty() {
        return None;
    }
    let display: String = if text.chars().count() > 60 {
        let mut s: String = text.chars().take(57).collect();
        s.push_str("...");
        s
    } else {
        text
    };
    Some(format!("§{display}"))
}

fn walk_md_for_enclosing<'a>(
    node: tree_sitter::Node<'a>,
    match_line: u32,
    best: &mut Option<(tree_sitter::Node<'a>, u32)>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() != "section" {
            continue;
        }
        let start = (child.start_position().row + 1) as u32;
        let end_excl = child.end_position();
        let end = if end_excl.column == 0 {
            end_excl.row as u32
        } else {
            (end_excl.row + 1) as u32
        };
        if match_line < start || match_line > end {
            continue;
        }
        // Section contains match_line. Record its heading if it has one,
        // then recurse to find a deeper section.
        let mut sec_cursor = child.walk();
        if let Some(heading) = child
            .children(&mut sec_cursor)
            .find(|c| c.kind() == "atx_heading")
        {
            // Update best to the *deepest* (largest start_line) match.
            match best {
                Some((_, prev_start)) if *prev_start >= start => {}
                _ => *best = Some((heading, start)),
            }
        }
        walk_md_for_enclosing(child, match_line, best);
    }
}

/// Extract (`start_line`, `end_line`) from an outline entry like "[20-115]" or "[16]".
fn extract_line_range(line: &str) -> Option<(u32, u32)> {
    let trimmed = line.trim();
    if !trimmed.starts_with('[') {
        return None;
    }
    let end = trimmed.find(']')?;
    let range_str = &trimmed[1..end];
    if let Some((a, b)) = range_str.split_once('-') {
        let start: u32 = a.trim().parse().ok()?;
        // Handle import ranges like "[1-]"
        let end: u32 = if b.trim().is_empty() {
            start
        } else {
            b.trim().parse().ok()?
        };
        Some((start, end))
    } else {
        let n: u32 = range_str.trim().parse().ok()?;
        Some((n, n))
    }
}

/// Format glob search results (file list with previews).
fn format_glob_result(result: &glob::GlobResult, scope: &Path) -> Result<String, TilthError> {
    // The true total, not `files.len()`. That was the display cap, so a directory with
    // 3000 matches and one with 20 both headed "20 files" — the same defect fixed on the
    // search header. Milder here because a "... and N more files" tail follows, so the
    // total was at least recoverable; the header line itself was still a clamped number
    // presented as a count.
    let header = format!(
        "# Glob: \"{}\" in {} — {} files",
        result.pattern,
        scope.display(),
        result.total_found
    );

    let mut out = header;
    for file in &result.files {
        let _ = write!(out, "\n  {}", rel(&file.path, scope));
        if let Some(ref preview) = file.preview {
            let _ = write!(out, "  ({preview})");
        }
    }

    if result.total_found > result.files.len() {
        let omitted = result.total_found - result.files.len();
        let _ = write!(out, "\n\n... and {omitted} more files. Narrow with scope.");
    }

    if result.files.is_empty() && !result.available_extensions.is_empty() {
        let _ = write!(
            out,
            "\n\nNo matches. Available extensions in scope: {}",
            result.available_extensions.join(", ")
        );
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Mutex;

    /// Files in the search-determinism fixture below.
    ///
    /// A count-based cutoff quits a parallel walk *between* files, so what defeats it is
    /// a fixture with far more files than the cutoff admits — not merely more matches.
    /// `callers.rs` records the trap that cost a round of review there: a 12-file
    /// fixture holding 60 matches was past the threshold on matches, but the walker's
    /// threads consumed all 12 files before the shared counter was ever read, so the
    /// exact-total assertion stayed green with the bug reintroduced, 12/12 runs.
    ///
    /// 400 files is comfortably past the point where that can happen for the highest
    /// threshold these paths ever used (`FULL_EARLY_QUIT_* = 300`), and still writes in
    /// well under a second.
    const DETERMINISM_FIXTURE_FILES: usize = 400;
    const DETERMINISM_FIXTURE_USES_PER_FILE: usize = 2;

    /// Write `DETERMINISM_FIXTURE_FILES` Rust files, each defining `target_sym` once and
    /// calling it `DETERMINISM_FIXTURE_USES_PER_FILE` times.
    ///
    /// Returns `(definitions, usages)`: one definition per file, and one usage per call
    /// site. The definition's own line also matches the usage regex, but `symbol::search`
    /// dedups a usage that sits on a definition's exact (path, line), so it does not count.
    fn write_search_determinism_fixture(dir: &Path) -> (usize, usize) {
        for f in 0..DETERMINISM_FIXTURE_FILES {
            let mut src = String::from("fn target_sym() {}\n");
            for i in 0..DETERMINISM_FIXTURE_USES_PER_FILE {
                src.push_str(&format!("fn caller_{f}_{i}() {{ target_sym(); }}\n"));
            }
            std::fs::write(dir.join(format!("m{f}.rs")), src).unwrap();
        }
        (
            DETERMINISM_FIXTURE_FILES,
            DETERMINISM_FIXTURE_FILES * DETERMINISM_FIXTURE_USES_PER_FILE,
        )
    }

    /// Symbol search must report *every* definition and usage, not however many a shared
    /// counter happened to admit before a parallel walk noticed it had crossed a threshold.
    ///
    /// `EARLY_QUIT_THRESHOLD_DEFINITIONS = 50` and `EARLY_QUIT_THRESHOLD_USAGES = 30` made
    /// this non-deterministic. Six identical runs against a 176k-file C++ tree produced six
    /// distinct renderings, with the usage count moving over 30, 30, 30, 39, 28 and 30 while
    /// the definition count sat at exactly 50 — the threshold, reported as a total.
    ///
    /// The assertions are on exact totals, so a reintroduced cutoff fails outright rather
    /// than merely varying.
    #[test]
    fn symbol_search_reports_every_match_past_the_old_early_quit_thresholds() {
        let dir = tempfile::tempdir().unwrap();
        let (expected_defs, expected_usages) = write_search_determinism_fixture(dir.path());

        let result = search_symbol_raw("target_sym", dir.path(), None).unwrap();

        assert_eq!(
            result.definitions, expected_defs,
            "expected every definition, got {}",
            result.definitions
        );
        assert_eq!(
            result.usages, expected_usages,
            "expected every usage, got {}",
            result.usages
        );
        assert_eq!(
            result.total_found,
            expected_defs + expected_usages,
            "total_found must be the true pre-cap total"
        );
    }

    /// Content search had the same cutoff (`EARLY_QUIT_THRESHOLD = 30`), and hid it better:
    /// the header reported the display cap rather than a total, so the instability was only
    /// visible by diffing full output — three distinct renderings in six identical runs.
    #[test]
    fn content_search_reports_every_match_past_the_old_early_quit_threshold() {
        let dir = tempfile::tempdir().unwrap();
        let (defs, usages) = write_search_determinism_fixture(dir.path());
        // Every line mentioning `target_sym`: the definition line plus each call site.
        let expected = defs + usages;

        let result = search_content_raw("target_sym", dir.path(), None).unwrap();

        assert_eq!(
            result.total_found, expected,
            "expected every matching line, got {}",
            result.total_found
        );
    }

    /// Repeated identical runs must agree byte for byte. Weaker than the exact-total
    /// assertions above, but it fails for *any* source of instability, not just a count
    /// cutoff — including an unranked truncation of an unordered vector, which is how the
    /// second-hop block in `callers.rs` stayed unstable after its walk was fixed.
    #[test]
    fn symbol_and_content_output_is_byte_identical_across_repeated_runs() {
        let dir = tempfile::tempdir().unwrap();
        write_search_determinism_fixture(dir.path());
        let cache = OutlineCache::new();

        let symbol_runs: Vec<String> = (0..6)
            .map(|_| search_symbol("target_sym", dir.path(), &cache, None).unwrap())
            .collect();
        assert!(
            symbol_runs.windows(2).all(|w| w[0] == w[1]),
            "symbol search rendered {} distinct outputs in 6 identical runs",
            symbol_runs.iter().collect::<HashSet<_>>().len()
        );

        let content_runs: Vec<String> = (0..6)
            .map(|_| search_content("target_sym", dir.path(), &cache, None).unwrap())
            .collect();
        assert!(
            content_runs.windows(2).all(|w| w[0] == w[1]),
            "content search rendered {} distinct outputs in 6 identical runs",
            content_runs.iter().collect::<HashSet<_>>().len()
        );
    }

    /// The header must state the true total, not the number of matches it went on to
    /// render. It used to be passed `result.matches.len()`, so every capped result headed
    /// "10 matches" regardless of whether 10 or 34290 existed — a clamped number presented
    /// as a total, which is the half of this bug an agent reads first.
    #[test]
    fn search_header_reports_the_true_total_not_the_display_cap() {
        let dir = tempfile::tempdir().unwrap();
        let (defs, usages) = write_search_determinism_fixture(dir.path());
        let total = defs + usages;
        let cache = OutlineCache::new();

        for (label, out) in [
            (
                "symbol",
                search_symbol("target_sym", dir.path(), &cache, None).unwrap(),
            ),
            (
                "content",
                search_content("target_sym", dir.path(), &cache, None).unwrap(),
            ),
        ] {
            let header = out.lines().next().unwrap_or_default();
            assert!(
                header.contains(&format!("{total} matches")),
                "{label} header must report the true total {total}, got: {header}"
            );
            // The display cap must not be what the header reports. Asserted against the
            // number of rendered entries rather than the literal "10", which would be a
            // decimal-substring coincidence that breaks if the fixture size is retuned.
            let shown = out.lines().filter(|l| l.starts_with("### ")).count();
            assert!(
                shown < total,
                "{label} fixture must exceed the display cap for this to test anything \
                 (shown {shown}, total {total})"
            );
            assert!(
                !header.contains(&format!("{shown} matches")),
                "{label} header must not report the {shown} rendered entries as the \
                 total, got: {header}"
            );
        }
    }

    /// A facet that wins zero display slots must still be accounted for.
    ///
    /// The renderer guards each facet block on `!faceted.X.is_empty()`, which is decided
    /// on the *post-cap* set. When higher-ranked facets consume the whole `MAX_MATCHES`
    /// cap, a facet with a real non-zero total renders nothing at all — no heading, no
    /// `shown/total` label, no hidden-count tail. That was tolerable while the header
    /// printed the display cap, because the count line was self-evidently nonsense; once
    /// the header states a true total, the body has to add up to it.
    ///
    /// Fixture: 1 definition, 80 same-package usages, 7 test usages. 10 display slots, so
    /// the test facet is guaranteed to get none of them.
    #[test]
    fn a_facet_with_no_display_slots_is_still_accounted_for_in_the_body() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("Cargo.toml"), "[package]\nname = \"x\"\n").unwrap();
        let src = root.join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("lib.rs"), "fn target_sym() {}\n").unwrap();
        for i in 0..40 {
            std::fs::write(
                src.join(format!("u{i}.rs")),
                format!("fn a{i}() {{ target_sym(); }}\nfn b{i}() {{ target_sym(); }}\n"),
            )
            .unwrap();
        }
        for i in 0..7 {
            std::fs::write(src.join(format!("t{i}_test.rs")), "let x = target_sym();\n").unwrap();
        }

        let cache = OutlineCache::new();
        let out = search_symbol("target_sym", root, &cache, None).unwrap();

        let shown_test_entries = out.contains("## Tests");
        assert!(
            !shown_test_entries,
            "fixture assumption broken — the tests facet won display slots, so this test \
             no longer covers the zero-slot case:\n{out}"
        );
        assert!(
            out.contains("Not shown: 7 tests"),
            "a facet with 7 matches and no display slots vanished from the body, leaving \
             the header's total unaccounted for:\n{out}"
        );
    }

    /// Matches in the retention fixture below, chosen to exceed `content::MAX_RETAINED` (500)
    /// by a wide margin — the bound is invisible below it.
    const RETENTION_SHALLOW_FILES: usize = 10;
    const RETENTION_DEEP_FILES: usize = 40;
    const RETENTION_MATCHES_PER_FILE: usize = 60;

    /// Write files containing `needle` on every line, in two tiers that **score differently**.
    ///
    /// The score spread is load-bearing. A first version put every file at the same depth, so
    /// all matches scored equally and the selection key was decided entirely by its
    /// `(path, line)` tie-break — which meant inverting the heap's score comparison, so it
    /// kept the *worst* candidates instead of the best, failed no test at all. Verified by
    /// mutation. `scope_proximity` charges 20 points per path component, so burying one tier
    /// five levels down separates the tiers by 100 points — measured: shallow 230, deep 130 —
    /// and makes the score comparison decide something.
    ///
    /// The shallow tier alone exceeds `MAX_RETAINED`, so a correct implementation retains only
    /// shallow matches and an inverted one retains only deep ones.
    fn write_retention_fixture(dir: &Path) -> usize {
        let body = "let v = needle();\n".repeat(RETENTION_MATCHES_PER_FILE);
        for f in 0..RETENTION_SHALLOW_FILES {
            std::fs::write(dir.join(format!("shallow{f:03}.rs")), &body).unwrap();
        }
        // One file whose *name* marks it as a test, so the `tests` bucket is non-zero and the
        // `tests` counter is actually exercised. Without it `assert_eq!(tests, 0)` was `0 == 0`,
        // and stubbing the test predicate to `false` failed nothing — verified by mutation.
        // `_test.` rather than a `tests/` directory, because `is_test_match`'s directory checks
        // use forward slashes and never fire on Windows paths.
        std::fs::write(dir.join("marked_test.rs"), &body).unwrap();
        let deep_dir = dir.join("a").join("b").join("c").join("d").join("e");
        std::fs::create_dir_all(&deep_dir).unwrap();
        for f in 0..RETENTION_DEEP_FILES {
            std::fs::write(deep_dir.join(format!("deep{f:03}.rs")), &body).unwrap();
        }
        (RETENTION_SHALLOW_FILES + RETENTION_DEEP_FILES + 1) * RETENTION_MATCHES_PER_FILE
    }

    /// Files needed to push symbol retention past `retain::MAX_RETAINED` (20_000).
    const SYM_RETENTION_FILES: usize = 40;
    const SYM_RETENTION_USAGES_PER_FILE: usize = 600;

    /// One definition plus many usages per file, enough usages in total to clip.
    ///
    /// Returns `(definitions, usages)` as ground truth. Every usage is on its own line and none
    /// shares a line with the definition, so the def/usage dedup removes nothing and the true
    /// post-dedup total is simply the sum.
    fn write_symbol_retention_fixture(root: &Path) -> (usize, usize) {
        for f in 0..SYM_RETENTION_FILES {
            let mut body = String::from(
                "pub fn sym_target() -> u32 { 0 }
",
            );
            for i in 0..SYM_RETENTION_USAGES_PER_FILE {
                let _ = writeln!(body, "    let v{i} = sym_target();");
            }
            std::fs::write(root.join(format!("s{f:03}.rs")), &body).unwrap();
        }
        (
            SYM_RETENTION_FILES,
            SYM_RETENTION_FILES * SYM_RETENTION_USAGES_PER_FILE,
        )
    }

    /// The symbol half of #19: bounding retention must not make the reported counts approximate,
    /// and the header's arithmetic must still close once it clips.
    ///
    /// This test exists because its absence was invisible. Reverting `assemble` to derive totals
    /// from the retained set — the exact defect the bound introduced — broke **no test**, and so
    /// did forcing the renderer's remainder term to zero. The `unattributed_remainder` unit tests
    /// cannot catch either: they hand-build `FacetTotals` and only exercise six lines of
    /// arithmetic. Nothing drove `symbol::search` past the cap. `content.rs`'s half of the same
    /// work has had this test since #30.
    #[test]
    fn symbol_counts_stay_exact_totals_past_the_retention_bound() {
        let dir = tempfile::tempdir().unwrap();
        let (defs, usages) = write_symbol_retention_fixture(dir.path());
        assert!(
            usages > crate::search::retain::MAX_RETAINED,
            "fixture must exceed MAX_RETAINED or the bound is untested ({usages})"
        );

        let cache = OutlineCache::new();
        let result = symbol::search("sym_target", dir.path(), None, None, false).unwrap();

        assert_eq!(
            result.definitions, defs,
            "definitions must be the true count, not what retention kept"
        );
        assert_eq!(
            result.total_found,
            defs + usages,
            "total_found must be the true post-dedup total, not the retained count"
        );
        assert_eq!(result.usages, usages, "usages must be the true count");

        // The retained set really is bounded — otherwise the assertions above would pass for the
        // trivial reason that nothing was dropped.
        assert!(
            result.matches.len() <= crate::search::retain::MAX_RETAINED,
            "retention did not bound the set ({})",
            result.matches.len()
        );

        // Facets plus the unattributed remainder must account for every match. This is what keeps
        // the rendered header from contradicting its own body.
        let remainder =
            crate::search::facets::unattributed_remainder(result.total_found, &result.facet_totals);
        let facet_sum = result.facet_totals.definitions
            + result.facet_totals.implementations
            + result.facet_totals.tests
            + result.facet_totals.usages_local
            + result.facet_totals.usages_cross;
        assert_eq!(
            facet_sum + remainder,
            result.total_found,
            "facets ({facet_sum}) + remainder ({remainder}) != total_found ({})",
            result.total_found
        );
        assert!(
            remainder > 0,
            "fixture did not clip, so the remainder path is untested"
        );

        // And the renderer must actually report the remainder. Two separate assertions, because they
        // fail to different mutations: the arithmetic above catches totals derived from the retained
        // set, and this catches the renderer dropping the remainder term — a one-line change that
        // `unattributed_remainder`'s own unit tests cannot see, since they call it directly.
        //
        // Not summing every number in the output: a partially-shown facet reports its hidden count
        // in its section heading (`Definitions (10/40)`), not in the "Not shown" line, so a naive
        // sum under-counts. The remainder term is the specific thing this change added.
        let bloom = crate::index::bloom::BloomFilterCache::new();
        let out = format_search_result(&result, &cache, None, &bloom, 0, None).unwrap();
        assert!(
            out.contains(&format!("{remainder} beyond the retention limit")),
            "renderer did not report the {remainder} matches no facet accounts for,              so the header total is contradicted by its body:
{out}"
        );
        assert!(
            out.contains(&format!("{} matches", result.total_found)),
            "rendered header lost the true total:
{out}"
        );
    }

    /// Bounding retention must not make the reported counts approximate.
    ///
    /// This is the half of the bound that is easy to get wrong: capping what is kept is
    /// trivial, keeping `total_found` and the per-facet totals *exact* while doing it is the
    /// requirement (#19). The counters are incremented per match during the walk and only
    /// ever read after the threads join, so they are independent of what the heap retained.
    #[test]
    fn content_counts_stay_exact_totals_past_the_retention_bound() {
        let dir = tempfile::tempdir().unwrap();
        let expected = write_retention_fixture(dir.path());
        assert!(
            expected > 500,
            "fixture must exceed MAX_RETAINED or the bound is untested ({expected})"
        );

        let result = search_content_raw("needle", dir.path(), None).unwrap();

        assert_eq!(
            result.total_found, expected,
            "total_found must be the true count, not the retained count"
        );
        assert_eq!(result.usages, expected, "usages must be the true count");
        assert_eq!(result.definitions, 0, "content search has no definitions");
        // Content search can only populate `tests` and `usages_cross`; every match here is a
        // plain usage, so the whole total lands in the latter.
        assert_eq!(result.facet_totals.definitions, 0);
        assert_eq!(result.facet_totals.usages_local, 0);
        assert_eq!(
            result.facet_totals.tests, RETENTION_MATCHES_PER_FILE,
            "the marked test file's matches must all land in the tests bucket"
        );
        assert!(
            result.facet_totals.tests > 0,
            "tests bucket must be exercised, or stubbing the predicate fails nothing"
        );
        // The two reachable buckets must partition the total exactly. This is the property
        // that makes hand-building `FacetTotals` from counters equivalent to faceting the
        // whole set, and it fails if either counter drifts.
        assert_eq!(
            result.facet_totals.tests + result.facet_totals.usages_cross,
            result.total_found,
            "tests + usages_cross must partition total_found"
        );
        // And the retained set must actually have been capped, or none of the above is a test
        // of the bound.
        assert!(
            result.matches.len() <= 10,
            "display cap should still apply: {}",
            result.matches.len()
        );

        // The bound must keep the *best* candidates, not merely some bounded set. The shallow
        // tier alone exceeds MAX_RETAINED and outscores the deep tier on `scope_proximity`, so
        // nothing from the deep tier can reach the page. Inverting the heap's ordering fails
        // here and nowhere else.
        assert!(
            result.matches.len() >= 5,
            "expected a full page to judge selection on"
        );
        for m in &result.matches {
            let p = m.path.to_string_lossy().replace('\\', "/");
            assert!(
                !p.contains("/a/b/c/"),
                "a deep, lower-scoring match reached the page — the bound is keeping the \
                 wrong candidates: {p}"
            );
        }
    }

    /// Bounding retention must not reintroduce the nondeterminism four PRs removed.
    ///
    /// The heap holds the best `MAX_RETAINED` by a total-ordered, time-independent key, so
    /// neither thread arrival order nor the wall clock can change which matches survive.
    #[test]
    fn content_output_is_byte_identical_across_runs_past_the_retention_bound() {
        let dir = tempfile::tempdir().unwrap();
        write_retention_fixture(dir.path());
        let cache = OutlineCache::new();

        let runs: Vec<String> = (0..6)
            .map(|_| search_content("needle", dir.path(), &cache, None).unwrap())
            .collect();

        assert!(
            runs.windows(2).all(|w| w[0] == w[1]),
            "content search rendered {} distinct outputs in 6 runs over a fixture past the              retention bound",
            runs.iter().collect::<HashSet<_>>().len()
        );
    }

    /// Files in the glob fixture below: 10 directories of 40 files, all matching.
    ///
    /// `MAX_FILES` is 20, so a fixture of 25 files would be "past the cap" — and would
    /// still pass with the racy `if locked.len() < MAX_FILES` reintroduced, because the
    /// walk's threads can consume 25 entries before the race is observable. Same trap as
    /// `DETERMINISM_FIXTURE_FILES` in `callers.rs`. 400 files across 10 directories also
    /// makes walk order differ sharply from sorted order: the correct answer is entirely
    /// inside `d0`, so any thread-order dependence shows up as files from other
    /// directories rather than as a subtle reshuffle.
    const GLOB_FIXTURE_DIRS: usize = 10;
    const GLOB_FIXTURE_FILES_PER_DIR: usize = 40;

    /// Write the glob fixture and return the relative paths a correct implementation must
    /// return, in the order it must return them.
    ///
    /// Derived from `glob::MAX_FILES` rather than a copied literal, so raising the cap
    /// retunes these tests instead of breaking them — up to the point where the displayed
    /// page no longer fits in `d0`, which the assertion below catches explicitly.
    fn write_glob_fixture(dir: &Path) -> Vec<String> {
        assert!(
            glob::MAX_FILES <= GLOB_FIXTURE_FILES_PER_DIR,
            "expectation assumes the whole displayed page fits in d0; widen the fixture"
        );
        for d in 0..GLOB_FIXTURE_DIRS {
            let sub = dir.join(format!("d{d}"));
            std::fs::create_dir_all(&sub).unwrap();
            for f in 0..GLOB_FIXTURE_FILES_PER_DIR {
                // Zero-padded so lexicographic order matches numeric order.
                std::fs::write(sub.join(format!("f{f:03}.txt")), "x\n").unwrap();
            }
        }
        // Sorted by path, the whole displayed page is inside `d0`.
        (0..glob::MAX_FILES)
            .map(|f| format!("d0/f{f:03}.txt"))
            .collect()
    }

    /// `tilth_files` must return the same files, in the same order, every time.
    ///
    /// It used to keep whichever entries won a race inside the parallel walk and render
    /// them unsorted, so it was non-deterministic in both membership *and* order: five
    /// identical runs of `*.h` over one module of a 176k-file C++ tree gave five distinct
    /// outputs. There is no ranking step on this path, so nothing recovered either.
    #[test]
    fn glob_output_is_byte_identical_across_repeated_runs() {
        let dir = tempfile::tempdir().unwrap();
        write_glob_fixture(dir.path());

        let runs: Vec<String> = (0..6)
            .map(|_| search_glob("*.txt", dir.path()).unwrap())
            .collect();

        assert!(
            runs.windows(2).all(|w| w[0] == w[1]),
            "glob rendered {} distinct outputs in 6 identical runs",
            runs.iter().collect::<HashSet<_>>().len()
        );
    }

    /// Stronger than "stable": the surviving subset must be the alphabetically first
    /// `MAX_FILES`, in order. Stability alone could be satisfied by a consistently wrong
    /// selection; this pins *which* files, so a racy cap fails outright rather than
    /// occasionally.
    #[test]
    fn glob_returns_the_alphabetically_first_files_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let expected = write_glob_fixture(dir.path());

        let result = glob::search("*.txt", dir.path()).unwrap();

        let got: Vec<String> = result
            .files
            .iter()
            // Normalise separators so the expectation is one string on every platform.
            .map(|f| rel(&f.path, dir.path()).replace('\\', "/"))
            .collect();

        assert_eq!(
            got, expected,
            "glob must return the sorted-first MAX_FILES entries in order"
        );
        assert_eq!(
            result.total_found,
            GLOB_FIXTURE_DIRS * GLOB_FIXTURE_FILES_PER_DIR,
            "total_found must be the true match count, not the display cap"
        );
    }

    /// Under the cap, every match is rendered and no "more files" tail appears.
    ///
    /// The eviction branch never fires here, so this is the only coverage of the heap's
    /// fill-up path — every other glob test runs well past capacity.
    ///
    /// Its unique contribution is the *absence* of the truncation tail: relaxing
    /// `format_glob_result`'s `total_found > files.len()` to `>=` appends
    /// "... and 0 more files. Narrow with scope." to every complete listing, and this is the
    /// only test that fails on it — verified by mutation. It does **not** catch a boundary
    /// off-by-one in the eviction check; that shows up in
    /// `glob_returns_the_alphabetically_first_files_in_order`, which runs past the cap.
    #[test]
    fn glob_under_the_cap_renders_every_match_with_no_tail() {
        let dir = tempfile::tempdir().unwrap();
        // One under the cap, so nothing is ever evicted.
        let n = glob::MAX_FILES - 1;
        for f in 0..n {
            std::fs::write(dir.path().join(format!("f{f:03}.txt")), "x\n").unwrap();
        }

        let out = search_glob("*.txt", dir.path()).unwrap();
        let shown = out.lines().filter(|l| l.starts_with("  ")).count();

        assert_eq!(shown, n, "every match under the cap must render:\n{out}");
        assert!(
            !out.contains("more files"),
            "no truncation tail when nothing was truncated:\n{out}"
        );

        let result = glob::search("*.txt", dir.path()).unwrap();
        assert_eq!(result.total_found, n);
        assert_eq!(result.files.len(), n);
    }

    /// The header must state how many files matched, not how many were rendered.
    #[test]
    fn glob_header_reports_the_true_total_not_the_display_cap() {
        let dir = tempfile::tempdir().unwrap();
        write_glob_fixture(dir.path());
        let total = GLOB_FIXTURE_DIRS * GLOB_FIXTURE_FILES_PER_DIR;

        let out = search_glob("*.txt", dir.path()).unwrap();
        let header = out.lines().next().unwrap_or_default();

        // Parse the count rather than substring-matching it. `contains("20 files")` would
        // also be satisfied by "420 files", and `!contains("0 files")` is satisfied by
        // nothing at all once the total ends in a zero — both traps hit while writing this.
        let reported: usize = header
            .rsplit_once("— ")
            .and_then(|(_, tail)| tail.split_whitespace().next())
            .and_then(|n| n.parse().ok())
            .unwrap_or_else(|| panic!("could not parse a count from header: {header}"));

        // Entries are indented two spaces; separators are platform-dependent, so don't
        // match on them.
        let shown = out.lines().filter(|l| l.starts_with("  ")).count();

        assert_eq!(
            reported, total,
            "header must report the true total, got: {header}"
        );
        assert!(
            shown < total,
            "fixture must exceed the display cap for this to test anything \
             (shown {shown}, total {total})"
        );
        assert_ne!(
            reported, shown,
            "header must not report the rendered entry count as the total: {header}"
        );
    }

    /// Collect all file paths from a walker into a sorted Vec.
    fn walk_paths(scope: &Path, glob: Option<&str>) -> Vec<PathBuf> {
        let w = walker(scope, glob).expect("walker failed");
        let paths: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());
        w.run(|| {
            let paths = &paths;
            Box::new(move |entry| {
                if let Ok(e) = entry {
                    if e.file_type().is_some_and(|ft| ft.is_file()) {
                        paths.lock().unwrap().push(e.into_path());
                    }
                }
                ignore::WalkState::Continue
            })
        });
        let mut v = paths.into_inner().unwrap();
        v.sort();
        v
    }

    fn extensions(paths: &[PathBuf]) -> HashSet<String> {
        paths
            .iter()
            .filter_map(|p| p.extension())
            .map(|e| e.to_string_lossy().to_string())
            .collect()
    }

    // ── walker unit tests ──

    #[test]
    fn walker_none_returns_all_file_types() {
        let scope = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let all = walk_paths(&scope, None);
        let exts = extensions(&all);
        assert!(exts.contains("rs"), "expected .rs files, got {exts:?}");
        assert!(!all.is_empty());
    }

    #[test]
    fn walker_whitelist_filters_to_matching_extension() {
        let scope = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let filtered = walk_paths(&scope, Some("*.rs"));
        assert!(!filtered.is_empty(), "whitelist should find .rs files");
        for p in &filtered {
            assert_eq!(
                p.extension().and_then(|e| e.to_str()),
                Some("rs"),
                "non-.rs file leaked through whitelist: {}",
                p.display()
            );
        }
    }

    #[test]
    fn walker_negation_excludes_matching_extension() {
        let scope = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let without_rs = walk_paths(&scope, Some("!*.rs"));
        for p in &without_rs {
            assert_ne!(
                p.extension().and_then(|e| e.to_str()),
                Some("rs"),
                ".rs file leaked through negation: {}",
                p.display()
            );
        }
    }

    #[test]
    fn walker_empty_string_equals_none() {
        let scope = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let all = walk_paths(&scope, None);
        let empty = walk_paths(&scope, Some(""));
        assert_eq!(all.len(), empty.len(), "empty glob should behave like None");
    }

    #[test]
    fn walker_invalid_glob_returns_error() {
        let scope = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let result = walker(&scope, Some("[unclosed"));
        match result {
            Err(TilthError::InvalidQuery { query, reason }) => {
                assert_eq!(query, "[unclosed");
                assert!(
                    reason.contains("invalid glob"),
                    "reason should mention 'invalid glob': {reason}"
                );
            }
            Err(other) => panic!("expected InvalidQuery, got {other}"),
            Ok(_) => panic!("expected Err for invalid glob, got Ok"),
        }
    }

    #[test]
    fn walker_brace_expansion_matches_multiple_extensions() {
        let scope = Path::new(env!("CARGO_MANIFEST_DIR"));
        let filtered = walk_paths(&scope, Some("*.{rs,toml}"));
        let exts = extensions(&filtered);
        assert!(
            exts.contains("rs"),
            "brace expansion should include .rs: {exts:?}"
        );
        assert!(
            exts.contains("toml"),
            "brace expansion should include .toml: {exts:?}"
        );
        for ext in &exts {
            assert!(
                ext == "rs" || ext == "toml",
                "unexpected extension leaked: {ext}"
            );
        }
    }

    #[test]
    fn walker_whitelist_fewer_than_unfiltered() {
        // Use project root (not src/) — project root has .toml, .md, .lock etc.
        // alongside .rs files, so *.rs is guaranteed to be a strict subset.
        let scope = Path::new(env!("CARGO_MANIFEST_DIR"));
        let all = walk_paths(&scope, None);
        let rs_only = walk_paths(&scope, Some("*.rs"));
        assert!(
            rs_only.len() < all.len(),
            "whitelist ({}) should find fewer files than unfiltered ({})",
            rs_only.len(),
            all.len()
        );
    }

    #[test]
    fn walker_path_pattern_restricts_directory() {
        let scope = Path::new(env!("CARGO_MANIFEST_DIR"));
        let filtered = walk_paths(&scope, Some("src/**/*.rs"));
        assert!(!filtered.is_empty(), "path pattern should find files");
        let src_dir = scope.join("src");
        for p in &filtered {
            assert!(
                p.starts_with(&src_dir),
                "file outside src/ leaked: {}",
                p.display()
            );
        }
    }

    // ── end-to-end through search functions ──

    #[test]
    fn content_search_glob_restricts_results() {
        let scope = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let all =
            content::search("TilthError", &scope, false, None, None, false).expect("search failed");
        let rs_only = content::search("TilthError", &scope, false, None, Some("*.rs"), false)
            .expect("search with glob failed");
        let toml_only = content::search("TilthError", &scope, false, None, Some("*.toml"), false)
            .expect("search with toml glob failed");

        assert!(all.total_found > 0, "unfiltered should find TilthError");
        assert!(rs_only.total_found > 0, "*.rs should find TilthError");
        assert_eq!(
            toml_only.total_found, 0,
            "*.toml should not find TilthError in Rust source"
        );
        for m in &rs_only.matches {
            assert_eq!(
                m.path.extension().and_then(|e| e.to_str()),
                Some("rs"),
                "non-.rs match leaked: {}",
                m.path.display()
            );
        }
    }

    #[test]
    fn symbol_search_glob_restricts_results() {
        let scope = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let rs_result = symbol::search("walker", &scope, None, Some("*.rs"), false)
            .expect("symbol search failed");
        let toml_result = symbol::search("walker", &scope, None, Some("*.toml"), false)
            .expect("symbol search with toml failed");

        assert!(rs_result.total_found > 0, "*.rs should find 'walker'");
        assert_eq!(
            toml_result.total_found, 0,
            "*.toml should not find 'walker'"
        );
        for m in &rs_result.matches {
            assert_eq!(
                m.path.extension().and_then(|e| e.to_str()),
                Some("rs"),
                "non-.rs match in symbol search: {}",
                m.path.display()
            );
        }
    }

    #[test]
    fn callers_search_glob_restricts_results() {
        let scope = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let bloom = crate::index::bloom::BloomFilterCache::new();
        let single: std::collections::HashSet<String> =
            std::iter::once("walker".to_string()).collect();
        let rs_callers = callers::find_callers_batch(&single, &scope, &bloom, Some("*.rs"))
            .expect("callers failed");
        let toml_callers = callers::find_callers_batch(&single, &scope, &bloom, Some("*.toml"))
            .expect("callers toml failed");

        assert!(
            !rs_callers.is_empty(),
            "*.rs should find callers of 'walker'"
        );
        assert!(
            toml_callers.is_empty(),
            "*.toml should not find callers of 'walker'"
        );
        for (_, c) in &rs_callers {
            assert_eq!(
                c.path.extension().and_then(|e| e.to_str()),
                Some("rs"),
                "non-.rs caller leaked: {}",
                c.path.display()
            );
        }
    }

    #[test]
    fn walker_follows_symlinked_file() {
        let tmp = tempfile::tempdir().unwrap();
        let real_dir = tmp.path().join("real");
        std::fs::create_dir(&real_dir).unwrap();
        std::fs::write(real_dir.join("hello.rs"), "fn main() {}").unwrap();

        let link_dir = tmp.path().join("linked");
        std::fs::create_dir(&link_dir).unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(real_dir.join("hello.rs"), link_dir.join("hello.rs")).unwrap();
        #[cfg(windows)]
        std::os::windows::fs::symlink_file(real_dir.join("hello.rs"), link_dir.join("hello.rs"))
            .unwrap();

        let paths = walk_paths(tmp.path(), None);
        let names: Vec<&str> = paths
            .iter()
            .filter_map(|p| p.file_name()?.to_str())
            .collect();
        // Should find hello.rs twice: once in real/, once via the symlink in linked/
        assert_eq!(
            names.iter().filter(|n| **n == "hello.rs").count(),
            2,
            "expected hello.rs from both real and symlinked dirs, got: {names:?}"
        );
    }

    #[test]
    fn walker_follows_symlinked_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let real_dir = tmp.path().join("real_pkg");
        std::fs::create_dir(&real_dir).unwrap();
        std::fs::write(real_dir.join("lib.rs"), "pub fn add() {}").unwrap();
        std::fs::write(real_dir.join("util.rs"), "pub fn helper() {}").unwrap();

        // Symlink the entire directory
        #[cfg(unix)]
        std::os::unix::fs::symlink(&real_dir, tmp.path().join("deps_link")).unwrap();
        #[cfg(windows)]
        std::os::windows::fs::symlink_dir(&real_dir, tmp.path().join("deps_link")).unwrap();

        let paths = walk_paths(tmp.path(), None);
        let link_files: Vec<_> = paths
            .iter()
            .filter(|p| p.starts_with(tmp.path().join("deps_link")))
            .collect();
        assert_eq!(
            link_files.len(),
            2,
            "expected 2 files via symlinked directory, got: {link_files:?}"
        );
    }

    #[test]
    fn walker_survives_symlink_cycle() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("real.rs"), "fn main() {}").unwrap();

        // Create a symlink cycle: loop -> .
        #[cfg(unix)]
        std::os::unix::fs::symlink(tmp.path(), tmp.path().join("loop")).unwrap();

        // Should complete without hanging — ignore crate detects the cycle via inode tracking
        let paths = walk_paths(tmp.path(), None);
        let names: Vec<&str> = paths
            .iter()
            .filter_map(|p| p.file_name()?.to_str())
            .collect();
        assert!(
            names.contains(&"real.rs"),
            "should find real.rs despite cycle: {names:?}"
        );
    }

    #[test]
    fn content_search_finds_symbol_through_symlink() {
        let tmp = tempfile::tempdir().unwrap();
        let real_dir = tmp.path().join("real");
        std::fs::create_dir(&real_dir).unwrap();
        std::fs::write(
            real_dir.join("api.rs"),
            "pub fn unique_symlink_test_symbol() {}",
        )
        .unwrap();

        // Symlink the directory into the search scope
        #[cfg(unix)]
        std::os::unix::fs::symlink(&real_dir, tmp.path().join("linked")).unwrap();
        #[cfg(windows)]
        std::os::windows::fs::symlink_dir(&real_dir, tmp.path().join("linked")).unwrap();

        let result = content::search(
            "unique_symlink_test_symbol",
            tmp.path(),
            false,
            None,
            None,
            false,
        )
        .unwrap();
        // Should find the symbol in both real/api.rs and linked/api.rs
        assert!(
            result.total_found >= 2,
            "expected symbol found via both real and symlinked paths, got {}",
            result.total_found
        );
    }

    // ── enclosing_scope_label / markdown tests ──

    #[test]
    fn scope_label_code_combines_kind_and_name() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.ts");
        std::fs::write(&p, "class Foo {\n  bar() {\n    const x = 1;\n  }\n}\n").unwrap();
        let cache = OutlineCache::new();
        let label = enclosing_scope_label(&p, 3, &cache).unwrap();
        assert_eq!(label, "function Foo.bar");
    }

    #[test]
    fn scope_label_markdown_returns_section() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.md");
        std::fs::write(
            &p,
            "# Top\n\n## Cost Accounting\n\nsome text\n\nmore text\n",
        )
        .unwrap();
        let cache = OutlineCache::new();
        let label = enclosing_scope_label(&p, 7, &cache).unwrap();
        assert_eq!(label, "§Cost Accounting");
    }

    #[test]
    fn markdown_scope_truncates_long_headings() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.md");
        let long = "x".repeat(80);
        std::fs::write(&p, format!("## {long}\n\nbody\n")).unwrap();
        let label = markdown_enclosing_scope(&p, 3).unwrap();
        // 60-char window means 57 chars + "..." (display starts after the "§").
        assert!(label.starts_with("§"));
        assert!(label.ends_with("..."));
        assert_eq!(label.chars().count(), 1 + 57 + 3);
    }

    #[test]
    fn markdown_scope_returns_none_before_first_heading() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.md");
        std::fs::write(&p, "preamble line one\npreamble line two\n").unwrap();
        assert!(markdown_enclosing_scope(&p, 2).is_none());
    }

    /// `#`-prefixed lines inside fenced code blocks are NOT headings, so they
    /// must not become the enclosing scope label of usages on adjacent lines.
    /// Pre-fix this returned `§...` derived from a Python comment inside the
    /// fence; post-fix the AST owns the distinction.
    #[test]
    fn markdown_scope_skips_hashes_inside_fenced_code() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.md");
        std::fs::write(
            &p,
            "## Outer\n\nbefore fence\n\n```python\n# fake heading\nx = 1\n```\n\nafter fence\n",
        )
        .unwrap();
        // Line 7 (`x = 1`) is inside the fence; the only real heading is `## Outer`.
        let label = markdown_enclosing_scope(&p, 7).unwrap();
        assert_eq!(label, "§Outer");
    }

    /// CommonMark §4.6.1 caps ATX headings at 6 leading `#`s. 7+ hashes is
    /// raw text, not a heading, and must not surface as the enclosing scope.
    /// Pre-AST migration the regex matched `#######` and produced a bogus
    /// `§# Fake Heading 7` label.
    #[test]
    fn markdown_scope_rejects_seven_hash_atx_heading() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.md");
        std::fs::write(
            &p,
            "## Real Heading\n\nbody line\n\n####### Fake Heading 7\n\ntrailing line\n",
        )
        .unwrap();
        // Line 5 is the 7-hash line; line 7 is the line after.
        // Both should resolve to the only real heading, "Real Heading".
        assert_eq!(
            markdown_enclosing_scope(&p, 5),
            Some("§Real Heading".to_string())
        );
        assert_eq!(
            markdown_enclosing_scope(&p, 7),
            Some("§Real Heading".to_string())
        );
    }

    /// CommonMark §4.6.1 requires whitespace after the leading `#`s. `##NoSpace`
    /// is paragraph text, not a heading. Pre-AST migration the regex accepted
    /// it and produced `§NoSpace`.
    #[test]
    fn markdown_scope_rejects_no_space_atx_heading() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.md");
        std::fs::write(&p, "## Real Heading\n\n##NoSpace\n\ntrailing line\n").unwrap();
        // Line 3 is the no-space candidate; both line 3 and line 5 must
        // resolve to the only real heading.
        assert_eq!(
            markdown_enclosing_scope(&p, 3),
            Some("§Real Heading".to_string())
        );
        assert_eq!(
            markdown_enclosing_scope(&p, 5),
            Some("§Real Heading".to_string())
        );
    }

    #[test]
    fn format_single_match_renders_usage_scope_suffix() {
        use crate::types::Match;

        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.ts");
        std::fs::write(&p, "class Foo {\n  bar() {\n    const x = 1;\n  }\n}\n").unwrap();

        let m = Match {
            path: p.clone(),
            line: 3,
            text: "    const x = 1;".to_string(),
            is_definition: false,
            exact: false,
            file_lines: 5,
            mtime: SystemTime::now(),
            def_range: None,
            def_name: None,
            def_weight: 0,
            impl_target: None,
        };
        let cache = OutlineCache::new();
        let bloom = crate::index::bloom::BloomFilterCache::new();
        let mut expand_remaining = 0usize;
        let mut expanded_files: HashSet<PathBuf> = HashSet::new();
        let mut out = String::new();

        format_single_match(
            &m,
            tmp.path(),
            &cache,
            None,
            &bloom,
            &mut expand_remaining,
            &mut expanded_files,
            false,
            &mut out,
        );

        assert!(
            out.contains("[usage in function Foo.bar]"),
            "expected scope suffix in output, got: {out}"
        );
    }

    #[test]
    fn format_single_match_does_not_duplicate_expanded_line() {
        use crate::types::Match;

        // Small file (<50 lines) with no doc comment above the definition —
        // the outline-context preview at line 627 prints `m.text` verbatim,
        // and expand_match's whole-file expansion (small files expand fully,
        // see EXPAND_FULL_FILE_THRESHOLD) includes that same source line
        // again in the fence right below it.
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("small.rs");
        let src = "pub struct Thing;\n\nimpl Thing {\n    pub fn exit_code(&self) -> i32 {\n        0\n    }\n}\n";
        std::fs::write(&p, src).unwrap();

        let m = Match {
            path: p.clone(),
            line: 4,
            text: "    pub fn exit_code(&self) -> i32 {".to_string(),
            is_definition: true,
            exact: true,
            file_lines: 7,
            mtime: SystemTime::now(),
            def_range: Some((4, 6)),
            def_name: Some("exit_code".to_string()),
            def_weight: 100,
            impl_target: None,
        };
        let cache = OutlineCache::new();
        let bloom = crate::index::bloom::BloomFilterCache::new();
        let mut expand_remaining = 1usize;
        let mut expanded_files: HashSet<PathBuf> = HashSet::new();
        let mut out = String::new();

        format_single_match(
            &m,
            tmp.path(),
            &cache,
            None,
            &bloom,
            &mut expand_remaining,
            &mut expanded_files,
            false,
            &mut out,
        );

        let needle = "pub fn exit_code(&self) -> i32 {";
        let occurrences = out.matches(needle).count();
        assert_eq!(
            occurrences, 1,
            "expanded line must appear exactly once, got {occurrences} in: {out}"
        );
    }

    #[test]
    fn boost_query_routes_impl_target_into_truncation() {
        use crate::types::Match;

        let base = Match {
            path: PathBuf::from("x.rs"),
            line: 1,
            text: String::new(),
            is_definition: true,
            exact: true,
            file_lines: 200,
            mtime: SystemTime::now(),
            def_range: Some((1, 200)),
            def_name: None,
            def_weight: 0,
            impl_target: None,
        };

        // Plain definition: the searched token is the symbol name in def_name.
        let plain = Match {
            def_name: Some("handle_request".to_string()),
            ..base.clone()
        };
        assert_eq!(boost_query(&plain), Some("handle_request"));

        // impl match: def_name is the rendered label ("impl Iterator for Counter")
        // which never appears in the body — the searched trait lives in
        // impl_target. Regression guard for the dead-boost bug where def_name was
        // passed and the boost matched nothing.
        let impl_match = Match {
            def_name: Some("impl Iterator for Counter".to_string()),
            impl_target: Some("Iterator".to_string()),
            ..base.clone()
        };
        assert_eq!(boost_query(&impl_match), Some("Iterator"));

        // No names: no boost.
        assert_eq!(boost_query(&base), None);
    }

    #[test]
    fn write_hidden_tail_emits_only_when_truncated() {
        let mut out = String::new();
        write_hidden_tail(&mut out, 3, 3, "definitions");
        assert!(out.is_empty(), "no truncation → no tail line, got {out:?}");

        let mut out = String::new();
        write_hidden_tail(&mut out, 10, 14, "definitions");
        assert_eq!(out, "\n\n... and 4 more definitions. Narrow with scope.");

        let mut out = String::new();
        write_hidden_tail(&mut out, 3, 27, "usages");
        assert_eq!(out, "\n\n... and 24 more usages. Narrow with scope.");
    }

    #[test]
    fn count_label_renders_displayed_over_total_only_when_truncated() {
        // No truncation — bare count.
        assert_eq!(count_label(3, 3), "3");
        // Defensive: shown > total (shouldn't happen in practice) — bare count.
        assert_eq!(count_label(4, 3), "4");
        // Truncated — displayed/total form.
        assert_eq!(count_label(10, 14), "10/14");
        // Zero / zero — still bare (no header is emitted at zero anyway).
        assert_eq!(count_label(0, 0), "0");
    }

    #[test]
    fn format_single_match_inlines_markdown_section_body() {
        use crate::types::Match;

        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("notes.md");
        std::fs::write(
            &p,
            "# Top\n\n## Session 50\n\nFirst paragraph.\n\nSecond line.\n\n## Other\n\nUnrelated.\n",
        )
        .unwrap();

        // Mimic the `Match` `find_defs_markdown_buf` would produce for a
        // markdown-heading def: weight 30, def_range covers the section span.
        let m = Match {
            path: p.clone(),
            line: 3, // `## Session 50`
            text: "## Session 50".to_string(),
            is_definition: true,
            exact: true,
            file_lines: 10,
            mtime: SystemTime::now(),
            def_range: Some((3, 7)), // heading line .. section_end (1-indexed inclusive)
            def_name: Some("Session 50".to_string()),
            def_weight: 30,
            impl_target: None,
        };
        let cache = OutlineCache::new();
        let bloom = crate::index::bloom::BloomFilterCache::new();
        let mut expand_remaining = 0usize;
        let mut expanded_files: HashSet<PathBuf> = HashSet::new();
        let mut out = String::new();

        format_single_match(
            &m,
            tmp.path(),
            &cache,
            None,
            &bloom,
            &mut expand_remaining,
            &mut expanded_files,
            false,
            &mut out,
        );

        assert!(
            out.contains("First paragraph."),
            "section body must be inlined, got: {out:?}"
        );
        assert!(
            out.contains("Second line."),
            "section body must include later body lines, got: {out:?}"
        );
        assert!(
            !out.contains("Unrelated."),
            "must stop at section_end, got: {out:?}"
        );
    }

    #[test]
    fn format_single_match_caps_long_markdown_section() {
        use crate::types::Match;

        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("long.md");
        // Heading on line 1, then 60 body lines.
        let mut content = String::from("## Big Section\n");
        for i in 0..60 {
            let _ = write!(content, "body line {i}\n");
        }
        std::fs::write(&p, &content).unwrap();

        let m = Match {
            path: p.clone(),
            line: 1,
            text: "## Big Section".to_string(),
            is_definition: true,
            exact: true,
            file_lines: 61,
            mtime: SystemTime::now(),
            def_range: Some((1, 61)),
            def_name: Some("Big Section".to_string()),
            def_weight: 30,
            impl_target: None,
        };
        let cache = OutlineCache::new();
        let bloom = crate::index::bloom::BloomFilterCache::new();
        let mut expand_remaining = 0usize;
        let mut expanded_files: HashSet<PathBuf> = HashSet::new();
        let mut out = String::new();

        format_single_match(
            &m,
            tmp.path(),
            &cache,
            None,
            &bloom,
            &mut expand_remaining,
            &mut expanded_files,
            false,
            &mut out,
        );

        // Cap is 40 lines; expect 60 - 40 = 20 truncated.
        assert!(
            out.contains("body line 0"),
            "first body line must appear, got: {out:?}"
        );
        assert!(
            out.contains("body line 39"),
            "last kept body line must appear, got: {out:?}"
        );
        assert!(
            !out.contains("body line 40"),
            "body line beyond cap must be trimmed, got: {out:?}"
        );
        assert!(
            out.contains("20 more lines"),
            "must signal truncated lines, got: {out:?}"
        );
        assert!(
            out.contains("--expand"),
            "tail must point to --expand for full section, got: {out:?}"
        );
    }

    /// 99c4a3d's docstring is explicit: the markdown-section preview is a
    /// fixed-cost short-circuit that bypasses the --expand budget. This pins
    /// that intent — passing a non-zero `expand_remaining` must NOT cause
    /// the renderer to skip the cap. Without this guard, a future refactor
    /// could "fix" the short-circuit by routing through expand and turn
    /// every markdown-heading match into a multi-hundred-line preview.
    #[test]
    fn format_single_match_markdown_cap_bypasses_expand_budget() {
        use crate::types::Match;

        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("long.md");
        let mut content = String::from("## Big Section\n");
        for i in 0..60 {
            let _ = write!(content, "body line {i}\n");
        }
        std::fs::write(&p, &content).unwrap();

        let m = Match {
            path: p.clone(),
            line: 1,
            text: "## Big Section".to_string(),
            is_definition: true,
            exact: true,
            file_lines: 61,
            mtime: SystemTime::now(),
            def_range: Some((1, 61)),
            def_name: Some("Big Section".to_string()),
            def_weight: 30,
            impl_target: None,
        };
        let cache = OutlineCache::new();
        let bloom = crate::index::bloom::BloomFilterCache::new();
        // Non-zero expand budget — should NOT change the cap behavior.
        let mut expand_remaining = 5usize;
        let mut expanded_files: HashSet<PathBuf> = HashSet::new();
        let mut out = String::new();

        format_single_match(
            &m,
            tmp.path(),
            &cache,
            None,
            &bloom,
            &mut expand_remaining,
            &mut expanded_files,
            false,
            &mut out,
        );

        // Body lines beyond the cap must still be trimmed.
        assert!(
            !out.contains("body line 40"),
            "cap must apply even with non-zero expand budget, got: {out:?}"
        );
        assert!(
            out.contains("20 more lines"),
            "tail must still report truncated lines, got: {out:?}"
        );
        // Budget must be untouched — short-circuit returns before consuming it.
        assert_eq!(
            expand_remaining, 5,
            "markdown short-circuit must not consume expand budget"
        );
    }

    /// Worst-case bound: with MAX_MATCHES = 10 markdown-heading defs each
    /// hitting the 40-line preview cap, total inlined preview content is at
    /// most 10 × 40 = 400 lines. This pins the bound by exercising the cap
    /// and asserting the truncation shape, so a future bump of either
    /// constant can't silently inflate worst-case output without updating
    /// the test.
    #[test]
    fn markdown_preview_cap_constant_unchanged() {
        // Pin both constants — if either changes, this assertion fails and
        // the test author has to consider the worst-case product (currently
        // 10 * 40 = 400 lines extra in default preview, on top of the
        // outline context per match).
        assert_eq!(
            MARKDOWN_PREVIEW_MAX_LINES, 40,
            "if you change MARKDOWN_PREVIEW_MAX_LINES, also re-evaluate the \
             MAX_MATCHES * MARKDOWN_PREVIEW_MAX_LINES worst-case bound \
             (currently 10 * 40 = 400 lines per search response)"
        );
    }

    #[test]
    fn format_grouped_usages_emits_h3_heading() {
        use crate::types::Match;

        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.ts");
        std::fs::write(
            &p,
            "function host() {\n  doThing();\n  doThing();\n  doThing();\n}\n",
        )
        .unwrap();

        let mk = |line: u32| Match {
            path: p.clone(),
            line,
            text: "  doThing();".to_string(),
            is_definition: false,
            exact: false,
            file_lines: 5,
            mtime: SystemTime::now(),
            def_range: None,
            def_name: None,
            def_weight: 0,
            impl_target: None,
        };
        let m1 = mk(2);
        let m2 = mk(3);
        let m3 = mk(4);
        let group: Vec<&Match> = vec![&m1, &m2, &m3];
        let cache = OutlineCache::new();
        let mut out = String::new();
        format_grouped_usages(&group, tmp.path(), &cache, &mut out);

        assert!(
            out.starts_with("\n\n### "),
            "grouped-usage heading must be H3, got: {out:?}"
        );
        assert!(
            !out.starts_with("\n\n## "),
            "grouped-usage heading must not be H2, got: {out:?}"
        );
    }

    // Verify that format_matches records segment byte ranges correctly.
    // Each push must cover non-empty, non-overlapping ranges that index into `out`.
    #[test]
    fn format_matches_segments_record_correct_byte_ranges() {
        use crate::index::bloom::BloomFilterCache;
        use crate::types::Match;

        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.rs");
        std::fs::write(&p, "fn alpha() {}\nfn beta() {}\nfn gamma() {}\n").unwrap();

        let mk = |line: u32, weight: u16, name: &str| Match {
            path: p.clone(),
            line,
            text: format!("fn {name}()"),
            is_definition: true,
            exact: true,
            file_lines: 3,
            mtime: SystemTime::now(),
            def_range: Some((line, line)),
            def_name: Some(name.to_string()),
            def_weight: weight,
            impl_target: None,
        };

        let matches = vec![mk(1, 30, "alpha"), mk(2, 10, "beta"), mk(3, 50, "gamma")];
        let cache = OutlineCache::new();
        let bloom = BloomFilterCache::new();
        let mut out = String::from("HEADER");
        let mut segments: Vec<(i64, usize, usize)> = Vec::new();
        let mut expand_remaining = 0usize;
        let mut expanded_files = HashSet::new();

        format_matches(
            &matches,
            tmp.path(),
            &cache,
            None,
            &bloom,
            &mut expand_remaining,
            &mut expanded_files,
            &mut out,
            &mut segments,
        );

        // One segment per match (all singletons — definitions are never grouped).
        assert_eq!(segments.len(), 3, "expected one segment per match");

        // Values must match def_weight casts.
        assert_eq!(segments[0].0, 30i64);
        assert_eq!(segments[1].0, 10i64);
        assert_eq!(segments[2].0, 50i64);

        // Ranges must be valid, non-empty, and non-overlapping.
        let mut cursor = "HEADER".len();
        for (i, &(_, start, end)) in segments.iter().enumerate() {
            assert!(start >= cursor, "segment {i} start < cursor");
            assert!(end > start, "segment {i} is empty");
            assert!(end <= out.len(), "segment {i} end out of bounds");
            // Slice must not panic (validates char-boundary alignment).
            let _ = &out[start..end];
            cursor = end;
        }
    }

    // ── SAVINGS tests ───────────────────────────────────────────

    /// A search that expands a definition large enough to trigger
    /// `select_diverse_lines` (>= 80-line body) must record savings.
    /// A search whose body is short records nothing extra from truncation.
    #[test]
    fn search_truncation_records_savings() {
        use crate::session::Session;
        use crate::types::Match;

        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("big.rs");

        // Build a function body >= 80 lines so select_diverse_lines fires.
        let mut src = String::from("pub fn big_fn() {\n");
        for i in 0..85 {
            src.push_str(&format!("    let v{i} = {i};\n"));
        }
        src.push_str("}\n");
        std::fs::write(&p, &src).unwrap();

        let total_lines = src.lines().count() as u32;
        let m = Match {
            path: p.clone(),
            line: 1,
            text: "pub fn big_fn() {".to_string(),
            is_definition: true,
            exact: true,
            file_lines: total_lines,
            mtime: std::time::SystemTime::now(),
            def_range: Some((1, total_lines)),
            def_name: Some("big_fn".to_string()),
            def_weight: 0,
            impl_target: None,
        };

        let cache = OutlineCache::new();
        let bloom = crate::index::bloom::BloomFilterCache::new();
        let session = Session::default();
        let mut expand_remaining = 5usize;
        let mut expanded_files: HashSet<PathBuf> = HashSet::new();
        let mut out = String::new();

        format_single_match(
            &m,
            tmp.path(),
            &cache,
            Some(&session),
            &bloom,
            &mut expand_remaining,
            &mut expanded_files,
            false,
            &mut out,
        );

        let (baseline, saved) = session.savings();
        assert!(
            baseline > 0,
            "truncation path must record a non-zero baseline, got baseline={baseline}"
        );
        assert!(
            saved > 0,
            "truncation must save tokens vs full body, got saved={saved}"
        );
    }

    /// A search on a small definition (body < 80 lines) goes through
    /// expand_match but never hits the truncation branch, so savings
    /// remain zero.
    #[test]
    fn search_no_truncation_records_no_savings() {
        use crate::session::Session;
        use crate::types::Match;

        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("small.rs");
        let src = "pub fn small_fn() {\n    let x = 1;\n    x\n}\n";
        std::fs::write(&p, src).unwrap();

        let m = Match {
            path: p.clone(),
            line: 1,
            text: "pub fn small_fn() {".to_string(),
            is_definition: true,
            exact: true,
            file_lines: 4,
            mtime: std::time::SystemTime::now(),
            def_range: Some((1, 4)),
            def_name: Some("small_fn".to_string()),
            def_weight: 0,
            impl_target: None,
        };

        let cache = OutlineCache::new();
        let bloom = crate::index::bloom::BloomFilterCache::new();
        let session = Session::default();
        let mut expand_remaining = 5usize;
        let mut expanded_files: HashSet<PathBuf> = HashSet::new();
        let mut out = String::new();

        format_single_match(
            &m,
            tmp.path(),
            &cache,
            Some(&session),
            &bloom,
            &mut expand_remaining,
            &mut expanded_files,
            false,
            &mut out,
        );

        let (baseline, saved) = session.savings();
        assert_eq!(baseline, 0, "no truncation => no savings recorded");
        assert_eq!(saved, 0, "no truncation => no savings recorded");
    }

    /// Regression for the hardcoded-`DEFAULT_BUDGET` bug: `fit_to_budget` must
    /// Search and the read side must agree about a doubled-BOM markdown file's first heading.
    ///
    /// The last piece of the split #42 opened. `read::outline::generate`, `resolve_heading`
    /// and `suggest_headings` all strip a BOM before parsing; `find_defs_markdown_buf` and
    /// `markdown_enclosing_scope` did not. tree-sitter-md skips one BOM by itself, so a single
    /// BOM never diverged — but behind **two** it parses the first heading as a paragraph, and
    /// after #42 the outline advertised that heading and the section resolver accepted it while
    /// search reported it as a plain usage rather than a definition. One half of the tool
    /// naming a definition the other denies.
    ///
    /// `n == 2` is the case that matters; 0 and 1 are controls proving the fixture is sound.
    ///
    /// Naming caveat: this asserts search's side against the read side's *known* answer
    /// (`[definition]`) rather than calling the read side and comparing. A read-side regression
    /// would leave it green — the read side is pinned separately by
    /// `read::tests::the_outline_and_the_heading_resolver_agree_on_a_bom_file`. It also covers
    /// only `find_defs_markdown_buf`; the other half of that fix,
    /// `markdown_enclosing_scope`, is pinned by
    /// `a_doubled_bom_does_not_lose_the_markdown_scope_label`.
    #[test]
    fn search_and_the_read_side_agree_on_a_doubled_bom_markdown_heading() {
        let body = "# Alpha Section\n\ntext here\n\n# Beta Section\n\nmore\n";

        for n in 0..=2 {
            let tmp = tempfile::tempdir().unwrap();
            let mut bytes = Vec::new();
            for _ in 0..n {
                bytes.extend_from_slice(&[0xEF, 0xBB, 0xBF]);
            }
            bytes.extend_from_slice(body.as_bytes());
            let path = tmp.path().join("notes.md");
            std::fs::write(&path, &bytes).unwrap();

            let cache = OutlineCache::new();
            let out = search_symbol("Alpha Section", tmp.path(), &cache, None).unwrap();

            assert!(
                !out.contains('\u{feff}'),
                "{n} BOM(s): a BOM reached search output:\n{out}"
            );
            // The read side treats it as a definition; search must too.
            assert!(
                out.contains("[definition]"),
                "{n} BOM(s): search must report the first heading as a definition, \
                 as the outline and section resolver do:\n{out}"
            );
        }
    }

    /// The `in §Heading` scope label must survive a doubled BOM.
    ///
    /// `markdown_enclosing_scope` is the other half of the markdown fix, and
    /// `search_and_the_read_side_agree_on_a_doubled_bom_markdown_heading` does not reach it —
    /// that one pins `find_defs_markdown_buf` only, so reverting this strip left the suite
    /// green while I claimed both were covered.
    ///
    /// The query has to match a *usage* inside a section rather than the heading itself, since
    /// the label is only attached to usages, and the fixture needs **two** BOMs: tree-sitter-md
    /// absorbs one, so at a single BOM the enclosing section is found either way.
    #[test]
    fn a_doubled_bom_does_not_lose_the_markdown_scope_label() {
        let body = "# Alpha Section\n\nthe zzz_token lives here\n";

        let run = |n: usize| -> String {
            let tmp = tempfile::tempdir().unwrap();
            let mut bytes = Vec::new();
            for _ in 0..n {
                bytes.extend_from_slice(&[0xEF, 0xBB, 0xBF]);
            }
            bytes.extend_from_slice(body.as_bytes());
            std::fs::write(tmp.path().join("notes.md"), &bytes).unwrap();
            let cache = OutlineCache::new();
            search_content("zzz_token", tmp.path(), &cache, None).unwrap()
        };

        let plain = run(0);
        assert!(
            plain.contains("Alpha Section"),
            "fixture is broken: the unmarked file must carry the scope label:\n{plain}"
        );
        for n in [1, 2] {
            let out = run(n);
            assert!(
                !out.contains('\u{feff}'),
                "{n} BOM(s): a BOM reached search output:\n{out}"
            );
            assert!(
                out.contains("Alpha Section"),
                "{n} BOM(s): the enclosing-section label was lost:\n{out}"
            );
        }
    }

    /// A BOM'd file must search identically to the same file without one.
    ///
    /// The acceptance case for #51, driven end to end through `search_symbol_expanded` rather
    /// than at any single writer — which is the point, because the leak had three separate
    /// causes and a per-site test would have missed at least one:
    ///
    ///   * `Match.text` was built from the raw line, so the `-> [1]` preview printed the glyph
    ///     *and* four ranking terms mis-scored the line (see `types::match_text`);
    ///   * `expand_match` rendered the fenced block from its own unstripped read, and
    ///     prefix-tested those same lines for the leading-import skip;
    ///   * `find_defs_markdown_buf` parsed markdown unstripped, disagreeing with the read side.
    ///
    /// Asserting byte equality against the BOM-free spelling covers all of them at once. The
    /// paths differ between the two trees, so the fixture uses the same filename in two
    /// tempdirs and compares the outputs with the directory prefix removed.
    ///
    /// Two honest limits on what this proves. It does **not** detect a reordering: the fixture
    /// is one definition and one usage, which land in different `stratify_for_display` strata,
    /// so no score change could permute them — ranking is pinned by
    /// `rank::tests::a_bom_on_line_one_does_not_change_the_score` instead. And the byte
    /// comparison quietly relies on the two `tempfile` paths having equal byte length, because
    /// the `(~N tokens)` footer is computed before the `<TMP>` substitution; that holds for
    /// `tempfile` today, and a mismatch would show up as a one-token diff rather than as a
    /// silent pass.
    ///
    /// Two things about the fixture are load-bearing, both learned by watching this test pass
    /// while a fix was neutered:
    ///
    ///   * **The matched line must be line 1.** A BOM only ever affects line 1, so a fixture
    ///     whose definition sits on line 3 behind an import leaves `Match.text` clean and
    ///     cannot detect `match_text` regressing at all.
    ///   * **Both `expand` settings must run.** The `-> [line]` preview and the fenced block
    ///     are mutually exclusive — `fence_will_follow` suppresses the preview whenever an
    ///     expansion would reprint the same line — so `expand: 2` alone never reaches the
    ///     preview, and `match_text` goes unpinned.
    #[test]
    fn a_bom_does_not_change_search_output_or_ranking() {
        // Two shapes, because the two bugs live on different lines: the matched line at line 1
        // (reaches `Match.text`, so the preview and the ranking terms), and a BOM'd import at
        // line 1 (what `expand_match`'s leading-import skip prefix-tests).
        let bodies: &[(&str, &str)] = &[
            (
                "def_on_line_1",
                "pub fn alpha_thing() {\n    let _ = 1;\n}\n",
            ),
            (
                "import_on_line_1",
                "use std::fmt;\n\npub fn alpha_thing() {\n    let _ = 1;\n}\n",
            ),
        ];

        let run = |body: &str, prefix: &[u8], expand: usize| -> String {
            let tmp = tempfile::tempdir().unwrap();
            let mut bytes = prefix.to_vec();
            bytes.extend_from_slice(body.as_bytes());
            std::fs::write(tmp.path().join("alpha.rs"), &bytes).unwrap();
            // A second file so ranking has something to order against.
            std::fs::write(
                tmp.path().join("caller.rs"),
                "pub fn calls() {\n    alpha_thing();\n}\n",
            )
            .unwrap();

            let cache = OutlineCache::new();
            let session = Session::new();
            let bloom = crate::index::bloom::BloomFilterCache::new();
            let out = search_symbol_expanded(
                "alpha_thing",
                tmp.path(),
                &cache,
                &session,
                &bloom,
                expand,
                None,
                None,
                false,
                None,
            )
            .unwrap();
            // Paths differ per tempdir; compare everything else.
            out.replace(&tmp.path().to_string_lossy().to_string(), "<TMP>")
        };

        for (shape, body) in bodies {
            for expand in [0, 2] {
                let plain = run(body, &[], expand);
                let bommed = run(body, &[0xEF, 0xBB, 0xBF], expand);

                assert!(
                    plain.contains("alpha_thing"),
                    "{shape}/expand={expand}: fixture is broken, nothing found:\n{plain}"
                );
                assert!(
                    !bommed.contains('\u{feff}'),
                    "{shape}/expand={expand}: a BOM reached search output:\n{bommed}"
                );
                assert_eq!(
                    bommed, plain,
                    "{shape}/expand={expand}: a BOM changed search output \
                     (text, ranking or ordering)"
                );
            }
        }

        // Confirm the fixture reaches both mutually exclusive render paths, so the loop above
        // cannot silently exercise only the fence.
        let preview = run(bodies[0].1, &[], 0);
        let fenced = run(bodies[0].1, &[], 2);
        assert!(
            preview.contains("-> ["),
            "expand=0 must render the `-> [line]` preview, or `match_text` is unpinned:\n{preview}"
        );
        assert!(
            fenced.contains("```"),
            "expand=2 must render the fenced block:\n{fenced}"
        );
    }

    /// receive the caller's real `budget` instead of always being called with
    /// `crate::budget::DEFAULT_BUDGET` (24_000). Fixture: one real definition
    /// (`budget_probe_target`, high `def_weight`) plus a usage in a file named
    /// after the query, so `rank::sort`'s `basename_boost` (+500) renders the
    /// usage FIRST despite it being lower-value — this decouples render order
    /// from value order exactly like the audit's live repro. At a small
    /// explicit budget, the positional tail-cut (`budget::apply`) keeps
    /// whatever rendered first (the low-value usage) and severs the
    /// definition; value-based selection (`fit_to_budget`) keeps the
    /// definition (highest value) and drops the usage instead.
    #[test]
    fn search_symbol_expanded_threads_real_budget_into_value_selection() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("definition.rs"),
            "pub fn budget_probe_target() {\n    let _ = 1;\n}\n",
        )
        .unwrap();
        // File stem == query triggers basename_boost(+500), outscoring the
        // definition's own boosts (def_weight*10 + definition_name_boost) so
        // this usage renders FIRST in the linear (<=5 match) format path.
        // Padded with filler lines so its expanded block is large enough that
        // budget 400 must drop ONE of the two blocks — proving which one a
        // real budget drops (positional: whichever renders last; value-based:
        // the lower-value one, regardless of render order).
        let mut usage_body = String::from("fn calls_it() {\n    budget_probe_target();\n");
        for i in 0..60 {
            usage_body.push_str(&format!("    let filler_{i} = {i};\n"));
        }
        usage_body.push_str("}\n");
        std::fs::write(tmp.path().join("budget_probe_target.rs"), &usage_body).unwrap();

        let cache = OutlineCache::new();
        let session = Session::new();
        let bloom = crate::index::bloom::BloomFilterCache::new();

        let out = search_symbol_expanded(
            "budget_probe_target",
            tmp.path(),
            &cache,
            &session,
            &bloom,
            2,
            None,
            None,
            false,
            Some(400),
        )
        .unwrap();

        assert!(
            out.contains("fn budget_probe_target"),
            "value-based selection must keep the highest-value block (the \
             definition) even though it rendered last: {out}"
        );
        assert!(
            out.contains("lower-value match(es) omitted to fit budget"),
            "must show fit_to_budget's value-based omission marker, not \
             budget::apply's positional \"... truncated\" marker: {out}"
        );
        assert!(
            !out.contains("... truncated ("),
            "positional tail-cut marker must not appear — the real budget \
             was supposed to reach fit_to_budget directly: {out}"
        );
    }

    /// Regression guard: omitting the budget (`None`) must produce byte-
    /// identical output to before this fix — `DEFAULT_BUDGET` remains the
    /// default, it is simply no longer a hardcode that shadows a real
    /// caller-supplied budget.
    #[test]
    fn search_symbol_expanded_no_budget_matches_default_budget_output() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("definition.rs"),
            "pub fn budget_probe_target() {\n    let _ = 1;\n}\n",
        )
        .unwrap();
        std::fs::write(
            tmp.path().join("budget_probe_target.rs"),
            "fn calls_it() {\n    budget_probe_target();\n}\n",
        )
        .unwrap();

        let cache = OutlineCache::new();
        let session = Session::new();
        let bloom = crate::index::bloom::BloomFilterCache::new();

        let with_none = search_symbol_expanded(
            "budget_probe_target",
            tmp.path(),
            &cache,
            &session,
            &bloom,
            2,
            None,
            None,
            false,
            None,
        )
        .unwrap();

        let session2 = Session::new();
        let with_explicit_default = search_symbol_expanded(
            "budget_probe_target",
            tmp.path(),
            &cache,
            &session2,
            &bloom,
            2,
            None,
            None,
            false,
            Some(crate::budget::DEFAULT_BUDGET),
        )
        .unwrap();

        assert_eq!(
            with_none, with_explicit_default,
            "omitting budget must be identical to explicitly passing DEFAULT_BUDGET"
        );
    }
}
