use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::sync::Mutex;
use std::time::SystemTime;

use super::file_metadata;
use crate::lang::treesitter::{
    definition_weight_for, elixir_definition_weight, extract_definition_name,
    extract_elixir_definition_name, extract_impl_trait, extract_impl_type,
    extract_implemented_interfaces, is_definition_node, is_elixir_definition,
};

use crate::error::TilthError;
use crate::lang::detect_file_type;
use crate::lang::outline::{heading_text, outline_language, parse_markdown};
use crate::search::rank;
use crate::types::{FileType, Match, SearchResult};
use grep_regex::RegexMatcher;
use grep_searcher::sinks::UTF8;
use grep_searcher::Searcher;

const MAX_MATCHES: usize = 10;

/// Match-count cap when `--full` is set. Generous but bounded so a `tilth
/// foo --full` on a huge repo can't blow up output.
const FULL_MAX_MATCHES: usize = 100;

// Both walks below used to stop once a shared `AtomicUsize` crossed a raw-match
// threshold (`EARLY_QUIT_THRESHOLD_DEFINITIONS = 50`, `EARLY_QUIT_THRESHOLD_USAGES =
// 30`, and `--full` variants). That made `tilth_search` **non-deterministic**, for
// exactly the reason spelled out above `find_callers_batch` in `callers.rs`: the walk
// is parallel, the counter is read once per file callback, and a single in-flight file
// can add many matches, so how far the walk got depended on thread scheduling.
//
// Six identical consecutive runs, one symbol, 176k-file C++ tree, nothing changed
// between runs: **six distinct renderings**, with the reported usage count moving over
// 30, 30, 30, 39, 28, 30. The definition count sat at exactly 50 every time, which is
// the tell — that was the threshold clamping, reported as if it were a total.
//
// Removing the bound was measured rather than assumed, because unlike `callers` this is
// the most-used path. Measured over MCP `tilth_search` with `expand: 0`, the path an
// agent actually takes. Three reps each, same tree, nothing changed between reps:
//
//   query               bounded                        walk completes
//   moderate symbol     5.31-5.68s / 48 MB, 3 of 3     3.55-4.06s / 56 MB, identical
//   hot symbol          8.39-9.04s / 41 MB, 3 of 3     13.4-13.7s / 89 MB, identical
//
// "3 of 3" is distinct renderings in three runs. The bound was not even buying time in
// the moderate case — it cost ~1.7s there. The reason is that `find_definitions` reads
// every file it visits before the `memmem` needle check, so 50 definitions on a large
// tree is not reached until most of the tree has been read anyway; quitting then saves
// only the tail, and pays for the two walks contending as they wind down. Only a
// genuinely hot symbol pays, at ~5s and ~48 MB.
//
// Both are far inside the 90s request timeout, and both are cheaper than the ~9.5s
// `callers` walk whose bound was removed in the same spirit. So the walks complete and
// the `MAX_MATCHES` / `FULL_MAX_MATCHES` caps below apply afterwards, to a fully
// collected and ranked set — the caps now truncate a stable ranking rather than
// deciding which matches ever got seen.
//
// Completing the walk is necessary but not by itself sufficient, and the rest of the
// argument is load-bearing enough to write down. `rank::sort` is stable but its key
// (`score`, then `path`, then `line`) is *not* a total order — two matches sharing a
// path and line compare equal, which happens for real (two overload declarations on one
// line whose `def_range`s differ, so `dedupe_same_span_definitions` keeps both). A
// stable sort leaves equal elements in input order, and input order here is the order
// the parallel walk appended them. So determinism additionally requires:
//
//   **each file's matches are appended as one contiguous block, in a deterministic
//   within-file order, under a single lock acquisition.**
//
// That holds at every `all.extend(...)` below, so threads can only
// interleave whole files, and ties can only ever be between matches from the same file —
// whose relative order is fixed. `merged.sort_by_key(stratum_for_display)` inherits it.
// Locking per match, or parallelising within a file, would reintroduce the bug without
// touching a line of the walk logic.
//
// `content.rs` no longer appends contiguous per-file blocks — it feeds a bounded heap —
// and does not need to. Every content match has `is_definition: false` and a unique
// `(path, line)`, so `rank::sort`'s key is already a total order on that input and there
// are no ties for arrival order to resolve.
//
// Two costs this shifts onto neighbouring code, both measured:
//
//  * Multi-symbol (comma) queries run one `search` per target, so they multiply the
//    above. A 5-target query on that tree went 22.1s -> 38.2s. It fits in the timeout,
//    but it is the symbol-path twin of the 1+N second-hop walks in `callers.rs` and
//    wants the same fix — one walk over the union of targets, partitioned afterwards.
//  * Peak RSS at this scale is dominated by `BloomFilterCache`, which holds one filter
//    per code file walked and is unbounded. Tracked separately.

/// Display-side stratum: 0 = code def, 1 = doc-heading def, 2 = usage. Used
/// as a stable sort key after `rank::sort` so the `MAX_MATCHES` cap can't drop
/// real code defs in favor of markdown-heading defs of the same query.
fn stratum_for_display(m: &Match) -> u8 {
    if m.is_definition {
        u8::from(m.def_weight < 60)
    } else {
        2
    }
}

/// Number of distinct values `stratum_for_display` can return.
const STRATA: usize = 3;

/// Stable-partition `matches` into the three display strata, preserving the relative
/// order `rank::sort` established within each.
///
/// This was `merged.sort_by_key(stratum_for_display)`. The comparator is cheap, so it was
/// never the time problem `rank::sort` was — but it is a full *stable sort* of
/// `Vec<Match>`, and Rust's stable sort asks for `n/2 * size_of::<Match>()` of scratch:
/// measured at 68 bytes per match, so 163 MB on a 2.4M-match search. Sorting on a key with
/// three possible values does not need that. A counting pass computes each element's exact
/// destination, and `apply_destination_permutation` moves them in place, so the only extra
/// allocation is one `usize` per match.
///
/// Stability falls out of the construction: within a stratum, destinations are handed out
/// in increasing input order.
fn stratify_for_display(matches: &mut [Match]) {
    if matches.len() < 2 {
        return;
    }

    let mut counts = [0usize; STRATA];
    for m in matches.iter() {
        counts[stratum_for_display(m) as usize] += 1;
    }

    // Running start offset for each stratum.
    let mut next = [0usize; STRATA];
    let mut acc = 0;
    for s in 0..STRATA {
        next[s] = acc;
        acc += counts[s];
    }

    let mut dest: Vec<usize> = vec![0; matches.len()];
    for (i, m) in matches.iter().enumerate() {
        let s = stratum_for_display(m) as usize;
        dest[i] = next[s];
        next[s] += 1;
    }

    rank::apply_destination_permutation(matches, &mut dest);
}

/// Symbol search: find definitions via tree-sitter, usages via ripgrep, concurrently.
/// Merge results, deduplicate, definitions first.
///
/// `full` controls the truncation cap: `false` (default) uses the tight
/// default that keeps agent token budgets in check; `true` raises it so
/// interactive `--full` callers see every match instead of "... and N more
/// matches." It does not affect how much of the tree is walked — both walks
/// always complete, so the same query returns the same answer either way.
pub fn search(
    query: &str,
    scope: &Path,
    context: Option<&Path>,
    glob: Option<&str>,
    full: bool,
) -> Result<SearchResult, TilthError> {
    let max_matches = if full { FULL_MAX_MATCHES } else { MAX_MATCHES };

    // Compile regex once, share across both arms
    let word_pattern = format!(r"\b{}\b", regex_syntax::escape(query));
    let matcher = RegexMatcher::new(&word_pattern).map_err(|e| TilthError::InvalidQuery {
        query: query.to_string(),
        reason: e.to_string(),
    })?;

    let (defs, usages) = rayon::join(
        || find_definitions(query, scope, glob),
        || find_usages(query, &matcher, scope, glob),
    );

    let defs = defs?;
    let mut usages = usages?;

    // Deduplicate: remove usage matches that overlap with definition matches.
    //
    // This was a nested scan, quadratic in (definitions × usages). That was free while
    // an early-quit threshold held definitions to ~50; now that both walks complete, a
    // symbol defined in many files across a large tree makes it the dominant cost. A
    // `HashSet` of the definition sites keeps it linear, and `retain` filters in place
    // so the usage set is never held twice.
    let mut merged: Vec<Match> = defs;
    let def_count = merged.len();

    let def_sites: HashSet<(&Path, u32)> =
        merged.iter().map(|d| (d.path.as_path(), d.line)).collect();
    usages.retain(|m| !def_sites.contains(&(m.path.as_path(), m.line)));
    // `def_sites` borrows `merged`; NLL ends that borrow here, before the extend.
    merged.extend(usages);

    let total = merged.len();
    let usage_count = total - def_count;

    rank::sort(&mut merged, query, scope, context);

    // Stratify so the cap can't drop a real code definition in favor of a
    // markdown-heading "definition" of the same query. Stable within each
    // stratum, so the relevance ordering from rank::sort is preserved.
    // Primary defs (def_weight >= 60) come first; the lower stratum holds
    // doc-heading defs (30) alongside definitions that are really variables —
    // JS `lexical_declaration` and C++ data members, both 40 — then usages
    // last. Display-side only: pre-cap totals below and the underlying
    // ranking semantics for `--json` callers are unchanged.
    stratify_for_display(&mut merged);

    // Compute per-subfacet totals on the *pre-cap* set so the renderer can
    // print `displayed/total` headings + per-facet hidden-count lines. Counted
    // by borrow — this used to clone the whole set, which was justified by the
    // early-quit bound holding it to ~80 entries. See `facets::facet_totals`.
    let totals = super::facets::facet_totals(&merged, scope);

    merged.truncate(max_matches);

    Ok(SearchResult {
        query: query.to_string(),
        scope: scope.to_path_buf(),
        matches: merged,
        total_found: total,
        definitions: def_count,
        usages: usage_count,
        facet_totals: totals,
    })
}

/// Find definitions using tree-sitter structural detection.
/// For each file containing the query string, parse with tree-sitter and walk
/// definition nodes to see if any declare the queried symbol.
/// Falls back to keyword heuristic for files without grammars.
///
/// Single-read design: reads each file once, checks for symbol via
/// `memchr::memmem` (SIMD), then reuses the buffer for tree-sitter parsing.
///
/// The walk completes. It is not cut short on a match count — see the note on
/// determinism at the top of this file. Per-file work is still bounded by the
/// size gate and the `memmem` needle check below.
fn find_definitions(
    query: &str,
    scope: &Path,
    glob: Option<&str>,
) -> Result<Vec<Match>, TilthError> {
    let matches: Mutex<Vec<Match>> = Mutex::new(Vec::new());
    let needle = query.as_bytes();

    let walker = super::walker(scope, glob)?;

    walker.run(|| {
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

            // Skip oversized files — avoid tree-sitter parsing multi-MB minified bundles
            let file_size = match std::fs::metadata(path) {
                Ok(meta) => {
                    if meta.len() > 500_000 {
                        return ignore::WalkState::Continue;
                    }
                    meta.len()
                }
                Err(_) => 0,
            };

            // Single read: read file once, use buffer for both check and parse
            let Ok(content) = fs::read_to_string(path) else {
                return ignore::WalkState::Continue;
            };

            // Fast byte check via memchr::memmem (SIMD) — skip files without the symbol
            if memchr::memmem::find(content.as_bytes(), needle).is_none() {
                return ignore::WalkState::Continue;
            }

            // Catch unmarked minified bundles that slipped past the filename check.
            if file_size >= crate::lang::detection::MINIFIED_CHECK_THRESHOLD
                && crate::lang::detection::is_minified_by_content(content.as_bytes())
            {
                return ignore::WalkState::Continue;
            }

            // Get file metadata once per file
            let (file_lines, mtime) = file_metadata(path);

            // Try tree-sitter structural detection
            let file_type = detect_file_type(path);
            let lang = match file_type {
                FileType::Code(l) => Some(l),
                _ => None,
            };

            let ts_language = lang.and_then(outline_language);

            let mut file_defs = if let Some(ref ts_lang) = ts_language {
                find_defs_treesitter(path, query, ts_lang, lang, &content, file_lines, mtime)
            } else {
                Vec::new()
            };

            // Per-file-type fallback dispatch. The semantics of "definition"
            // differ by file kind, so handle them separately:
            //
            // * Code without a tree-sitter grammar: keyword heuristic (looks
            //   for lines starting with `function`/`const`/`class`/etc.).
            // * Markdown / RST: heading-as-definition. A heading whose text
            //   contains the query (`## parseCitations` in a doc) marks that
            //   section AS being about the symbol — that is the documentation
            //   analogue of a code definition. Quoted code blocks inside
            //   docs are NOT treated as definitions; they're usages, because
            //   the keyword heuristic would false-positive on every snippet
            //   that quotes the real source. Heading defs carry a lower
            //   `def_weight` (30) than a primary code definition (60-100) so
            //   the real source still ranks first.
            // * Structured data / tabular / log / other: no fallback.
            //   Mentions are config values, data, or noise — not definitions.
            //   (A future patch could treat top-level config keys matching
            //   the query as soft definitions, but that's ambiguous enough
            //   to skip for now.)
            if file_defs.is_empty() && ts_language.is_none() {
                file_defs = match file_type {
                    FileType::Code(_) => {
                        find_defs_heuristic_buf(path, query, &content, file_lines, mtime)
                    }
                    FileType::Markdown => {
                        find_defs_markdown_buf(path, query, &content, file_lines, mtime)
                    }
                    _ => Vec::new(),
                };
            }

            if !file_defs.is_empty() {
                let mut all = matches
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                // One lock, one contiguous block per file — see the determinism note at the
                // top of this file. Extending per match would break tie-ordering.
                all.extend(file_defs);
            }

            ignore::WalkState::Continue
        })
    });

    Ok(matches
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner))
}

/// Tree-sitter structural definition detection.
/// Accepts pre-read content — no redundant file read.
fn find_defs_treesitter(
    path: &Path,
    query: &str,
    ts_lang: &tree_sitter::Language,
    lang: Option<crate::types::Lang>,
    content: &str,
    file_lines: u32,
    mtime: SystemTime,
) -> Vec<Match> {
    let mut parser = tree_sitter::Parser::new();
    if parser.set_language(ts_lang).is_err() {
        return Vec::new();
    }

    let Some(tree) = parser.parse(content, None) else {
        return Vec::new();
    };

    let lines: Vec<&str> = content.lines().collect();
    let root = tree.root_node();
    let mut defs = Vec::new();

    walk_for_definitions(
        root, query, path, &lines, file_lines, mtime, &mut defs, lang, 0,
    );

    dedupe_same_span_definitions(&mut defs);

    defs
}

/// Collapse definition matches that describe the same definition at the same span,
/// keeping the most specific one.
///
/// Two nodes can name one definition: in C++ a nested class is reachable both as the
/// `field_declaration` wrapping it and as the `class_specifier` inside it, and in
/// TS/JS an exported declaration is reachable both as the `export_statement` and as
/// the `class_declaration` it wraps. Both pairs share a span, so without this the
/// class is reported twice.
///
/// Keeping the *highest `def_weight`* rather than the first is what makes this safe
/// across languages. The walk is depth-first pre-order, so the first of a run is the
/// enclosing node — which for TS/JS is the `export_statement` wrapper, deliberately
/// the lowest definition tier (30) precisely because it is not the interesting node.
/// Keeping it would demote every exported definition below an unrelated local `let`
/// (weight 40) in `rank::sort`, which multiplies `def_weight` by 10.
fn dedupe_same_span_definitions(defs: &mut Vec<Match>) {
    if defs.len() < 2 {
        return;
    }
    // Pre-order emission puts an enclosing node adjacent to the node it wraps, so a
    // single pass over adjacent runs is sufficient.
    let mut out: Vec<Match> = Vec::with_capacity(defs.len());
    for m in defs.drain(..) {
        match out.last_mut() {
            Some(prev) if prev.def_range == m.def_range && prev.def_name == m.def_name => {
                if m.def_weight > prev.def_weight {
                    *prev = m;
                }
            }
            _ => out.push(m),
        }
    }
    *defs = out;
}

/// Recursively walk AST nodes looking for definitions of the queried symbol.
fn walk_for_definitions(
    node: tree_sitter::Node,
    query: &str,
    path: &Path,
    lines: &[&str],
    file_lines: u32,
    mtime: SystemTime,
    defs: &mut Vec<Match>,
    lang: Option<crate::types::Lang>,
    depth: usize,
) {
    if depth > 3 {
        return;
    }

    let kind = node.kind();

    if is_definition_node(node, lang) {
        // Check if this node defines the queried symbol
        if let Some(name) = extract_definition_name(node, lines) {
            if name == query {
                let line_num = node.start_position().row as u32 + 1;
                let line_text = lines
                    .get(node.start_position().row)
                    .unwrap_or(&"")
                    .trim_end();
                defs.push(Match {
                    path: path.to_path_buf(),
                    line: line_num,
                    text: line_text.to_string(),
                    is_definition: true,
                    exact: true,
                    file_lines,
                    mtime,
                    def_range: Some((
                        node.start_position().row as u32 + 1,
                        node.end_position().row as u32 + 1,
                    )),
                    def_name: Some(query.to_string()),
                    def_weight: definition_weight_for(node),
                    impl_target: None,
                });
            }
        }

        // Impl/interface detection: surface `impl Trait for Type` and
        // `class X implements Interface` blocks when searching for the trait/interface.
        if kind == "impl_item" {
            if let Some(trait_name) = extract_impl_trait(node, lines) {
                if trait_name == query {
                    let impl_type =
                        extract_impl_type(node, lines).unwrap_or_else(|| "<unknown>".to_string());
                    let line_num = node.start_position().row as u32 + 1;
                    let line_text = lines
                        .get(node.start_position().row)
                        .unwrap_or(&"")
                        .trim_end();
                    defs.push(Match {
                        path: path.to_path_buf(),
                        line: line_num,
                        text: line_text.to_string(),
                        is_definition: true,
                        exact: true,
                        file_lines,
                        mtime,
                        def_range: Some((
                            node.start_position().row as u32 + 1,
                            node.end_position().row as u32 + 1,
                        )),
                        def_name: Some(format!("impl {query} for {impl_type}")),
                        def_weight: 80,
                        impl_target: Some(query.to_string()),
                    });
                }
            }
        } else if kind == "class_declaration" || kind == "class_definition" {
            let interfaces = extract_implemented_interfaces(node, lines);
            if interfaces.iter().any(|i| i == query) {
                let class_name = extract_definition_name(node, lines)
                    .unwrap_or_else(|| "<anonymous>".to_string());
                let line_num = node.start_position().row as u32 + 1;
                let line_text = lines
                    .get(node.start_position().row)
                    .unwrap_or(&"")
                    .trim_end();
                defs.push(Match {
                    path: path.to_path_buf(),
                    line: line_num,
                    text: line_text.to_string(),
                    is_definition: true,
                    exact: true,
                    file_lines,
                    mtime,
                    def_range: Some((
                        node.start_position().row as u32 + 1,
                        node.end_position().row as u32 + 1,
                    )),
                    def_name: Some(format!("{class_name} implements {query}")),
                    def_weight: 80,
                    impl_target: Some(query.to_string()),
                });
            }
        }
    } else if lang == Some(crate::types::Lang::Elixir) && is_elixir_definition(node, lines) {
        // Elixir: definitions are `call` nodes — check separately
        if let Some(name) = extract_elixir_definition_name(node, lines) {
            if name == query {
                let line_num = node.start_position().row as u32 + 1;
                let line_text = lines
                    .get(node.start_position().row)
                    .unwrap_or(&"")
                    .trim_end();
                defs.push(Match {
                    path: path.to_path_buf(),
                    line: line_num,
                    text: line_text.to_string(),
                    is_definition: true,
                    exact: true,
                    file_lines,
                    mtime,
                    def_range: Some((
                        node.start_position().row as u32 + 1,
                        node.end_position().row as u32 + 1,
                    )),
                    def_name: Some(query.to_string()),
                    def_weight: elixir_definition_weight(node, lines),
                    impl_target: None,
                });
            }
        }
    }

    // Recurse into children (for nested definitions, class bodies, impl blocks, etc.).
    //
    // A C/C++ namespace is a transparent wrapper: it costs two AST levels
    // (`namespace_definition` + `declaration_list`) while adding no nesting an agent
    // cares about, so counting it against the depth budget spends the whole allowance
    // before reaching a class's members. `namespace NS { class Holder { int Count; } }`
    // put `Count` at depth 5 and made it unfindable. Not consuming a level here
    // mirrors how `outline::node_to_entry` already treats namespaces, and keeps C++
    // at parity with the languages whose members sit two levels under the file.
    let child_depth = if is_transparent_wrapper(kind, lang) {
        depth
    } else {
        depth + 1
    };
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk_for_definitions(
            child,
            query,
            path,
            lines,
            file_lines,
            mtime,
            defs,
            lang,
            child_depth,
        );
    }
}

/// True for the C/C++ wrapper nodes that should not consume a depth level.
///
/// Namespaces cost two AST levels (`namespace_definition` + `declaration_list`) while
/// adding no nesting an agent cares about. `template_declaration` is transparent for a
/// second reason as well: it is not a definition kind (see `DEFINITION_KINDS`), so the
/// walk has to reach the declaration it wraps for a member template to resolve at all.
///
/// Scoped to C/C++ so no other grammar's budget changes: `namespace_definition` is
/// also a PHP kind, and `declaration_list` is also C#'s class body — where it is the
/// single body level those languages already spend, not an extra one.
fn is_transparent_wrapper(kind: &str, lang: Option<crate::types::Lang>) -> bool {
    matches!(lang, Some(crate::types::Lang::C | crate::types::Lang::Cpp))
        && matches!(
            kind,
            "namespace_definition"
                | "declaration_list"
                | "linkage_specification"
                | "template_declaration"
        )
}

/// Keyword heuristic fallback for files without tree-sitter grammars.
/// Operates on pre-read buffer — no redundant file read.
fn find_defs_heuristic_buf(
    path: &Path,
    query: &str,
    content: &str,
    file_lines: u32,
    mtime: SystemTime,
) -> Vec<Match> {
    let mut defs = Vec::new();

    for (i, line) in content.lines().enumerate() {
        if line.contains(query) && is_definition_line(line) {
            defs.push(Match {
                path: path.to_path_buf(),
                line: (i + 1) as u32,
                text: line.trim_end().to_string(),
                is_definition: true,
                exact: true,
                file_lines,
                mtime,
                def_range: None,
                def_name: Some(query.to_string()),
                def_weight: 60,
                impl_target: None,
            });
        }
    }

    defs
}

/// Find all usages via ripgrep (word-boundary matching).
/// Collects per-file, locks once per file (not per line).
///
/// The walk completes — see the determinism note at the top of this file.
fn find_usages(
    query: &str,
    matcher: &RegexMatcher,
    scope: &Path,
    glob: Option<&str>,
) -> Result<Vec<Match>, TilthError> {
    let matches: Mutex<Vec<Match>> = Mutex::new(Vec::new());

    let walker = super::walker(scope, glob)?;

    walker.run(|| {
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

            // Skip oversized files
            let file_size = match std::fs::metadata(path) {
                Ok(meta) => {
                    if meta.len() > 500_000 {
                        return ignore::WalkState::Continue;
                    }
                    meta.len()
                }
                Err(_) => 0,
            };

            // Read once and dispatch via `search_slice` so the minified
            // heuristic and the search share a single kernel read.
            let Ok(bytes) = std::fs::read(path) else {
                return ignore::WalkState::Continue;
            };

            // Catch unmarked minified bundles between 100KB and 500KB — they
            // were not skipped by the filename check or the size cap above.
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
                        exact: line.contains(query),
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

    Ok(matches
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner))
}

/// Markdown heading definition detector.
///
/// An ATX heading (`^#{1,6}\s+<text>`) in a `.md`/`.mdx`/`.rst` file is
/// treated as a soft definition of the section about <query> when <query>
/// appears in <text> as a whole identifier (flanked by non-word chars).
/// Setext headings, indented code blocks, and lines inside fenced code
/// blocks are filtered out by the tree-sitter-md parser before we see them.
///
/// Section span (`def_range`) covers the heading line through the last
/// non-blank line before the next same-or-higher-level heading, and is
/// computed from the enclosing `section` node's end position. Sub-headings
/// nest as child sections of the parent and don't terminate the parent.
///
/// Whole-identifier match (not substring-anywhere) prevents false positives
/// like query `func` matching heading `## refactoring guidelines`.
fn find_defs_markdown_buf(
    path: &Path,
    query: &str,
    content: &str,
    file_lines: u32,
    mtime: SystemTime,
) -> Vec<Match> {
    let Some(tree) = parse_markdown(content) else {
        return Vec::new();
    };
    let lines: Vec<&str> = content.lines().collect();
    let mut defs = Vec::new();
    walk_md_sections(
        tree.root_node(),
        &lines,
        query,
        path,
        file_lines,
        mtime,
        &mut defs,
    );
    defs
}

#[allow(clippy::too_many_arguments)]
fn walk_md_sections(
    node: tree_sitter::Node,
    lines: &[&str],
    query: &str,
    path: &Path,
    file_lines: u32,
    mtime: SystemTime,
    defs: &mut Vec<Match>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "section" => {
                emit_md_section_match(child, lines, query, path, file_lines, mtime, defs);
                walk_md_sections(child, lines, query, path, file_lines, mtime, defs);
            }
            // The parser owns these — no headings hide inside.
            "fenced_code_block" | "indented_code_block" | "html_block" => {}
            _ => walk_md_sections(child, lines, query, path, file_lines, mtime, defs),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_md_section_match(
    section: tree_sitter::Node,
    lines: &[&str],
    query: &str,
    path: &Path,
    file_lines: u32,
    mtime: SystemTime,
    defs: &mut Vec<Match>,
) {
    let mut cursor = section.walk();
    let Some(heading) = section
        .children(&mut cursor)
        .find(|c| c.kind() == "atx_heading")
    else {
        return;
    };
    let text = heading_text(heading, lines);
    if !contains_identifier(&text, query) {
        return;
    }
    let heading_line = (heading.start_position().row + 1) as u32;
    let raw_end = md_section_end_line(section);
    let section_end = trim_trailing_blank_lines(lines, heading_line, raw_end);
    let line_text = lines
        .get(heading.start_position().row)
        .copied()
        .unwrap_or("");
    defs.push(Match {
        path: path.to_path_buf(),
        line: heading_line,
        text: line_text.trim_end().to_string(),
        is_definition: true,
        exact: true,
        file_lines,
        mtime,
        // Populating def_range lets the renderer expand to the section
        // body — the markdown analogue of a code definition's body.
        def_range: Some((heading_line, section_end)),
        def_name: Some(query.to_string()),
        // Soft definition — code definitions are 60-80, usages 0. Sits
        // between them so docs headings outrank passing mentions but
        // never outrank the real source.
        def_weight: 30,
        impl_target: None,
    });
}

/// 1-indexed inclusive last line of a tree-sitter section node.
fn md_section_end_line(section: tree_sitter::Node) -> u32 {
    let end = section.end_position();
    if end.column == 0 {
        end.row as u32
    } else {
        (end.row + 1) as u32
    }
}

fn trim_trailing_blank_lines(lines: &[&str], start: u32, end: u32) -> u32 {
    let mut e = end;
    while e > start
        && lines
            .get((e - 1) as usize)
            .is_some_and(|l| l.trim().is_empty())
    {
        e -= 1;
    }
    e
}

/// True if `query` appears in `text` as a whole identifier — flanked by
/// non-word characters (anything outside `[A-Za-z0-9_]`) or string ends.
fn contains_identifier(text: &str, query: &str) -> bool {
    if query.is_empty() {
        return false;
    }
    text.match_indices(query).any(|(abs, _)| {
        let bytes = text.as_bytes();
        let before_ok = abs == 0 || !is_word_byte(bytes[abs - 1]);
        let end_pos = abs + query.len();
        let after_ok = end_pos == bytes.len() || !is_word_byte(bytes[end_pos]);
        before_ok && after_ok
    })
}

fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Keyword heuristic fallback — only used when tree-sitter grammar unavailable.
fn is_definition_line(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with("fn ")
        || trimmed.starts_with("pub fn ")
        || trimmed.starts_with("pub(crate) fn ")
        || trimmed.starts_with("async fn ")
        || trimmed.starts_with("pub async fn ")
        || trimmed.starts_with("function ")
        || trimmed.starts_with("export function ")
        || trimmed.starts_with("export default function ")
        || trimmed.starts_with("export async function ")
        || trimmed.starts_with("async function ")
        || trimmed.starts_with("const ")
        || trimmed.starts_with("export const ")
        || trimmed.starts_with("let ")
        || trimmed.starts_with("export let ")
        || trimmed.starts_with("var ")
        || trimmed.starts_with("export var ")
        || trimmed.starts_with("class ")
        || trimmed.starts_with("export class ")
        || trimmed.starts_with("interface ")
        || trimmed.starts_with("export interface ")
        || trimmed.starts_with("type ")
        || trimmed.starts_with("export type ")
        || trimmed.starts_with("struct ")
        || trimmed.starts_with("pub struct ")
        || trimmed.starts_with("enum ")
        || trimmed.starts_with("pub enum ")
        || trimmed.starts_with("trait ")
        || trimmed.starts_with("pub trait ")
        || trimmed.starts_with("impl ")
        || trimmed.starts_with("def ")
        || trimmed.starts_with("async def ")
        || trimmed.starts_with("func ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::SystemTime;

    /// `stratify_for_display` replaced `sort_by_key(stratum_for_display)` to avoid the
    /// stable sort's `n/2 * size_of::<Match>()` scratch buffer on an unbounded match set.
    /// It must be indistinguishable from what it replaced, including stability *within*
    /// each stratum — that ordering is the one `rank::sort` just established, and losing it
    /// would silently change which matches survive the display cap.
    #[test]
    fn stratify_for_display_matches_a_stable_sort_by_key() {
        let strata_source = |i: usize| -> (bool, u16) {
            match i % 3 {
                0 => (true, 80), // stratum 0: primary code definition
                1 => (true, 30), // stratum 1: doc-heading / variable definition
                _ => (false, 0), // stratum 2: usage
            }
        };

        let build = || -> Vec<Match> {
            (0..97)
                .map(|i| {
                    let (is_definition, def_weight) = strata_source(i);
                    Match {
                        // Distinct path and line per element so the assertion can identify
                        // each one and detect reordering within a stratum.
                        path: PathBuf::from(format!("/repo/src/f{i}.rs")),
                        line: u32::try_from(i).unwrap() + 1,
                        text: format!("line {i}"),
                        is_definition,
                        exact: false,
                        file_lines: 10,
                        mtime: SystemTime::UNIX_EPOCH,
                        def_range: None,
                        def_name: None,
                        def_weight,
                        impl_target: None,
                    }
                })
                .collect()
        };

        let mut actual = build();
        stratify_for_display(&mut actual);

        let mut expected = build();
        expected.sort_by_key(stratum_for_display);

        let key = |v: &[Match]| {
            v.iter()
                .map(|m| (stratum_for_display(m), m.line))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            key(&actual),
            key(&expected),
            "counting-sort stratification disagreed with the stable sort_by_key it replaced"
        );

        // Independently: strata must be non-decreasing, and lines strictly increasing
        // within each stratum (the input order, since every element has a distinct line).
        let mut last = (0u8, 0u32);
        for m in &actual {
            let cur = (stratum_for_display(m), m.line);
            assert!(
                cur.0 > last.0 || (cur.0 == last.0 && cur.1 > last.1),
                "stratification is not stable at {cur:?} after {last:?}"
            );
            last = cur;
        }
        // All three strata must actually be populated, or the above proves little.
        for s in 0..3u8 {
            assert!(
                actual.iter().any(|m| stratum_for_display(m) == s),
                "stratum {s} unpopulated — fixture no longer covers the partition"
            );
        }
    }

    #[test]
    fn rust_definitions_detected() {
        let code = r#"pub fn hello(name: &str) -> String {
    format!("Hello, {}", name)
}

pub struct Foo {
    bar: i32,
}

pub(crate) fn dispatch_tool(tool: &str) -> Result<String, String> {
    match tool {
        "read" => Ok("read".to_string()),
        _ => Err("unknown".to_string()),
    }
}
"#;
        let ts_lang = crate::lang::outline::outline_language(crate::types::Lang::Rust).unwrap();

        let defs = find_defs_treesitter(
            std::path::Path::new("test.rs"),
            "hello",
            &ts_lang,
            Some(crate::types::Lang::Rust),
            code,
            15,
            SystemTime::now(),
        );
        assert!(!defs.is_empty(), "should find 'hello' definition");
        assert!(defs[0].is_definition);
        assert!(defs[0].def_range.is_some());

        let defs = find_defs_treesitter(
            std::path::Path::new("test.rs"),
            "Foo",
            &ts_lang,
            Some(crate::types::Lang::Rust),
            code,
            15,
            SystemTime::now(),
        );
        assert!(!defs.is_empty(), "should find 'Foo' definition");

        let defs = find_defs_treesitter(
            std::path::Path::new("test.rs"),
            "dispatch_tool",
            &ts_lang,
            Some(crate::types::Lang::Rust),
            code,
            15,
            SystemTime::now(),
        );
        assert!(!defs.is_empty(), "should find 'dispatch_tool' definition");
    }

    #[test]
    fn typescript_export_const_detected_as_definition() {
        let code = r#"export const UNTAGGED_REQUESTS_SQL = `SELECT foo FROM bar`;

export const anotherConst = 42;

const unexported = "hello";
"#;
        let ts_lang =
            crate::lang::outline::outline_language(crate::types::Lang::TypeScript).unwrap();
        let lines = code.lines().count() as u32;

        let defs = find_defs_treesitter(
            std::path::Path::new("test.ts"),
            "UNTAGGED_REQUESTS_SQL",
            &ts_lang,
            Some(crate::types::Lang::TypeScript),
            code,
            lines,
            SystemTime::now(),
        );
        assert!(
            !defs.is_empty(),
            "should find 'UNTAGGED_REQUESTS_SQL' definition"
        );
        assert!(defs[0].is_definition);
        assert!(defs[0].def_range.is_some());

        // Non-exported const also detected
        let defs = find_defs_treesitter(
            std::path::Path::new("test.ts"),
            "unexported",
            &ts_lang,
            Some(crate::types::Lang::TypeScript),
            code,
            lines,
            SystemTime::now(),
        );
        assert!(!defs.is_empty(), "should find 'unexported' definition");
        assert!(defs[0].is_definition);
    }

    /// Helper: search for an Elixir definition by name in a code snippet.
    fn elixir_find(code: &str, name: &str) -> Vec<Match> {
        let ts_lang = crate::lang::outline::outline_language(crate::types::Lang::Elixir).unwrap();
        let lines = code.lines().count() as u32;
        find_defs_treesitter(
            std::path::Path::new("test.ex"),
            name,
            &ts_lang,
            Some(crate::types::Lang::Elixir),
            code,
            lines,
            SystemTime::now(),
        )
    }

    #[test]
    fn elixir_definitions_detected() {
        let code = r#"defmodule MyApp.Greeter do
  @type t :: %{name: String.t()}

  def hello(name) do
    "Hello, #{name}!"
  end

  defp private_helper(x), do: x + 1

  defmacro my_macro(expr) do
    quote do: unquote(expr)
  end
end
"#;
        // Dotted module name
        let defs = elixir_find(code, "MyApp.Greeter");
        assert!(!defs.is_empty(), "should find 'MyApp.Greeter' module def");
        assert!(defs[0].is_definition);

        // Public function (block form with parens)
        assert!(
            !elixir_find(code, "hello").is_empty(),
            "should find 'hello'"
        );

        // Private function (keyword form: `, do:`)
        assert!(
            !elixir_find(code, "private_helper").is_empty(),
            "should find 'private_helper'"
        );

        // Macro
        assert!(
            !elixir_find(code, "my_macro").is_empty(),
            "should find 'my_macro'"
        );
    }

    #[test]
    fn elixir_guard_clause_definitions() {
        let code = r#"defmodule Guards do
  def safe_div(a, b) when b != 0 do
    a / b
  end

  defp checked(x) when is_integer(x), do: x

  defguard is_positive(x) when x > 0
end
"#;
        // Guard clause with `when` — block form
        assert!(
            !elixir_find(code, "safe_div").is_empty(),
            "should find 'safe_div' with guard clause"
        );

        // Guard clause with `when` — keyword form
        assert!(
            !elixir_find(code, "checked").is_empty(),
            "should find 'checked' with guard clause"
        );

        // defguard
        assert!(
            !elixir_find(code, "is_positive").is_empty(),
            "should find 'is_positive' defguard"
        );
    }

    #[test]
    fn elixir_multi_clause_and_no_arg() {
        let code = r#"defmodule Dispatch do
  def handle(:ok), do: :success
  def handle(:error), do: :failure

  def version, do: "1.0"
end
"#;
        // Multi-clause: both clauses should be found
        let defs = elixir_find(code, "handle");
        assert!(
            defs.len() >= 2,
            "should find both 'handle' clauses, got {}: {defs:?}",
            defs.len()
        );

        // No-arg function (bare identifier, no parens)
        assert!(
            !elixir_find(code, "version").is_empty(),
            "should find no-arg 'version'"
        );
    }

    #[test]
    fn elixir_protocol_impl_exception() {
        let code = r#"defprotocol Printable do
  @callback format(t) :: String.t()
  def to_string(data)
end

defimpl Printable, for: User do
  def to_string(user), do: user.name
end

defmodule MyError do
  defexception [:message, :code]
end
"#;
        // Protocol + defimpl: both indexed under the protocol name "Printable"
        let defs = elixir_find(code, "Printable");
        assert!(
            defs.len() >= 2,
            "should find both defprotocol and defimpl for 'Printable', got {}",
            defs.len()
        );

        // defexception
        assert!(
            !elixir_find(code, "defexception").is_empty(),
            "should find 'defexception'"
        );

        // Module containing exception
        assert!(
            !elixir_find(code, "MyError").is_empty(),
            "should find 'MyError' module"
        );
    }

    #[test]
    fn elixir_delegate_and_nested_modules() {
        let code = r#"defmodule Outer do
  defdelegate count(list), to: Enum

  defmodule Inner do
    def nested_func, do: :ok
  end
end
"#;
        // defdelegate
        assert!(
            !elixir_find(code, "count").is_empty(),
            "should find 'count' defdelegate"
        );

        // Nested module
        assert!(
            !elixir_find(code, "Inner").is_empty(),
            "should find nested 'Inner' module"
        );
    }

    fn md_find(content: &str, query: &str) -> Vec<Match> {
        let lines = content.lines().count() as u32;
        find_defs_markdown_buf(
            std::path::Path::new("test.md"),
            query,
            content,
            lines,
            SystemTime::now(),
        )
    }

    #[test]
    fn markdown_heading_named_for_query_matches() {
        let content = "# Intro\n\n## parseCitations\n\nProse.\n";
        let defs = md_find(content, "parseCitations");
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].line, 3);
        assert!(defs[0].is_definition);
        assert_eq!(defs[0].def_weight, 30);
    }

    #[test]
    fn markdown_heading_levels_one_through_six() {
        for level in 1..=6 {
            let hashes = "#".repeat(level);
            let content = format!("{hashes} parseCitations\n");
            assert_eq!(md_find(&content, "parseCitations").len(), 1, "h{level}");
        }
        // h7 is not a heading
        assert!(md_find("####### parseCitations\n", "parseCitations").is_empty());
    }

    #[test]
    fn markdown_heading_without_query_does_not_match() {
        let content = "## Other section\n\n## Another heading\n";
        assert!(md_find(content, "parseCitations").is_empty());
    }

    #[test]
    fn markdown_substring_inside_word_does_not_match() {
        // query "func" must not match "function" — that's the maintainer's
        // word-boundary concern. Same for "factor" inside "refactoring".
        assert!(md_find("## function pointers\n", "func").is_empty());
        assert!(md_find("## refactoring guidelines\n", "factor").is_empty());
        assert!(md_find("## getCitationsBatch\n", "Citations").is_empty());
    }

    #[test]
    fn markdown_whole_word_in_phrase_matches() {
        // Whole-word match anywhere in the heading text is a definition —
        // a heading like `## How parseCitations works` IS naming the symbol.
        let defs = md_find("## How parseCitations works\n", "parseCitations");
        assert_eq!(defs.len(), 1);
    }

    #[test]
    fn markdown_query_with_hyphen_matches() {
        // Tracking-doc identifiers like `GUM-1732` must match. The hyphen
        // is part of the query; word-boundary check applies only at the ends.
        let defs = md_find("## GUM-1732: Cost attribution\n", "GUM-1732");
        assert_eq!(defs.len(), 1);
    }

    #[test]
    fn markdown_code_block_lines_do_not_match() {
        // Fenced code block — line is not an ATX heading, even though
        // the text contains `function parseCitations`.
        let content = "## Real heading\n\n```ts\nfunction parseCitations() {}\n```\n";
        let defs = md_find(content, "parseCitations");
        assert!(defs.is_empty(), "fenced-code mention is not a definition");

        // Indented code block (4+ space indent) — a `## ...` line indented
        // 4 spaces is a code block per CommonMark, not a heading.
        let content = "Intro.\n\n    ## parseCitations\n";
        assert!(
            md_find(content, "parseCitations").is_empty(),
            "4-space-indented `## foo` is a code block, not a heading"
        );
    }

    #[test]
    fn markdown_heading_with_up_to_three_space_indent_matches() {
        // 0-3 space indents are valid ATX headings per CommonMark.
        for indent in 0..=3 {
            let content = format!("{}## parseCitations\n", " ".repeat(indent));
            assert_eq!(
                md_find(&content, "parseCitations").len(),
                1,
                "indent {indent} should be a heading"
            );
        }
    }

    #[test]
    fn markdown_heading_with_trailing_hashes_matches() {
        // ATX allows optional trailing `#`s — strip them before matching.
        assert_eq!(md_find("## parseCitations ##\n", "parseCitations").len(), 1);
        assert_eq!(
            md_find("### parseCitations ###\n", "parseCitations").len(),
            1
        );
    }

    #[test]
    fn markdown_hashes_without_space_are_not_headings() {
        // `##foo` (no space after `#`s) is not a heading.
        assert!(md_find("##parseCitations\n", "parseCitations").is_empty());
    }

    #[test]
    fn markdown_section_span_runs_to_next_same_level_heading() {
        // `## parseCitations` body ends at the next `## ...` (same level).
        // The blank line on line 4 (between body and next heading) is
        // trimmed, so the span ends at line 3.
        let content = "\
## parseCitations

Body line.

## Other section

Unrelated.
";
        let defs = md_find(content, "parseCitations");
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].line, 1);
        assert_eq!(defs[0].def_range, Some((1, 3)));
    }

    #[test]
    fn markdown_section_span_runs_to_higher_level_heading() {
        // A `## ...` ends a sub-section under `### parseCitations` because
        // the outer heading is higher level (smaller hash count). The blank
        // line preceding `## Outer two` is trimmed.
        let content = "\
## Outer

### parseCitations

Body.

## Outer two
";
        let defs = md_find(content, "parseCitations");
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].line, 3);
        assert_eq!(defs[0].def_range, Some((3, 5)));
    }

    #[test]
    fn markdown_section_span_skips_deeper_subheadings() {
        // A `### ...` does NOT end the enclosing `## parseCitations`
        // section — only same-or-higher-level headings do.
        let content = "\
## parseCitations

Lead-in.

### Detail

Subprose.

## Next
";
        let defs = md_find(content, "parseCitations");
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].line, 1);
        assert_eq!(defs[0].def_range, Some((1, 7)));
    }

    #[test]
    fn markdown_section_span_runs_to_eof_when_no_following_heading() {
        let content = "\
## parseCitations

Body to end.
";
        let defs = md_find(content, "parseCitations");
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].line, 1);
        // Three content lines; trailing newline does not produce a 4th.
        assert_eq!(defs[0].def_range, Some((1, 3)));
    }

    #[test]
    fn markdown_section_span_handles_heading_with_no_body() {
        // Adjacent headings: span is just the heading line itself.
        let content = "\
## parseCitations
## Other
";
        let defs = md_find(content, "parseCitations");
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].line, 1);
        assert_eq!(defs[0].def_range, Some((1, 1)));
    }

    #[test]
    fn stratify_for_display_keeps_code_defs_above_doc_defs() {
        // When the cap drops matches, real code defs must keep their slots
        // and doc-heading defs slide below them. Rank order within each
        // stratum is preserved by the stable sort.
        let mk = |line: u32, weight: u16, is_definition: bool| Match {
            path: PathBuf::from("test.rs"),
            line,
            text: String::new(),
            is_definition,
            exact: false,
            file_lines: 100,
            mtime: SystemTime::now(),
            def_range: None,
            def_name: None,
            def_weight: weight,
            impl_target: None,
        };

        // Pre-cap order (after rank::sort): doc def, code def, doc def, code def, usage.
        let mut matches = vec![
            mk(1, 30, true), // doc def — high relevance
            mk(2, 70, true), // code def
            mk(3, 30, true), // doc def
            mk(4, 70, true), // code def
            mk(5, 0, false), // usage
        ];
        matches.sort_by_key(stratum_for_display);

        // Code defs first (stable order: line 2 before line 4),
        // then doc defs (line 1 before line 3), then the usage.
        let lines: Vec<u32> = matches.iter().map(|m| m.line).collect();
        assert_eq!(lines, vec![2, 4, 1, 3, 5]);

        // Truncate-to-2 should keep both code defs, drop both doc defs.
        matches.truncate(2);
        assert!(
            matches.iter().all(|m| m.def_weight >= 60),
            "displayed slice after cap must be all code defs, got {:?}",
            matches.iter().map(|m| m.def_weight).collect::<Vec<_>>()
        );
    }

    /// Helper: search for a C++ definition by name in a `.h` snippet.
    fn cpp_find(code: &str, name: &str) -> Vec<Match> {
        let ts_lang = crate::lang::outline::outline_language(crate::types::Lang::Cpp).unwrap();
        find_defs_treesitter(
            std::path::Path::new("Probe.h"),
            name,
            &ts_lang,
            Some(crate::types::Lang::Cpp),
            code,
            code.lines().count() as u32,
            SystemTime::now(),
        )
    }

    #[test]
    fn cpp_type_definitions_detected() {
        let code = "\
class PlainThing { public: void DoPlainWork(); };
class BaseThing {};
class FinalWithBase final : public BaseThing {};
struct PlainStruct { int A; };
enum class ScopedEnum : uint8_t { SA, SB };
template <typename T> class TemplateThing { public: void Work(); };
typedef int MyTypedef;
using MyAlias = float;
";
        for name in [
            "PlainThing",
            "BaseThing",
            "FinalWithBase",
            "PlainStruct",
            "ScopedEnum",
            "TemplateThing",
            "MyTypedef",
            "MyAlias",
        ] {
            let defs = cpp_find(code, name);
            assert!(!defs.is_empty(), "should find C++ definition of {name}");
            assert!(defs[0].is_definition, "{name} should be a definition");
            assert!(defs[0].def_range.is_some(), "{name} needs a def_range");
        }
    }

    #[test]
    fn cpp_class_definition_is_reported_once() {
        // A nested class is reachable both as the `field_declaration` wrapping it and
        // as the `class_specifier` inside it, both starting on the same line. Only one
        // match may survive, or every C++ class would be reported twice.
        let code = "class Outer { public: class Inner { void Deep(); }; };\n";
        let inner = cpp_find(code, "Inner");
        assert_eq!(
            inner.len(),
            1,
            "nested class must be reported once, got {inner:?}"
        );
        let outer = cpp_find(code, "Outer");
        assert_eq!(outer.len(), 1, "class must be reported once, got {outer:?}");
    }

    /// `dedupe_same_span_definitions` must keep the *highest-weight* node of a
    /// same-span run, not the first. The walk is pre-order, so the first is the
    /// enclosing node — for TS/JS that is the `export_statement` wrapper, weight 30,
    /// the lowest definition tier. Keeping it demoted every exported definition below
    /// an unrelated local `let` (weight 40), because `rank::sort` multiplies
    /// `def_weight` by 10. This is the run the dedup actually fires on; the C++ nested
    /// class it was written for is depth-limited out of reach.
    #[test]
    fn exported_ts_definition_survives_dedup_with_its_real_weight() {
        let code = "export class Widget {}\nexport function handle() {}\n";
        let ts_lang = crate::lang::outline::outline_language(crate::types::Lang::TypeScript)
            .expect("ts grammar");
        for (name, want_weight) in [("Widget", 100u16), ("handle", 100)] {
            let defs = find_defs_treesitter(
                std::path::Path::new("thing.ts"),
                name,
                &ts_lang,
                Some(crate::types::Lang::TypeScript),
                code,
                2,
                SystemTime::now(),
            );
            assert_eq!(
                defs.len(),
                1,
                "{name} should be reported once, got {defs:?}"
            );
            assert_eq!(
                defs[0].def_weight, want_weight,
                "{name} must keep the inner declaration's weight, not export_statement's 30"
            );
        }
    }

    #[test]
    fn exported_definition_outranks_unrelated_local_binding() {
        // End-to-end consequence of the above: the real definition must still lead.
        let dir = tempfile::tempdir().expect("tempdir");
        let scope = dir.path();
        std::fs::create_dir_all(scope.join("deep")).expect("mkdir");
        std::fs::write(
            scope.join("deep").join("thing.ts"),
            "export class Widget {}\n",
        )
        .expect("write");
        std::fs::write(scope.join("local.ts"), "let Widget = 1;\n").expect("write");

        let result = search("Widget", scope, None, None, false).expect("search");
        let top = result.matches.first().expect("a match");
        assert!(
            top.path.ends_with("thing.ts"),
            "`export class Widget` must outrank a local `let Widget`, got {:?}",
            result
                .matches
                .iter()
                .map(|m| (m.path.file_name(), m.def_weight))
                .collect::<Vec<_>>()
        );
    }

    /// A C++ namespace costs two AST levels while adding no nesting an agent cares
    /// about, so counting it against the walk's depth budget spent the whole allowance
    /// before reaching a class's members — `namespace NS { class Holder { int Count; } }`
    /// made `Count` unfindable, which undercuts resolving C++ *members* at all.
    #[test]
    fn cpp_members_inside_namespaces_are_findable() {
        let code = "namespace N0 {\n\
                    namespace N1 {\n\
                    class Target { public: void Method(); int Count; };\n\
                    }\n\
                    }\n";
        for name in ["Target", "Method", "Count"] {
            let defs = cpp_find(code, name);
            assert_eq!(
                defs.len(),
                1,
                "{name} should be found exactly once inside nested namespaces, got {defs:?}"
            );
        }
        // C++17 nested-namespace form resolves the same way.
        let joined = "namespace A::B::C { class Target { public: void Method(); }; }\n";
        assert_eq!(cpp_find(joined, "Target").len(), 1);
        assert_eq!(cpp_find(joined, "Method").len(), 1);
    }

    /// A template whose `template <…>` clause sits on its own line — the normal spelling
    /// in real C++ — was reported twice, once for the `template_declaration` wrapper and
    /// once for the declaration it wraps. Their spans differ, so
    /// `dedupe_same_span_definitions` could not collapse them; only the single-line
    /// spelling happened to coincide and dedupe, which is why the original tests missed
    /// it. Fixed by making the wrapper transparent rather than a definition.
    #[test]
    fn cpp_multi_line_template_is_reported_once() {
        let cases: &[(&str, &str)] = &[
            (
                "template <typename T>\nclass Vector { public: void Add(T V); };\n",
                "Vector",
            ),
            (
                "template <typename T> class Vector { public: void Add(T V); };\n",
                "Vector",
            ),
            ("template <typename T>\nstruct Holder { T V; };\n", "Holder"),
            ("template <typename T>\nvoid Swap(T& A, T& B) {}\n", "Swap"),
        ];
        for (src, name) in cases {
            let defs = cpp_find(src, name);
            assert_eq!(
                defs.len(),
                1,
                "{name} should be reported once for {src:?}, got {defs:?}"
            );
        }

        // An explicit specialization is a *different* entity from the in-class member it
        // specializes, so two definitions here are correct — what must not happen is two
        // reports of the same one. Assert distinct spans rather than a count of 1.
        let spec = "template <typename T> class Foo { public: static int v; };\n\
                    template<>\n\
                    int Foo<int>::v = 0;\n";
        let defs = cpp_find(spec, "v");
        assert_eq!(defs.len(), 2, "expected the member and its specialization");
        let spans: std::collections::HashSet<_> = defs.iter().map(|m| m.def_range).collect();
        assert_eq!(
            spans.len(),
            2,
            "the two definitions must have distinct spans"
        );
    }

    /// A member template must still resolve. The wrapper being transparent is what makes
    /// this work: it costs no depth level, so the walk reaches the declaration inside a
    /// class body rather than exhausting its budget on the wrapper.
    #[test]
    fn cpp_member_template_resolves() {
        let src = "class Holder {\npublic:\ntemplate <typename T>\nvoid Apply(T V);\n};\n";
        assert_eq!(cpp_find(src, "Apply").len(), 1);
    }

    /// Registering C++ member declarations as definitions made member variables compete
    /// with real type definitions: searching a name that is both a data member somewhere
    /// and a class elsewhere could lead with the member. The class must win.
    #[test]
    fn cpp_data_member_ranks_below_a_real_type_of_the_same_name() {
        let dir = tempfile::tempdir().expect("tempdir");
        let scope = dir.path();
        // The member lives in the file whose *basename* matches the query, so
        // `rank::basename_boost` pushes it up. That boost is what made the old weight of
        // 70 win: with it, a fixture where the member sits in an unrelated file cannot
        // distinguish 70 from 40 — the class led either way. Here the boost and the
        // weight gap pull against each other, so the ordering actually depends on the
        // member being demoted to the data tier.
        std::fs::write(
            scope.join("AbilityLevel.h"),
            "class HeroComponent\n{\nprivate:\n\tint AbilityLevel;\n};\n",
        )
        .expect("write");
        std::fs::write(
            scope.join("GameTypes.h"),
            "class AbilityLevel { public: int Value; };\n",
        )
        .expect("write");

        let result = search("AbilityLevel", scope, None, None, false).expect("search");
        let top = result.matches.first().expect("a match");
        assert!(
            top.path.ends_with("GameTypes.h"),
            "the class must outrank the same-named data member even when the member's \
             file wins the basename boost, got {:?}",
            result
                .matches
                .iter()
                .map(|m| (m.path.file_name(), m.def_weight))
                .collect::<Vec<_>>()
        );
        // The member is still findable — just ranked below.
        assert!(
            result
                .matches
                .iter()
                .any(|m| m.path.ends_with("AbilityLevel.h") && m.is_definition),
            "the data member should still be reported as a definition"
        );
    }

    #[test]
    fn cpp_forward_declaration_is_not_a_definition() {
        // `class Fwd;` declares nothing; a definition match here would put a bogus
        // hit at every forward declaration in every header.
        let code = "class Fwd;\nclass Fwd* Global;\n";
        assert!(
            cpp_find(code, "Fwd").is_empty(),
            "forward declaration must not be a definition"
        );
    }

    #[test]
    fn cpp_class_definition_outranks_its_usages() {
        // The day-to-day payoff: a class definition used to be reported as a *usage*
        // (its `class_specifier` was in no definition table), so search results led
        // with mentions rather than with the declaration.
        let dir = tempfile::tempdir().expect("tempdir");
        let scope = dir.path();
        std::fs::write(
            scope.join("Types.h"),
            "class BaseThing {};\nclass Derived final : public BaseThing {};\n",
        )
        .expect("write header");
        std::fs::write(
            scope.join("Use.cpp"),
            "#include \"Types.h\"\nvoid Take(BaseThing* T) {}\nvoid Also(BaseThing& R) {}\n",
        )
        .expect("write source");

        let result = search("BaseThing", scope, None, None, false).expect("search");
        assert_eq!(
            result.definitions, 1,
            "expected exactly one definition, got {result:?}"
        );
        let top = result.matches.first().expect("at least one match");
        assert!(
            top.is_definition,
            "the definition must rank first, got {top:?}"
        );
        assert_eq!(top.line, 1, "the definition is on line 1 of Types.h");
        assert!(
            result.usages >= 2,
            "expected the parameter mentions as usages, got {}",
            result.usages
        );
    }

    #[test]
    fn full_flag_raises_match_cap() {
        let dir = tempfile::tempdir().expect("tempdir");
        let scope = dir.path();

        // Create 15 Rust files each defining WidelyUsedThing.
        for i in 0..15 {
            let path = scope.join(format!("file_{i:02}.rs"));
            std::fs::write(&path, format!("pub fn WidelyUsedThing() {{}}\n")).expect("write");
        }

        let result_default =
            search("WidelyUsedThing", scope, None, None, false).expect("search default");
        let result_full = search("WidelyUsedThing", scope, None, None, true).expect("search full");

        // Default cap is 10 — should not exceed it.
        assert!(
            result_default.matches.len() <= 10,
            "default: expected ≤10 matches, got {}",
            result_default.matches.len()
        );
        // Full cap is 100 — all 15 definitions should be visible.
        assert!(
            result_full.matches.len() > 10,
            "full: expected >10 matches, got {}",
            result_full.matches.len()
        );
        // total_found is measured pre-truncation and should be equal.
        assert_eq!(
            result_default.total_found, result_full.total_found,
            "total_found must be the same regardless of full flag"
        );
    }
}
