//! File-level dependency analysis: what a file imports and what imports it.
//! Used by `tilth_deps` for blast-radius checks before breaking changes.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::TilthError;
use crate::lang::detect_file_type;
use crate::lang::outline::{extract_import_source, get_outline_entries};
use crate::read::imports::{is_external, is_import_line, resolve_local_imports};
use crate::search::callees::{extract_callee_names, resolve_callees};
use crate::search::callers::find_callers_batch;
use crate::types::{FileType, OutlineKind};

/// Maximum number of exported symbols to search for in the reverse direction.
const MAX_EXPORTED_SYMBOLS: usize = 25;

/// Maximum number of dependents to show before truncation.
const MAX_DEPENDENTS: usize = 15;

/// Maximum number of external dependencies to list before truncating.
///
/// The list was unbounded, which was tolerable while it only held `<system>` headers.
/// Reporting unresolvable quoted includes as external enlarged the population — a
/// framework header can pull in dozens — so it needs the same bound `used_by` has.
const MAX_EXTERNAL_DEPS: usize = 20;

/// Result of a full dependency analysis for a single file.
pub struct DepsResult {
    pub target: PathBuf,
    pub uses_local: Vec<LocalDep>,
    pub uses_external: Vec<String>,
    pub used_by: Vec<Dependent>,
    /// Total dependents found before truncation.
    pub total_dependents: usize,
    pub exported_count: usize,
    /// Actual number of symbols searched (may be < `exported_count` if capped).
    pub searched_count: usize,
}

/// A local file dependency with the symbols used from it.
pub struct LocalDep {
    pub path: PathBuf,
    pub symbols: Vec<String>,
}

/// A file that depends on the target, with symbol-level call detail.
pub struct Dependent {
    pub path: PathBuf,
    /// (`calling_function`, `called_symbol`, `line`) triples.
    pub symbols: Vec<(String, String, u32)>,
    pub is_test: bool,
}

/// Analyse the dependency graph for `path` within `scope`.
///
/// Phase 1: Extract exported symbols from the outline.
/// Phase 2: Forward dependencies — what this file uses.
/// Phase 3: Reverse dependencies — what uses this file.
pub fn analyze_deps(
    path: &Path,
    scope: &Path,
    bloom: &crate::index::bloom::BloomFilterCache,
) -> Result<DepsResult, TilthError> {
    // Canonicalize for reliable path comparison (callers return absolute paths).
    let path = &path.canonicalize().map_err(|e| TilthError::IoError {
        path: path.to_path_buf(),
        source: e,
    })?;

    let content = fs::read_to_string(path).map_err(|e| TilthError::IoError {
        path: path.clone(),
        source: e,
    })?;

    let FileType::Code(lang) = detect_file_type(path) else {
        // Non-code file: return empty deps gracefully.
        return Ok(DepsResult {
            target: path.clone(),
            uses_local: Vec::new(),
            uses_external: Vec::new(),
            used_by: Vec::new(),
            total_dependents: 0,
            exported_count: 0,
            searched_count: 0,
        });
    };

    // ── Phase 1: Extract exported symbols ────────────────────────────────────

    let entries = get_outline_entries(&content, lang);

    let mut all_names: Vec<String> = Vec::new();
    for entry in &entries {
        // Skip imports and re-export wrappers — they don't define symbols here.
        if matches!(entry.kind, OutlineKind::Import | OutlineKind::Export) {
            continue;
        }
        collect_symbol_names(entry, &mut all_names);
    }

    // Deduplicate
    all_names.sort();
    all_names.dedup();

    // Filter placeholder / noise names
    all_names.retain(|n| !is_placeholder_name(n));

    let exported_count = all_names.len();

    // Cap at MAX_EXPORTED_SYMBOLS, preferring longer (more specific) names
    let searched_count = if all_names.len() > MAX_EXPORTED_SYMBOLS {
        all_names.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        all_names.truncate(MAX_EXPORTED_SYMBOLS);
        MAX_EXPORTED_SYMBOLS
    } else {
        all_names.len()
    };

    // ── Phase 2: Forward dependencies ────────────────────────────────────────

    // The C/C++ include-root boundary, computed exactly once.
    //
    // Both halves of Phase 2 ask "does this include resolve?" — the local bucket below via
    // `resolve_local_imports`, the external bucket via `resolve_import_to_file` — and they
    // must get the same answer. Computing it separately in each place left the invariant
    // resting on a comment: desynchronising them puts an include in both buckets or, worse,
    // in neither, which is the silent-loss bug this whole path exists to avoid.
    let boundary = crate::read::imports::canonical_boundary(Some(scope));

    // Local deps via callee resolution
    let callee_names = extract_callee_names(&content, lang, None);
    let resolved = resolve_callees(&callee_names, path, &content, bloom, boundary.as_deref());

    // Group resolved callees by file
    let mut local_by_file: HashMap<PathBuf, Vec<String>> = HashMap::new();
    for callee in resolved {
        if callee.file != *path {
            local_by_file
                .entry(callee.file)
                .or_default()
                .push(callee.name);
        }
    }

    // Merge in import-resolved files (may not have resolved callees if symbols
    // weren't matched, but the import relationship itself is meaningful)
    // Uncapped: `resolve_related_files_with_content` truncates to 8 for the read-time
    // hint, and truncating here loses dependencies outright — an import that resolves is
    // excluded from the external bucket below, so one dropped by a cap would appear in
    // neither list. A C++ translation unit with more than 8 project includes is ordinary.
    let import_files = resolve_local_imports(path, &content, boundary.as_deref());
    for import_path in import_files {
        local_by_file.entry(import_path).or_default();
    }

    // Sort symbols within each dep, then build the list sorted by path
    let mut uses_local: Vec<LocalDep> = local_by_file
        .into_iter()
        .map(|(dep_path, mut syms)| {
            syms.sort();
            syms.dedup();
            LocalDep {
                path: dep_path,
                symbols: syms,
            }
        })
        .collect();
    uses_local.sort_by(|a, b| a.path.cmp(&b.path));

    // External deps via line-level import parsing
    let mut external_set: HashSet<String> = HashSet::new();
    for line in content.lines() {
        if !is_import_line(line, lang) {
            continue;
        }
        let source = extract_import_source(line, Some(lang));
        if source.is_empty() {
            continue;
        }
        // A quoted C/C++ include is "local" only in the sense that it is not a system
        // header — whether tilth can *see* the file is a separate question, and the
        // answer depends on include paths that live in the build system. When it does
        // not resolve on disk the dependency is real but outside scope, so surface it
        // here instead of dropping it: previously `#include "Engine/Widget.h"` appeared
        // in neither `uses_local` (unresolvable) nor `uses_external` (not a system
        // header), silently under-reporting a header's forward dependencies.
        //
        // Scoped to C/C++ deliberately. Other languages' relative imports normally do
        // resolve on disk, so applying the same fallback there would reclassify genuine
        // resolution failures as external deps.
        let external = is_external(&source, lang);
        let unresolved_local = !external
            && matches!(lang, crate::types::Lang::C | crate::types::Lang::Cpp)
            && path.parent().is_none_or(|dir| {
                crate::read::imports::resolve_import_to_file(
                    dir,
                    &source,
                    lang,
                    boundary.as_deref(),
                )
                .is_none()
            });

        if (!external && !unresolved_local) || is_stdlib(&source, lang) {
            continue;
        }
        // C/C++ `extract_import_source` deliberately keeps the `<…>` / `"…"`
        // delimiters so `is_external` can tell a system header from a
        // project-relative one. Strip them before recording: `is_valid_module_path`
        // requires an alphanumeric first character, so a leading `<` silently
        // dropped every `#include <…>` and left `uses_external` permanently empty
        // for C and C++. Other languages' sources never carry these delimiters
        // (`extract_import_source` already strips JS/TS quotes), so this is a no-op
        // for them.
        let module = source.trim_matches(|c| c == '<' || c == '>' || c == '"');
        if is_valid_module_path(module) {
            external_set.insert(module.to_string());
        }
    }
    let mut uses_external: Vec<String> = external_set.into_iter().collect();
    uses_external.sort();

    // ── Phase 3: Reverse dependencies ────────────────────────────────────────

    let mut used_by = if searched_count > 0 {
        let symbols_set: HashSet<String> = all_names.iter().cloned().collect();
        let raw_matches = find_callers_batch(&symbols_set, scope, bloom, None)?;

        // Group by file path
        let mut by_file: HashMap<PathBuf, Vec<(String, String, u32)>> = HashMap::new();
        for (matched_symbol, caller_match) in raw_matches {
            // Exclude calls from within the target file itself (self-references)
            if caller_match.path == *path {
                continue;
            }
            by_file.entry(caller_match.path).or_default().push((
                caller_match.calling_function,
                matched_symbol,
                caller_match.line,
            ));
        }

        // Build Dependent list
        let target_dir = path.parent();
        let mut dependents: Vec<Dependent> = by_file
            .into_iter()
            .map(|(dep_path, mut pairs)| {
                pairs.sort();
                pairs.dedup();
                let is_test = is_test_file(&dep_path);
                Dependent {
                    path: dep_path,
                    symbols: pairs,
                    is_test,
                }
            })
            .collect();

        // Sort: same directory first, non-tests before tests, then alphabetical
        dependents.sort_by(|a, b| {
            let a_same_dir = target_dir.is_some_and(|d| a.path.parent() == Some(d));
            let b_same_dir = target_dir.is_some_and(|d| b.path.parent() == Some(d));
            b_same_dir
                .cmp(&a_same_dir)
                .then_with(|| a.is_test.cmp(&b.is_test))
                .then_with(|| a.path.cmp(&b.path))
        });

        dependents
    } else {
        Vec::new()
    };

    let total_dependents = used_by.len();
    used_by.truncate(MAX_DEPENDENTS);

    Ok(DepsResult {
        target: path.clone(),
        uses_local,
        uses_external,
        used_by,
        total_dependents,
        exported_count,
        searched_count,
    })
}

/// Format a `DepsResult` as a compact, readable string.
///
/// Budget truncation priority (when `budget` tokens is too tight):
/// 1. Truncate "Used by" entries (keep header count)
/// 2. Truncate "Uses (external)" to count only
/// 3. Truncate "Uses (local)" symbol lists to file paths only
/// 4. Never truncate the header line
pub fn format_deps(result: &DepsResult, scope: &Path, budget: Option<usize>) -> String {
    let dep_count = result.total_dependents;
    let (prod_deps, test_deps): (Vec<_>, Vec<_>) = result.used_by.iter().partition(|d| !d.is_test);

    // ── Build sections (full fidelity first) ─────────────────────────────────

    // Header
    let rel_target = result
        .target
        .strip_prefix(scope)
        .unwrap_or(&result.target)
        .display()
        .to_string();
    let header = format!(
        "# Deps: {} — {} local, {} external, {} dependent{}",
        rel_target,
        result.uses_local.len(),
        result.uses_external.len(),
        dep_count,
        if dep_count == 1 { "" } else { "s" },
    );

    let uses_local_section = format_uses_local(&result.uses_local, scope, true);
    let uses_external_section = format_uses_external(&result.uses_external);
    let used_by_section = format_used_by(&prod_deps, scope, "## Used by");
    let used_by_tests_section = format_used_by(&test_deps, scope, "## Used by (tests)");

    let barrel_note = if result.exported_count > MAX_EXPORTED_SYMBOLS {
        format!(
            "\n\n> ({} of {} exports shown — barrel file detected)",
            result.searched_count, result.exported_count
        )
    } else {
        String::new()
    };

    // Full output
    let mut parts: Vec<String> = Vec::new();
    parts.push(header.clone());
    if !uses_local_section.is_empty() {
        parts.push(uses_local_section.clone());
    }
    if !uses_external_section.is_empty() {
        parts.push(uses_external_section.clone());
    }
    if !used_by_section.is_empty() {
        parts.push(used_by_section.clone());
    }
    if !used_by_tests_section.is_empty() {
        parts.push(used_by_tests_section.clone());
    }
    let truncated = result.total_dependents.saturating_sub(result.used_by.len());
    if truncated > 0 {
        parts.push(format!("... and {truncated} more dependents"));
    }
    if !barrel_note.is_empty() {
        parts.push(barrel_note.clone());
    }

    let full = parts.join("\n\n");
    let full_tokens = crate::types::estimate_tokens(full.len() as u64) as usize;

    let output = match budget {
        None => full,
        Some(b) if full_tokens <= b => full,
        Some(b) => {
            // Apply truncation in priority order
            apply_budget_truncation(
                &header,
                &uses_local_section,
                &uses_external_section,
                &prod_deps,
                &test_deps,
                &barrel_note,
                scope,
                b,
            )
        }
    };

    let token_est = crate::types::estimate_tokens(output.len() as u64);
    format!("{output}\n\n[~{token_est} tokens]")
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Collect symbol names from an outline entry and its children.
fn collect_symbol_names(entry: &crate::types::OutlineEntry, out: &mut Vec<String>) {
    out.push(entry.name.clone());
    for child in &entry.children {
        // Include public methods of classes/structs/impls
        if !matches!(child.kind, OutlineKind::Import | OutlineKind::Export) {
            out.push(child.name.clone());
        }
    }
}

/// Returns true if the name is a noise/placeholder that should be excluded
/// from the reverse-dependency search.
fn is_placeholder_name(name: &str) -> bool {
    if name == "<anonymous>" {
        return true;
    }
    if name.starts_with('<') {
        return true;
    }
    if name.starts_with("impl ") {
        return true;
    }
    // Single-character names are too generic (e.g. `T`, `E`, `f`)
    if name.chars().count() == 1 {
        return true;
    }
    false
}

/// Root (first `/`-segment) of each Go stdlib package. A Go import is stdlib
/// when its first path segment is one of these — covering both single-segment
/// (`fmt`) and multi-segment (`net/http`, `encoding/json`) forms. Matching the
/// root (not the whole path) avoids misclassifying a local package like
/// `mypackage` while still suppressing the noisy multi-segment stdlib paths.
const GO_STDLIB_ROOTS: &[&str] = &[
    "archive",
    "bufio",
    "bytes",
    "cmp",
    "compress",
    "container",
    "context",
    "crypto",
    "database",
    "debug",
    "embed",
    "encoding",
    "errors",
    "flag",
    "fmt",
    "go",
    "hash",
    "html",
    "image",
    "index",
    "io",
    "log",
    "maps",
    "math",
    "mime",
    "net",
    "os",
    "path",
    "plugin",
    "reflect",
    "regexp",
    "runtime",
    "slices",
    "sort",
    "strconv",
    "strings",
    "sync",
    "syscall",
    "testing",
    "text",
    "time",
    "unicode",
    "unsafe",
    // Go 1.23+ additions
    "iter",
    "unique",
];

/// Returns true if the import source is a standard library module.
/// Agents can't navigate into stdlib — showing these is noise.
fn is_stdlib(source: &str, lang: crate::types::Lang) -> bool {
    use crate::types::Lang;
    match lang {
        Lang::Rust => {
            source.starts_with("std::")
                || source.starts_with("core::")
                || source.starts_with("alloc::")
        }
        Lang::Python => {
            // Common stdlib modules — not exhaustive but covers the noisy ones
            matches!(
                source.split('.').next().unwrap_or(""),
                "os" | "sys"
                    | "re"
                    | "json"
                    | "math"
                    | "time"
                    | "datetime"
                    | "pathlib"
                    | "typing"
                    | "collections"
                    | "functools"
                    | "itertools"
                    | "abc"
                    | "io"
                    | "logging"
                    | "unittest"
                    | "dataclasses"
                    | "enum"
                    | "copy"
                    | "hashlib"
                    | "subprocess"
                    | "threading"
                    | "asyncio"
            )
        }
        Lang::Go => GO_STDLIB_ROOTS.contains(&source.split('/').next().unwrap_or(source)),
        _ => false,
    }
}

/// Returns true if the string looks like a valid module/package path.
/// Filters out garbage from string literals that pass `is_import_line`.
fn is_valid_module_path(source: &str) -> bool {
    // Must not contain spaces (real module paths don't)
    if source.contains(' ') {
        return false;
    }
    // Must start with an alphanumeric, @, or dot
    source
        .chars()
        .next()
        .is_some_and(|c| c.is_alphanumeric() || c == '@' || c == '.')
}

use crate::types::is_test_file;

/// Format the "Uses (local)" section.
fn format_uses_local(deps: &[LocalDep], scope: &Path, with_symbols: bool) -> String {
    if deps.is_empty() {
        return String::new();
    }
    let mut out = String::from("## Uses (local)");
    for dep in deps {
        let rel = dep
            .path
            .strip_prefix(scope)
            .unwrap_or(&dep.path)
            .display()
            .to_string();
        if with_symbols && !dep.symbols.is_empty() {
            let _ = write!(out, "\n{:<30} {}", rel, dep.symbols.join(", "));
        } else {
            let _ = write!(out, "\n{rel}");
        }
    }
    out
}

/// Format the "Uses (external)" section.
fn format_uses_external(externals: &[String]) -> String {
    if externals.is_empty() {
        return String::new();
    }
    let mut out = String::from("## Uses (external)");
    for ext in externals.iter().take(MAX_EXTERNAL_DEPS) {
        let _ = write!(out, "\n{ext}");
    }
    if externals.len() > MAX_EXTERNAL_DEPS {
        let _ = write!(
            out,
            "\n... and {} more",
            externals.len() - MAX_EXTERNAL_DEPS
        );
    }
    out
}

/// Format a "Used by" section from a slice of dependents.
fn format_used_by(deps: &[&Dependent], scope: &Path, heading: &str) -> String {
    if deps.is_empty() {
        return String::new();
    }
    let mut out = String::from(heading);
    for dep in deps {
        let rel = dep
            .path
            .strip_prefix(scope)
            .unwrap_or(&dep.path)
            .display()
            .to_string();
        // Group by (caller, line) for readability — keep the earliest line per caller
        let mut by_caller: HashMap<&str, (u32, Vec<&str>)> = HashMap::new();
        for (caller, symbol, line) in &dep.symbols {
            let entry = by_caller
                .entry(caller.as_str())
                .or_insert((*line, Vec::new()));
            entry.0 = entry.0.min(*line);
            entry.1.push(symbol.as_str());
        }
        let mut callers: Vec<(&str, u32, Vec<&str>)> = by_caller
            .into_iter()
            .map(|(caller, (line, syms))| (caller, line, syms))
            .collect();
        callers.sort_unstable_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(b.0)));
        for (caller, line, syms) in callers {
            let loc = format!("{rel}:{line}");
            let joined = syms.join(", ");
            let _ = write!(out, "\n{loc:<30} {caller:<20} \u{2192} {joined}");
        }
    }
    out
}

/// Apply progressive budget truncation and reassemble the output.
#[allow(clippy::too_many_arguments)]
fn apply_budget_truncation(
    header: &str,
    uses_local_full: &str,
    uses_external_full: &str,
    prod_deps: &[&Dependent],
    test_deps: &[&Dependent],
    barrel_note: &str,
    scope: &Path,
    budget: usize,
) -> String {
    // Try progressively degraded versions
    #[allow(clippy::type_complexity)]
    let candidates: &[fn(
        &str,
        &str,
        &str,
        &[&Dependent],
        &[&Dependent],
        &str,
        &Path,
    ) -> String] = &[
        // Level 0: no tests
        |hdr, ul, ue, pd, _td, bn, sc| {
            assemble(&[hdr, ul, ue, &format_used_by(pd, sc, "## Used by"), bn])
        },
        // Level 1: no used-by entries at all
        |hdr, ul, ue, pd, _td, bn, _sc| {
            let count = pd.len();
            let note = if count > 0 {
                format!("\n\n(... {count} more dependents)")
            } else {
                String::new()
            };
            assemble(&[hdr, ul, ue, &note, bn])
        },
        // Level 2: external as count only
        |hdr, ul, _ue, _pd, _td, bn, _sc| assemble(&[hdr, ul, bn]),
        // Level 3: local as paths only (no symbols)
        |hdr, ul, _ue, _pd, _td, _bn, _sc| {
            // Strip symbol lists: each line is "path_padded  symbols" — take only up to first space run
            let local_lines: Vec<&str> = ul
                .lines()
                .skip(1) // skip heading
                .map(|l| l.split_whitespace().next().unwrap_or(l))
                .collect();
            let paths_only = if local_lines.is_empty() {
                String::new()
            } else {
                format!("## Uses (local)\n{}", local_lines.join("\n"))
            };
            assemble(&[hdr, &paths_only])
        },
        // Level 4: header only
        |hdr, _ul, _ue, _pd, _td, _bn, _sc| hdr.to_string(),
    ];

    for candidate_fn in candidates {
        let candidate = candidate_fn(
            header,
            uses_local_full,
            uses_external_full,
            prod_deps,
            test_deps,
            barrel_note,
            scope,
        );
        let tokens = crate::types::estimate_tokens(candidate.len() as u64) as usize;
        if tokens <= budget {
            return candidate;
        }
    }

    // Absolute fallback: just the header
    header.to_string()
}

/// Join non-empty parts with double newlines.
fn assemble(parts: &[&str]) -> String {
    parts
        .iter()
        .filter(|s| !s.trim().is_empty())
        .copied()
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `extract_import_source` keeps the `<…>` / `"…"` delimiters on a C/C++ include
    /// so `is_external` can tell a system header from a project-relative one — but
    /// `is_valid_module_path` requires an alphanumeric first character, so a leading
    /// `<` silently dropped every `#include <…>` and left `uses_external` permanently
    /// empty for C and C++.
    /// Pins the delimiter-stripping through the real `analyze_deps` path rather than by
    /// re-implementing it: asserting `is_valid_module_path("<vector>") == false` and
    /// then trimming inline would pass with the production fix reverted.
    #[test]
    fn cpp_system_includes_are_reported_as_external_deps() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(
            root.join("Uses.cpp"),
            "#include <vector>\n#include <memory>\n#include \"Local.h\"\nvoid F() {}\n",
        )
        .unwrap();
        std::fs::write(root.join("Local.h"), "struct S { int V; };\n").unwrap();

        let bloom = crate::index::bloom::BloomFilterCache::new();
        let result = analyze_deps(&root.join("Uses.cpp"), root, &bloom).unwrap();

        assert_eq!(
            result.uses_external,
            vec!["memory".to_string(), "vector".to_string()],
            "angle-bracket includes must survive module-path validation, stripped of \
             their delimiters"
        );
        // The quoted include is a local dep, not an external one.
        assert!(
            result
                .uses_local
                .iter()
                .any(|d| d.path.ends_with("Local.h")),
            "quoted include should resolve as local, got {:?}",
            result
                .uses_local
                .iter()
                .map(|d| &d.path)
                .collect::<Vec<_>>()
        );
    }

    /// Both halves of #17, driven through the function that actually produces the buckets.
    ///
    /// The unit tests in `read::imports` stop at resolution; the acceptance criteria are
    /// phrased in terms of `uses_local` / `uses_external`, and reaching those also runs
    /// `is_valid_module_path` and `is_stdlib`, which no other #17 test exercises.
    ///
    /// Two independent bugs, one fixture, because they compound: `Main.cpp` sits at the
    /// scope root — so the include-root walk had no ancestor inside the root to find
    /// `include/` from — *and* spells its includes with a space after the `#`, which meant
    /// no consumer saw them as includes at all. Before the fix this file reported
    /// `0 local, 0 external`: both dependencies gone, in neither bucket, with no warning.
    #[test]
    fn cpp_spaced_includes_at_the_scope_root_reach_the_right_buckets() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("include/lib")).unwrap();
        std::fs::write(root.join("include/lib/db.h"), "struct DB { int V; };\n").unwrap();
        std::fs::write(
            root.join("Main.cpp"),
            "#  include \"lib/db.h\"\n# include <vector>\nint main() { return 0; }\n",
        )
        .unwrap();

        let bloom = crate::index::bloom::BloomFilterCache::new();
        let result = analyze_deps(&root.join("Main.cpp"), root, &bloom).unwrap();

        assert!(
            result.uses_local.iter().any(|d| d.path.ends_with("db.h")),
            "a file at the scope root must reach its own include/ sibling, got {:?}",
            result
                .uses_local
                .iter()
                .map(|d| &d.path)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            result.uses_external,
            vec!["vector".to_string()],
            "the spaced angle include must still bucket as external"
        );
        // The header must not be double-counted: resolving locally excludes it from the
        // external bucket, and a regression that broke only the walk would show up here as
        // `lib/db.h` appearing alongside `vector`.
        assert!(
            !result.uses_external.iter().any(|e| e.contains("db.h")),
            "a resolved include must not also appear as external: {:?}",
            result.uses_external
        );
    }

    /// A UTF-8 BOM on line 1, driven through the function that produces the buckets.
    ///
    /// U+FEFF is not Unicode `White_Space`, so `trim_start` left it in front of the first
    /// import and every line-prefix test failed. The import went into neither `uses_local`
    /// nor `uses_external` — the same silent-drop shape as the trailing-comment and
    /// spaced-`#` include bugs, from a third direction and for every language at once.
    ///
    /// The `read::imports` unit tests stop at local resolution; this path also runs
    /// `is_valid_module_path` and `is_stdlib`, which is where the external half of the
    /// acceptance criteria lives.
    ///
    /// Only line 1 carries the BOM, so which import sits there decides which bucket the
    /// test actually exercises. The cases put a *local* import on line 1 for C++, Rust and
    /// Python, then repeat C++ with the order reversed so an external import takes the
    /// BOM too — without that row the `uses_external` half of the comparison comes from
    /// two identical un-BOM'd lines and is satisfied trivially.
    ///
    /// Each variant gets its own tempdir so the reverse-dependency scan cannot see the
    /// other one's copy of the fixture.
    #[test]
    fn bom_on_line_one_reaches_the_same_buckets_as_an_unmarked_file() {
        const UTF8_BOM: &[u8] = &[0xEF, 0xBB, 0xBF];

        // (subdirectory, entry file, body, sibling files, expected local, expected external)
        let cases: &[(&str, &str, &str, &[(&str, &str)], &str, &[&str])] = &[
            (
                "",
                "Main.cpp",
                "#include \"Local.h\"\n#include <vector>\nvoid F() {}\n",
                &[("Local.h", "struct S { int V; };\n")],
                "Local.h",
                &["vector"],
            ),
            (
                // `crate::` resolution needs a `src` ancestor to anchor on.
                "src",
                "main.rs",
                "use crate::helper;\nuse serde::Deserialize;\nfn f() { helper::go(); }\n",
                &[("helper.rs", "pub fn go() {}\n")],
                "helper.rs",
                &["serde::Deserialize"],
            ),
            (
                "",
                "app.py",
                "from .mod_a import X\nimport requests\n",
                &[("mod_a.py", "X = 1\n")],
                "mod_a.py",
                &["requests"],
            ),
            (
                // The BOM lands on the system include this time, so the external bucket is
                // the one under test: detection, `is_valid_module_path` and the `<…>`
                // delimiter stripping all have to survive it.
                "",
                "Reversed.cpp",
                "#include <vector>\n#include \"Local.h\"\nvoid F() {}\n",
                &[("Local.h", "struct S { int V; };\n")],
                "Local.h",
                &["vector"],
            ),
        ];

        for (subdir, entry, body, siblings, want_local, want_external) in cases {
            let mut buckets = Vec::new();
            for prefix in [&[][..], UTF8_BOM] {
                let dir = tempfile::tempdir().unwrap();
                let root = dir.path();
                let src = root.join(subdir);
                std::fs::create_dir_all(&src).unwrap();
                for (name, contents) in *siblings {
                    std::fs::write(src.join(name), contents).unwrap();
                }
                let mut bytes = prefix.to_vec();
                bytes.extend_from_slice(body.as_bytes());
                std::fs::write(src.join(entry), &bytes).unwrap();

                let bloom = crate::index::bloom::BloomFilterCache::new();
                let result = analyze_deps(&src.join(entry), root, &bloom).unwrap();
                buckets.push((
                    result
                        .uses_local
                        .iter()
                        .map(|d| {
                            std::path::Path::new(&d.path)
                                .file_name()
                                .unwrap()
                                .to_string_lossy()
                                .into_owned()
                        })
                        .collect::<Vec<_>>(),
                    result.uses_external,
                ));
            }

            let (plain, bom) = (&buckets[0], &buckets[1]);
            // Pin the un-BOM'd baseline against literals, so the comparison below cannot
            // pass by both variants reporting nothing.
            assert_eq!(
                plain.0,
                vec![want_local.to_string()],
                "fixture is broken: {entry} without a BOM reported local {:?}",
                plain.0
            );
            assert_eq!(
                plain.1,
                want_external
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>(),
                "fixture is broken: {entry} without a BOM reported external {:?}",
                plain.1
            );
            assert_eq!(
                bom, plain,
                "a BOM changed the dependency buckets of {entry}"
            );
        }
    }

    /// A quoted include that does not resolve on disk used to vanish from both lists:
    /// not a system header, so not "external", and unresolvable, so not "local". Any
    /// project whose headers sit behind a build-system include path — most non-trivial
    /// C++ builds — therefore had its forward dependencies under-reported.
    #[test]
    fn cpp_unresolvable_quoted_include_is_reported_as_external() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("Sibling.h"), "struct Sibling { int V; };\n").unwrap();
        std::fs::write(
            root.join("Consumer.cpp"),
            "#include <vector>\n\
             #include \"Sibling.h\"\n\
             #include \"Engine/Faraway.h\"\n\
             void F() {}\n",
        )
        .unwrap();

        let bloom = crate::index::bloom::BloomFilterCache::new();
        let result = analyze_deps(&root.join("Consumer.cpp"), root, &bloom).unwrap();

        // Resolvable quoted include stays local.
        assert!(
            result
                .uses_local
                .iter()
                .any(|d| d.path.ends_with("Sibling.h")),
            "resolvable quoted include should be local, got {:?}",
            result
                .uses_local
                .iter()
                .map(|d| &d.path)
                .collect::<Vec<_>>()
        );
        // System header and the unresolvable quoted one are both reported.
        assert!(
            result.uses_external.iter().any(|e| e == "vector"),
            "expected `vector`, got {:?}",
            result.uses_external
        );
        assert!(
            result.uses_external.iter().any(|e| e == "Engine/Faraway.h"),
            "unresolvable quoted include must still be reported, got {:?}",
            result.uses_external
        );
        // It must not be double-counted as local.
        assert!(
            !result
                .uses_local
                .iter()
                .any(|d| d.path.to_string_lossy().contains("Faraway")),
            "unresolvable include must not appear as a local dep"
        );
    }

    /// End-to-end for the include-root case, which is the common layout: a header
    /// including `"lib/other.h"` where that path is relative to an include root one level
    /// up. Such includes previously resolved to nothing and were therefore reported as
    /// *external* — a project header misfiled as a third-party dependency, with local
    /// deps stuck at zero.
    #[test]
    fn cpp_include_relative_to_an_include_root_is_local_not_external() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::create_dir_all(root.join("include/lib")).unwrap();
        std::fs::write(
            root.join("include/lib/status.h"),
            "struct Status { int V; };\n",
        )
        .unwrap();
        std::fs::write(
            root.join("include/lib/env.h"),
            "#pragma once\n\
             #include <vector>\n\
             #include \"lib/status.h\"\n\
             class Env { public: void Sync(); };\n",
        )
        .unwrap();

        let bloom = crate::index::bloom::BloomFilterCache::new();
        let result = analyze_deps(&root.join("include/lib/env.h"), root, &bloom).unwrap();

        assert!(
            result
                .uses_local
                .iter()
                .any(|d| d.path.ends_with("status.h")),
            "an include-root-relative header must be local, got local={:?} external={:?}",
            result
                .uses_local
                .iter()
                .map(|d| &d.path)
                .collect::<Vec<_>>(),
            result.uses_external
        );
        assert!(
            !result.uses_external.iter().any(|e| e.contains("status.h")),
            "it must not also be reported as external, got {:?}",
            result.uses_external
        );
        // The genuine system header is still external.
        assert!(result.uses_external.iter().any(|e| e == "vector"));
    }

    /// A code-generating framework's body macro must not become an exported symbol.
    ///
    /// `GENERATED_BODY()` in a class body is a typeless `declaration` — the same shape as
    /// a constructor — so it was outlined as a member and became one of the header's
    /// "exported symbols". `tilth_deps` then searched for call sites of it, and since
    /// every such header invokes the same macro, reported files that never include the
    /// header as dependents. On a real tree that is most of the project.
    ///
    /// The two class forms here are both needed to reproduce it: a cleanly-parsed class
    /// exports the macro name as a symbol, while a class behind an export macro misparses
    /// so the same text becomes a *call site* that matches it.
    #[test]
    fn cpp_framework_body_macro_does_not_create_false_dependents() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        // Parses cleanly -> would export `BODY_MACRO` as a symbol.
        std::fs::write(
            root.join("HeroComponent.h"),
            "#pragma once\n\
             ANNOTATE()\n\
             class UHeroComponent : public UPawnComponent\n\
             {\n\
             \tBODY_MACRO()\n\
             public:\n\
             \tstatic const char* GetLogNameSafe(const Actor* Subject);\n\
             };\n",
        )
        .unwrap();
        // Behind an export macro -> misparses, so `BODY_MACRO()` here is a call site.
        // It does not include the header above.
        std::fs::write(
            root.join("HudTypes.h"),
            "#pragma once\n\
             ANNOTATE()\n\
             class MYLIB_API UHudTypes final : public UObject\n\
             {\n\
             \tBODY_MACRO()\n\
             public:\n\
             \tvoid Layout();\n\
             };\n",
        )
        .unwrap();

        let bloom = crate::index::bloom::BloomFilterCache::new();
        let result = analyze_deps(&root.join("HeroComponent.h"), root, &bloom).unwrap();

        assert!(
            result.used_by.is_empty(),
            "no file includes this header, so it has no dependents; got {:?}",
            result
                .used_by
                .iter()
                .map(|d| (&d.path, &d.symbols))
                .collect::<Vec<_>>()
        );
    }

    /// A trailing comment on an `#include` must not delete the dependency.
    ///
    /// The header name used to be the whole rest of the line, so `#include "X.h" // note`
    /// asked the filesystem for `X.h" // note`, which does not exist — and `is_external`
    /// still saw the leading quote, so the include was neither local nor external. It
    /// vanished, with nothing in the output to say a line had been skipped. Per-line, so a
    /// file mixing commented and clean includes under-reported by exactly the commented
    /// ones.
    #[test]
    fn cpp_include_with_a_trailing_comment_still_counts_as_a_dependency() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(root.join("Sibling.h"), "#pragma once\nclass Sibling {};\n").unwrap();
        std::fs::create_dir_all(root.join("Sub")).unwrap();
        std::fs::write(
            root.join("Sub/Nested.h"),
            "#pragma once\nclass Nested {};\n",
        )
        .unwrap();
        std::fs::write(
            root.join("Target.h"),
            "#pragma once\n\
             #include \"Sibling.h\" // line comment\n\
             #include \"Sub/Nested.h\" /* block comment */\n\
             #include <vector> // still a system header\n\
             class Target {};\n",
        )
        .unwrap();

        let bloom = crate::index::bloom::BloomFilterCache::new();
        let result = analyze_deps(&root.join("Target.h"), root, &bloom).unwrap();

        let local: Vec<String> = result
            .uses_local
            .iter()
            .map(|d| d.path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert!(
            local.contains(&"Sibling.h".to_string()) && local.contains(&"Nested.h".to_string()),
            "both commented includes must resolve, got {local:?}"
        );
        assert!(
            result.uses_external.iter().any(|e| e == "vector"),
            "the commented system header must still be external, got {:?}",
            result.uses_external
        );
    }

    /// The include-root layout, end to end, in a tree that is not a git checkout.
    ///
    /// This is the shape that motivated include-root resolution: a module source root is an
    /// include root, so a header in `<Module>/Character/` writes `"Character/Peer.h"` to
    /// reach a sibling. Own-directory resolution turns that into
    /// `Character/Character/Peer.h`, which does not exist, and the dependency bucketed as
    /// external. Resolution was gated on finding `.git`, so it also did nothing at all
    /// outside a repository even though the caller had named a scope.
    #[test]
    fn cpp_include_root_relative_deps_resolve_under_a_declared_scope() {
        let dir = tempfile::tempdir().unwrap();
        let module = dir.path().join("Source").join("TheGame");
        for d in ["Character", "Camera"] {
            std::fs::create_dir_all(module.join(d)).unwrap();
        }
        // Deliberately no `.git`.
        std::fs::write(
            module.join("Character/PawnComponent.h"),
            "#pragma once\nclass UPawnComponent {};\n",
        )
        .unwrap();
        std::fs::write(
            module.join("Camera/CameraMode.h"),
            "#pragma once\nclass UCameraMode {};\n",
        )
        .unwrap();
        std::fs::write(
            module.join("Character/HeroComponent.h"),
            "#pragma once\n\
             #include \"Character/PawnComponent.h\"\n\
             #include \"Camera/CameraMode.h\"\n\
             #include \"OutOfTree/TagContainer.h\"\n\
             class UHeroComponent {};\n",
        )
        .unwrap();

        let bloom = crate::index::bloom::BloomFilterCache::new();
        let result =
            analyze_deps(&module.join("Character/HeroComponent.h"), &module, &bloom).unwrap();

        let local: Vec<String> = result
            .uses_local
            .iter()
            .map(|d| d.path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert!(
            local.contains(&"PawnComponent.h".to_string()),
            "a sibling reached via the include root must be local, got {local:?}"
        );
        assert!(
            local.contains(&"CameraMode.h".to_string()),
            "a peer directory reached via the include root must be local, got {local:?}"
        );
        assert!(
            result
                .uses_external
                .iter()
                .any(|e| e == "OutOfTree/TagContainer.h"),
            "an include with no file in scope stays external, got {:?}",
            result.uses_external
        );
    }

    /// Every include lands in exactly one bucket, whatever the scope is.
    ///
    /// Phase 2 asks "does this include resolve?" twice — once to build `uses_local`, once to
    /// decide `uses_external` — and the two must agree. While each computed its own
    /// boundary, desynchronising them was a one-line edit that no test noticed: the include
    /// then appeared in both lists, or in neither. "Neither" is the silent loss this whole
    /// path exists to prevent, so the invariant is asserted directly rather than left to a
    /// comment, and asserted across scopes that exercise both the declared-scope root and
    /// the repository fallback.
    #[test]
    fn cpp_every_include_lands_in_exactly_one_bucket() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("proj/.git")).unwrap();
        std::fs::create_dir_all(root.join("proj/Alpha")).unwrap();
        std::fs::create_dir_all(root.join("proj/Gamma")).unwrap();
        std::fs::create_dir_all(root.join("elsewhere")).unwrap();
        std::fs::write(
            root.join("proj/Alpha/Beta.h"),
            "#pragma once\nclass Beta {};\n",
        )
        .unwrap();
        std::fs::write(
            root.join("proj/Gamma/Delta.h"),
            "#pragma once\n\
             #include \"Alpha/Beta.h\"\n\
             #include <vector>\n\
             #include \"NotInTree/Absent.h\"\n\
             class Delta {};\n",
        )
        .unwrap();

        let target = root.join("proj/Gamma/Delta.h");
        let bloom = crate::index::bloom::BloomFilterCache::new();
        for scope in [
            root.join("proj"),
            root.join("elsewhere"),
            root.to_path_buf(),
        ] {
            let result = analyze_deps(&target, &scope, &bloom).unwrap();

            // `Alpha/Beta.h` resolves, so it belongs in local and must not also be external.
            let in_local = result.uses_local.iter().any(|d| d.path.ends_with("Beta.h"));
            let in_external = result.uses_external.iter().any(|e| e.contains("Beta.h"));
            assert!(
                in_local && !in_external,
                "scope {}: expected exactly local, got local={in_local} external={in_external} \
                 ({:?} / {:?})",
                scope.display(),
                result
                    .uses_local
                    .iter()
                    .map(|d| &d.path)
                    .collect::<Vec<_>>(),
                result.uses_external
            );

            // The unresolvable one must still be reported, as external.
            assert!(
                result
                    .uses_external
                    .iter()
                    .any(|e| e == "NotInTree/Absent.h"),
                "scope {}: an include with no file anywhere must stay external, got {:?}",
                scope.display(),
                result.uses_external
            );
        }
    }

    /// The same invariant in a tree with no `.git`, which is where the two buckets can
    /// actually diverge.
    ///
    /// With a repository present, a bucket that forgot the boundary still resolves via the
    /// `.git` fallback and agrees by luck — so the git fixture above cannot detect the
    /// desync at all (verified: it does not). Remove `.git` and the boundary becomes the
    /// only thing that resolves an include-root-relative path, so a bucket computed without
    /// it disagrees: the include lands in `uses_local` *and* `uses_external` at once.
    #[test]
    fn cpp_bucket_invariant_holds_in_a_tree_without_git() {
        let dir = tempfile::tempdir().unwrap();
        let module = dir.path().join("ModuleRoot");
        std::fs::create_dir_all(module.join("Alpha")).unwrap();
        std::fs::create_dir_all(module.join("Gamma")).unwrap();
        std::fs::write(
            module.join("Alpha/Beta.h"),
            "#pragma once\nclass Beta {};\n",
        )
        .unwrap();
        std::fs::write(
            module.join("Gamma/Delta.h"),
            "#pragma once\n#include \"Alpha/Beta.h\"\nclass Delta {};\n",
        )
        .unwrap();

        let bloom = crate::index::bloom::BloomFilterCache::new();
        let result = analyze_deps(&module.join("Gamma/Delta.h"), &module, &bloom).unwrap();

        let in_local = result.uses_local.iter().any(|d| d.path.ends_with("Beta.h"));
        let in_external = result.uses_external.iter().any(|e| e.contains("Beta.h"));
        assert!(
            in_local,
            "the include resolves under the declared scope, so it must be local: {:?}",
            result
                .uses_local
                .iter()
                .map(|d| &d.path)
                .collect::<Vec<_>>()
        );
        assert!(
            !in_external,
            "it must not ALSO be external — that means the two buckets used different \
             boundaries: {:?}",
            result.uses_external
        );
    }

    /// The post-read hint and `deps` must agree about the same include, in both directions,
    /// in a tree that is not a git checkout (#15).
    ///
    /// Only `deps` passed a boundary to the include resolver; the hint and callee resolution
    /// passed `None` and so needed a `.git` ancestor. Outside a repository that split the two
    /// apart: `deps` listed `Peer.h` as a local dependency while the hint did not mention it,
    /// and — because callee resolution shares the same resolver — the `uses_local` entry
    /// carried no symbols, so `deps` named the file without saying what it used from it.
    ///
    /// Both directions are asserted, not just "the hint caught up". During #10's review the
    /// asymmetry ran the *other* way, with `deps` resolving fewer includes than the hint,
    /// which a one-sided subset check would have passed.
    ///
    /// Set equality is the right assertion for *this* fixture specifically: every local
    /// dependency here arrives through an `#include`, so the callee-resolved half of
    /// `uses_local` cannot contribute a path the import-based hint lacks. The fixture also
    /// stays under `MAX_SUGGESTIONS`, or the hint's cap would break equality on its own.
    ///
    /// Scope: this asserts the two *resolvers* agree given the same boundary. It calls
    /// `resolve_related_files` directly, so it would not notice `tools::read` reverting to
    /// `None` — that half is pinned by
    /// `tool_read_hint_resolves_include_root_headers_under_a_declared_root`, which drives
    /// the real `tool_read`.
    #[test]
    fn cpp_read_hint_and_deps_agree_about_an_include_root_header_without_git() {
        let dir = tempfile::tempdir().unwrap();
        let module = dir.path().join("ModuleRoot");
        for d in ["Character", "Camera"] {
            std::fs::create_dir_all(module.join(d)).unwrap();
        }
        // Deliberately no `.git`: with one, both paths resolve via the repository fallback
        // and agree by luck, so the fixture could not detect the split at all.
        assert!(
            !module.join(".git").exists() && !dir.path().join(".git").exists(),
            "fixture must not sit inside a repository, or it proves nothing"
        );
        std::fs::write(
            module.join("Character/Peer.h"),
            "#pragma once\nvoid ConnectPeer(int id);\n",
        )
        .unwrap();
        std::fs::write(
            module.join("Camera/CameraMode.h"),
            "#pragma once\nvoid ApplyCameraMode(int mode);\n",
        )
        .unwrap();
        // Written relative to the module root, not to the including file: own-directory
        // resolution looks for `Character/Character/Peer.h` and finds nothing.
        let target = module.join("Character/HeroComponent.h");
        std::fs::write(
            &target,
            "#pragma once\n\
             #include \"Character/Peer.h\"\n\
             #include \"Camera/CameraMode.h\"\n\
             void Setup() {\n\
             \x20   ConnectPeer(1);\n\
             \x20   ApplyCameraMode(2);\n\
             }\n",
        )
        .unwrap();

        // The hint's boundary is the declared project root; deps' is the declared scope.
        // Here they are the same directory, which is the case the two must agree on.
        let boundary = crate::read::imports::canonical_boundary(Some(&module));
        let hint = crate::read::imports::resolve_related_files(&target, boundary.as_deref());

        let bloom = crate::index::bloom::BloomFilterCache::new();
        let result = analyze_deps(&target, &module, &bloom).unwrap();

        // Compare canonical forms: deps canonicalizes its target before resolving, the hint
        // does not, and on Windows that is the difference between `C:\x` and `\\?\C:\x`.
        let canon = |p: &std::path::Path| p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
        let hint_set: HashSet<PathBuf> = hint.iter().map(|p| canon(p)).collect();
        let deps_set: HashSet<PathBuf> = result.uses_local.iter().map(|d| canon(&d.path)).collect();

        assert!(
            hint_set.iter().any(|p| p.ends_with("Peer.h")),
            "the include-root-relative header must reach the hint: {hint_set:?}"
        );
        assert_eq!(
            hint_set, deps_set,
            "the hint and deps must resolve the same includes to the same files"
        );

        // The half that made `deps` output visibly thinner: callee resolution shares the
        // hint's resolver, so with no boundary it could not open `Peer.h` and the entry
        // named the file with an empty symbol list.
        let peer = result
            .uses_local
            .iter()
            .find(|d| d.path.ends_with("Peer.h"))
            .expect("Peer.h must be a local dependency");
        assert!(
            peer.symbols.iter().any(|s| s == "ConnectPeer"),
            "the local dep must carry per-symbol detail, got {:?}",
            peer.symbols
        );
    }

    /// Callee resolution must see every import, not the first eight.
    ///
    /// `resolve_related_files_with_content` truncates to `MAX_SUGGESTIONS` (8) because it
    /// feeds a display hint, and its own doc says callers needing completeness must use
    /// `resolve_local_imports` instead. `resolve_callees` was on the wrong side of that
    /// line, and the ninth import onwards was never opened — so a symbol defined there was
    /// reported as external.
    ///
    /// #15 turned that latent bug into a live regression. Include-root-relative includes
    /// used to resolve to nothing in a non-git tree and cost no slots; once the boundary
    /// made them resolve, they filled the cap and *evicted* the plain sibling include that
    /// had been resolving all along. `deps` re-merges the uncapped import list, so the file
    /// still appears in `uses_local` — but with no symbols, which is the exact "names the
    /// file without saying what it uses" failure #15 set out to remove. `grok` has no such
    /// merge and calls the symbol `extern` outright.
    ///
    /// The sibling is asserted specifically because it is the eviction victim: it resolves
    /// with or without a boundary, so if it loses its symbols the cap is the only cause.
    #[test]
    fn cpp_callee_resolution_sees_imports_past_the_suggestion_cap() {
        let dir = tempfile::tempdir().unwrap();
        let module = dir.path().join("ModuleRoot");
        std::fs::create_dir_all(module.join("A")).unwrap();
        std::fs::create_dir_all(module.join("Sub")).unwrap();
        assert!(
            !module.join(".git").exists() && !dir.path().join(".git").exists(),
            "fixture must not sit inside a repository, or it proves nothing"
        );

        // 8 include-root-relative headers — exactly MAX_SUGGESTIONS, so they fill the cap
        // on their own and anything after them is truncated away.
        let mut includes = String::new();
        let mut calls = String::new();
        for i in 0..8 {
            std::fs::write(
                module.join(format!("A/a{i}.h")),
                format!("#pragma once\nvoid AFunc{i}(int v);\n"),
            )
            .unwrap();
            let _ = writeln!(includes, "#include \"A/a{i}.h\"");
            let _ = writeln!(calls, "    AFunc{i}(1);");
        }
        // The plain sibling: resolves by the direct check, with or without a boundary.
        std::fs::write(
            module.join("Sub/Sib.h"),
            "#pragma once\nvoid SibFunc(int v);\n",
        )
        .unwrap();
        includes.push_str("#include \"Sib.h\"\n");
        calls.push_str("    SibFunc(2);\n");

        let target = module.join("Sub/Target.h");
        std::fs::write(
            &target,
            format!("#pragma once\n{includes}void Run() {{\n{calls}}}\n"),
        )
        .unwrap();

        let bloom = crate::index::bloom::BloomFilterCache::new();
        let result = analyze_deps(&target, &module, &bloom).unwrap();

        let sib = result
            .uses_local
            .iter()
            .find(|d| d.path.ends_with("Sib.h"))
            .expect("the sibling include must be a local dependency");
        assert!(
            sib.symbols.iter().any(|s| s == "SibFunc"),
            "the 9th import must still be opened for callee resolution — got {:?}. \
             A capped import list drops it and the symbol is reported as external.",
            sib.symbols
        );
    }

    /// Every resolvable include must be reported, however many there are.
    ///
    /// `resolve_related_files_with_content` caps at 8 for the read-time hint. deps used
    /// that capped view, and because an include that *resolves* is excluded from the
    /// external bucket, everything past the eighth appeared in neither list and vanished
    /// from the report entirely — on leveldb's `db/db_impl.cc` that was 10 real
    /// dependencies, fewer total names than before include-root resolution existed. A C++
    /// translation unit with more than 8 project includes is completely ordinary.
    #[test]
    fn cpp_reports_every_local_include_past_the_suggestion_cap() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::create_dir_all(root.join("lib")).unwrap();

        // 12 project headers — comfortably past MAX_SUGGESTIONS (8).
        let count = 12;
        let mut includes = String::from("#include <vector>\n");
        for i in 0..count {
            std::fs::write(
                root.join(format!("lib/h{i}.h")),
                format!("struct S{i} {{ int V; }};\n"),
            )
            .unwrap();
            includes.push_str(&format!("#include \"lib/h{i}.h\"\n"));
        }
        std::fs::write(
            root.join("lib/user.cc"),
            format!("{includes}void Use() {{}}\n"),
        )
        .unwrap();

        let bloom = crate::index::bloom::BloomFilterCache::new();
        let result = analyze_deps(&root.join("lib/user.cc"), root, &bloom).unwrap();

        assert_eq!(
            result.uses_local.len(),
            count,
            "all {count} local includes must be reported, got {:?}",
            result
                .uses_local
                .iter()
                .map(|d| &d.path)
                .collect::<Vec<_>>()
        );
        // None of them may have leaked into the external bucket either.
        assert_eq!(
            result.uses_external,
            vec!["vector".to_string()],
            "only the system header is external"
        );
    }

    /// The fallback above is C/C++-only. Other languages' relative imports normally do
    /// resolve on disk, so applying it there would reclassify genuine resolution
    /// failures as external dependencies.
    #[test]
    fn unresolvable_relative_import_stays_unreported_for_other_languages() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(
            root.join("app.ts"),
            "import { x } from './missing';\nexport const y = 1;\n",
        )
        .unwrap();

        let bloom = crate::index::bloom::BloomFilterCache::new();
        let result = analyze_deps(&root.join("app.ts"), root, &bloom).unwrap();
        assert!(
            !result.uses_external.iter().any(|e| e.contains("missing")),
            "a TS relative import must not be reclassified as external, got {:?}",
            result.uses_external
        );
    }

    /// End-to-end: a C++ header reported `0 local, 0 external, 0 dependents`. Local
    /// and dependent counts depend on the outline naming the class and its members,
    /// and the dependent is reached through a qualified static call.
    #[test]
    fn cpp_header_reports_local_external_and_dependents() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::write(
            root.join("Shared.h"),
            "struct SharedThing { int Value; };\n",
        )
        .unwrap();
        // A header in the shape a code-generating C++ framework produces: an export
        // macro in the class head, call-shaped annotation macros, and static helpers
        // reached only through qualified calls. Every identifier is invented — the
        // point is the AST shape, not any particular framework.
        std::fs::write(
            root.join("LogUtil.h"),
            "#pragma once\n\
             #include <vector>\n\
             #include \"Shared.h\"\n\
             \n\
             ANNOTATE()\n\
             class MYLIB_API LogUtilities final : public HelperLibraryBase\n\
             {\n\
             \tBODY_MACRO()\n\
             public:\n\
             \tstatic const char* GetLogNameSafe(const Actor* Subject);\n\
             };\n",
        )
        .unwrap();
        std::fs::write(
            root.join("Consumer.cpp"),
            "#include \"LogUtil.h\"\n\
             void DoSomething(const Actor* Subject)\n\
             {\n\
             \tLogUtilities::GetLogNameSafe(Subject);\n\
             }\n",
        )
        .unwrap();

        let bloom = crate::index::bloom::BloomFilterCache::new();
        let result = analyze_deps(&root.join("LogUtil.h"), root, &bloom).unwrap();

        assert!(
            result
                .uses_local
                .iter()
                .any(|d| d.path.ends_with("Shared.h")),
            "expected Shared.h as a local dep, got {:?}",
            result
                .uses_local
                .iter()
                .map(|d| &d.path)
                .collect::<Vec<_>>()
        );
        assert!(
            result.uses_external.iter().any(|e| e == "vector"),
            "expected `vector` as an external dep, got {:?}",
            result.uses_external
        );
        assert!(
            result
                .used_by
                .iter()
                .any(|d| d.path.ends_with("Consumer.cpp")),
            "expected Consumer.cpp as a dependent, got {:?}",
            result.used_by.iter().map(|d| &d.path).collect::<Vec<_>>()
        );
    }

    #[test]
    fn go_stdlib_fmt_is_stdlib() {
        assert!(is_stdlib("fmt", crate::types::Lang::Go));
    }

    #[test]
    fn go_stdlib_fmtlib_is_not_stdlib() {
        // "fmtlib" is not a Go stdlib package—previously matched via starts_with("fmt")
        assert!(!is_stdlib("fmtlib", crate::types::Lang::Go));
    }

    #[test]
    fn go_stdlib_fmtutil_is_not_stdlib() {
        assert!(!is_stdlib("fmtutil", crate::types::Lang::Go));
    }

    #[test]
    fn go_stdlib_multi_segment_paths_are_stdlib() {
        // Regression: multi-segment stdlib imports (single-line form
        // `import "net/http"`) must classify as stdlib via their root segment.
        // The exact-match allowlist briefly regressed these to "external".
        for path in [
            "net/http",
            "encoding/json",
            "path/filepath",
            "crypto/sha256",
            "text/template",
            "container/list",
            "database/sql",
        ] {
            assert!(
                is_stdlib(path, crate::types::Lang::Go),
                "{path} should be classified as Go stdlib"
            );
        }
    }

    #[test]
    fn go_local_multi_segment_path_is_not_stdlib() {
        // A local/third-party multi-segment package whose root isn't stdlib.
        assert!(!is_stdlib("mypackage/sub", crate::types::Lang::Go));
    }

    #[test]
    fn go_local_package_without_dot_is_not_stdlib() {
        // A local package like "mypackage" has no dot but is NOT stdlib—
        // the old !source.contains('.') rule wrongly classified it as stdlib.
        assert!(!is_stdlib("mypackage", crate::types::Lang::Go));
    }

    #[test]
    fn go_external_dotted_path_is_not_stdlib() {
        assert!(!is_stdlib(
            "github.com/gin-gonic/gin",
            crate::types::Lang::Go
        ));
    }

    #[test]
    fn go_stdlib_cmp_and_maps_are_stdlib() {
        // Go 1.21+ added `cmp` and `maps` to the standard library.
        assert!(is_stdlib("cmp", crate::types::Lang::Go));
        assert!(is_stdlib("maps", crate::types::Lang::Go));
    }
}
