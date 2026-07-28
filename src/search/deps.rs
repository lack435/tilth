//! File-level dependency analysis: what a file imports and what imports it.
//! Used by `tilth_deps` for blast-radius checks before breaking changes.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::TilthError;
use crate::lang::detect_file_type;
use crate::lang::outline::{extract_import_source, get_outline_entries};
use crate::read::imports::{is_external, is_import_line, resolve_related_files_with_content};
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

    // Local deps via callee resolution
    let callee_names = extract_callee_names(&content, lang, None);
    let resolved = resolve_callees(&callee_names, path, &content, bloom);

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
    let import_files = resolve_related_files_with_content(path, &content);
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
                crate::read::imports::resolve_import_to_file(dir, &source, lang).is_none()
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
        let raw_matches = find_callers_batch(
            &symbols_set,
            scope,
            bloom,
            None,
            crate::search::callers::BATCH_EARLY_QUIT,
        )?;

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
