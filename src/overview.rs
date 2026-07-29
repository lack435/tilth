//! Project fingerprint for MCP initialization.
//! Gives agents instant orientation without a tool call.

use std::collections::HashMap;
use std::fmt::Write;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use serde::Deserialize;

use crate::lang::detect_file_type;
use crate::read::imports::is_import_line;
use crate::search::SKIP_DIRS;
use crate::types::{FileType, Lang};

/// Compute a project fingerprint for MCP initialization.
/// Must be fast (<250ms) — runs synchronously in the initialize handler.
/// Returns empty string on any failure (no error propagation).
#[must_use]
pub fn fingerprint(root: &Path) -> String {
    let start = Instant::now();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| fingerprint_inner(root)));
    let elapsed = start.elapsed();
    if elapsed.as_millis() > 250 {
        eprintln!(
            "[tilth] fingerprint took {}ms (>250ms budget)",
            elapsed.as_millis()
        );
    }
    result.unwrap_or_default()
}

/// How many module directories the `dirs:` line lists.
///
/// The *only* cap on that list. The header reports how many qualified, so anything beyond
/// this is accounted for by the `+N more` suffix rather than silently dropped.
const MAX_LISTED_DIRS: usize = 10;

/// How many dependencies the `deps:` line lists. See `cap_deps`.
const MAX_LISTED_DEPS: usize = 10;

fn fingerprint_inner(root: &Path) -> String {
    let mut lines: Vec<String> = Vec::new();

    // Walk files (depth 2) — collect language counts, modules, entry points
    let walk = walk_files(root);

    // Determine primary language.
    //
    // `max_by_key` returns the *last* maximum it sees, and iterating a `HashMap` visits
    // entries in an order that `RandomState` reseeds every process. So a tie between two
    // languages picked a different primary language run to run — and that choice cascades
    // into the displayed language, the file count, which directories qualify as modules,
    // and whether `hot_files` runs at all. Tie-break on the display name so the key is a
    // total order; `Lang` is deliberately not `Ord`, and a stable string does the job.
    //
    // The display name alone is not a *good* order, only a total one: `.ts` and `.tsx` at
    // equal counts resolved to "TSX" because `'S' < 'y'`, so a React codebase with a 50/50
    // split was called a TSX project. `lang_tiebreak_rank` states the preference explicitly
    // and the name still breaks ties within a rank, so the key stays total.
    let primary_lang = walk
        .lang_counts
        .iter()
        .max_by_key(|(lang, count)| {
            (
                **count,
                std::cmp::Reverse(lang_tiebreak_rank(**lang)),
                std::cmp::Reverse(lang_display_name(**lang)),
            )
        })
        .map(|(lang, _)| *lang);

    let lang_name = primary_lang.map_or("Unknown", lang_display_name);

    // The file count, and the noun that says what it counted.
    //
    // This was `lang_counts[primary_lang]` under a label reading "source files", so a tied
    // 4-Rust/4-Python tree of 8 files reported "4 source files" — while the fallback for a
    // tree with no detected language *summed* instead, so the number meant one of two
    // different things depending on the tree.
    //
    // Naming the language is the fix rather than summing, because the header's *other*
    // number is primary-language-scoped too: a directory qualifies as a module on its count
    // of primary-language files (see below). Summing here would have made one sentence
    // report two differently-measured populations — `Rust project — 95 source files, 8
    // directories`, where the 8 directories account for 39 of the 95. One population, named.
    //
    // The old fallback is now provably dead rather than merely unused: `primary_lang` is
    // `None` only when `max_by_key` saw an empty map, and `lang_counts` holds no zero
    // entries, so no primary language implies no code files at all.
    let (file_count, file_noun) = match primary_lang {
        Some(l) => (walk.lang_counts.get(&l).copied().unwrap_or(0), lang_name),
        None => (0, "source"),
    };

    // Modules: dirs with >=2 files of the primary language, with common prefix stripped.
    // Keys in module_lang_counts may be "dir" or "dir/subdir" (for deeply nested projects).
    //
    // Returns the listed names *and* how many qualified before the cap, because the header
    // counts the second and the `dirs:` line shows the first.
    let (modules, qualifying_dirs): (Vec<String>, usize) = {
        // Collect dirs with >=2 primary-language files, sorted by file count descending
        let mut mods: Vec<(String, usize)> = walk
            .module_lang_counts
            .iter()
            .filter_map(|(name, lang_map)| {
                let count = primary_lang
                    .and_then(|l| lang_map.get(&l))
                    .copied()
                    .unwrap_or(0);
                if count >= 2 {
                    Some((name.clone(), count))
                } else {
                    None
                }
            })
            .collect();
        // Most files first, then by name.
        //
        // The name tie-break here is **defensive, not load-bearing** — worth saying plainly
        // because mutation testing proved it: reverting this sort alone changes nothing,
        // since the second sort below re-establishes the same total order before the
        // truncation that actually matters. It is kept because everything between the two
        // sorts (`common_dir_prefix`, the non-source `retain`) happens to be
        // order-independent today and need not stay that way; a future step that reads
        // `mods` in order would otherwise silently reintroduce the bug here.
        mods.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        // If all modules (or at least most) share a common top-level prefix
        // (e.g., all are "src/..."), strip it so we display short names
        // ("diff/" not "src/diff/"). Also exclude the bare prefix entry itself.
        if mods.len() >= 2 {
            let prefix = common_dir_prefix(&mods);
            if !prefix.is_empty() {
                // The prefix without trailing slash (e.g., "src")
                let prefix_bare = prefix.trim_end_matches('/');
                mods = mods
                    .into_iter()
                    .filter_map(|(name, count)| {
                        if name == prefix_bare {
                            // Drop the bare prefix itself (it's the container, not a module)
                            None
                        } else if let Some(stripped) = name.strip_prefix(&prefix) {
                            let s = stripped.trim_start_matches('/');
                            if s.is_empty() {
                                None
                            } else {
                                Some((s.to_string(), count))
                            }
                        } else {
                            Some((name, count))
                        }
                    })
                    .collect();
            }
        }
        // Filter out well-known non-source directories
        let non_source = [
            "test",
            "tests",
            "__tests__",
            "spec",
            "specs",
            "doc",
            "docs",
            "docs_src",
            "documentation",
            "example",
            "examples",
            "sample",
            "samples",
            "script",
            "scripts",
            "tools",
            "fixtures",
            "benchmark",
            "benchmarks",
            "bench",
            ".github",
            ".vscode",
            ".idea",
            "vendor",
            "node_modules",
            "target",
            "dist",
            "build",
        ];
        mods.retain(|(name, _)| {
            let lower = name.to_lowercase();
            // Check if ANY path component is a non-source dir
            !lower.split('/').any(|part| non_source.contains(&part))
        });
        // Sort by file count descending, then by name, truncate to 10, extract names.
        //
        // **This is the load-bearing one.** `mods` comes from iterating
        // `module_lang_counts`, a `HashMap`; `sort_by_key` is stable, so directories with
        // *equal* file counts kept hash-iteration order, and `truncate` then chose
        // membership from it — `RandomState` reseeds per process, so identical runs listed
        // different directories. Measured on a large tree: six runs, six distinct `dirs:`
        // lines, differing in *which* directory appeared, not merely its position.
        //
        // The name tie-break makes the key a total order, so nothing can reorder equal
        // counts. Note the shape rather than the specific sort: a truncation applied to a
        // collection whose order was never pinned is the same defect fixed in `callers`,
        // `symbol`/`content`, and `glob`.
        mods.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        // Count what qualified *before* the cap. `dir_count` used to be taken after the
        // truncation, so a tree with 15 qualifying directories announced "10 directories".
        // And there were two independent caps — this one and a second `truncate(10)` on the
        // way to the `dirs:` line — so raising one alone would have made the header and the
        // list it introduces silently disagree. One cap, applied here.
        let qualifying = mods.len();
        mods.truncate(MAX_LISTED_DIRS);
        (mods.into_iter().map(|(name, _)| name).collect(), qualifying)
    };

    // Header line. Both nouns are pluralised — this said "1 directories".
    lines.push(format!(
        "[tilth] {lang_name} project — {file_count} {file_noun} file{}, {qualifying_dirs} director{}",
        if file_count == 1 { "" } else { "s" },
        if qualifying_dirs == 1 { "y" } else { "ies" }
    ));

    // Directories (capped, sorted by file count descending). The count above is the true
    // one, so say plainly when this list is only part of it.
    if !modules.is_empty() {
        let display: Vec<String> = modules.iter().map(|m| format!("{m}/")).collect();
        let mut dirs_line = format!("  dirs: {}", display.join(" "));
        let hidden = qualifying_dirs.saturating_sub(modules.len());
        if hidden > 0 {
            write!(dirs_line, " +{hidden} more").unwrap();
        }
        lines.push(dirs_line);
    }

    // Manifest — name, version, deps
    if let Some(manifest) = find_manifest(root) {
        if let Some(info) = parse_manifest(root, &manifest) {
            // Deps line. Same disclosure as `dirs:` above — this line drops more than it
            // shows on a real manifest (tilth: 10 of 40, alphabetical, so every
            // `tree-sitter-*` grammar falls off the end), and saying nothing about that in
            // an orientation payload is the defect this change exists to fix.
            if !info.deps.is_empty() {
                let dep_str = info.deps.join(", ");
                let mut deps_line = format!("  deps: {dep_str}");
                let hidden = info.dep_total.saturating_sub(info.deps.len());
                if hidden > 0 {
                    write!(deps_line, " +{hidden} more").unwrap();
                }
                lines.push(deps_line);
            }

            // Hot files (only for projects with local imports)
            if let Some(hot) = hot_files(root, &walk, primary_lang) {
                lines.push(format!("  hot (× = importers): {hot}"));
            }

            // Git context
            if let Some(git) = git_context(root) {
                lines.push(format!("  git: {git}"));
            }

            // Test style
            if let Some(tests) = test_style(root, &walk, primary_lang) {
                lines.push(format!("  tests: {tests}"));
            }

            // Manifest line
            let mut manifest_line = format!("  manifest: {manifest}");
            if let Some(name) = &info.name {
                write!(manifest_line, " ({name}").unwrap();
                if let Some(version) = &info.version {
                    write!(manifest_line, " v{version}").unwrap();
                }
                manifest_line.push(')');
            }
            lines.push(manifest_line);
        }
    } else {
        // No manifest — still show hot, git, tests
        if let Some(hot) = hot_files(root, &walk, primary_lang) {
            lines.push(format!("  hot (× = importers): {hot}"));
        }
        if let Some(git) = git_context(root) {
            lines.push(format!("  git: {git}"));
        }
        if let Some(tests) = test_style(root, &walk, primary_lang) {
            lines.push(format!("  tests: {tests}"));
        }
    }

    lines.join("\n")
}

// ---------------------------------------------------------------------------
// Common dir prefix helper
// ---------------------------------------------------------------------------

/// If all module names (which may be "a/b" style) share the same first path
/// component, return that component followed by "/". Otherwise return "".
fn common_dir_prefix(mods: &[(String, usize)]) -> String {
    if mods.is_empty() {
        return String::new();
    }
    // Extract the first path component from each name
    let first_components: Vec<&str> = mods
        .iter()
        .map(|(n, _)| n.split('/').next().unwrap_or(n))
        .collect();
    let first = first_components[0];
    if first_components.iter().all(|c| *c == first) && mods.iter().any(|(n, _)| n.contains('/')) {
        // All share the same first component and at least some have a subdir
        format!("{first}/")
    } else {
        String::new()
    }
}

// ---------------------------------------------------------------------------
// Language display
// ---------------------------------------------------------------------------

fn lang_display_name(lang: Lang) -> &'static str {
    match lang {
        Lang::Rust => "Rust",
        Lang::TypeScript => "TypeScript",
        Lang::Tsx => "TSX",
        Lang::JavaScript => "JavaScript",
        Lang::Python => "Python",
        Lang::Go => "Go",
        Lang::Java => "Java",
        Lang::Scala => "Scala",
        Lang::C => "C",
        Lang::Cpp => "C++",
        Lang::Ruby => "Ruby",
        Lang::Php => "PHP",
        Lang::Swift => "Swift",
        Lang::Kotlin => "Kotlin",
        Lang::CSharp => "C#",
        Lang::Elixir => "Elixir",
        Lang::Bash => "Bash",
        Lang::Dockerfile => "Docker",
        Lang::Make => "Make",
    }
}

/// Preference between languages with the same file count. **Lower wins.**
///
/// Deliberately near-flat: it exists only where the alphabetical fallback picks something
/// misleading rather than merely arbitrary. A `.ts`/`.tsx` split is a TypeScript project
/// whether or not it uses React, and `primary_lang` does more than set a label — it also
/// decides which directories qualify as modules and gates `hot_files` — so the choice is
/// worth stating rather than inheriting from a display string.
///
/// Everything at rank 0 falls through to the name tie-break, whose totality is what
/// `lang_display_names_are_unique` pins.
fn lang_tiebreak_rank(lang: Lang) -> u8 {
    match lang {
        Lang::Tsx => 1,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Path rendering
// ---------------------------------------------------------------------------

/// Render a relative path with `/` separators, whatever the platform.
///
/// Paths reach the fingerprint from `strip_prefix(root)`, which is backslash-separated on
/// Windows. That made `has_py_tests`' `"/test_"` probe dead there, and rendered the `hot`
/// line as `src\types.rs` — so the same tree fingerprinted differently on different
/// platforms, in one case losing a true fact.
///
/// Only rewrites where the platform separator *is* a backslash: on Unix a backslash is a
/// legal filename character, and rewriting it there would corrupt real paths. `MAIN_SEPARATOR`
/// is a const, so the branch compiles away.
fn rel_display(path: &Path) -> String {
    let s = path.to_string_lossy();
    if std::path::MAIN_SEPARATOR == '\\' {
        s.replace('\\', "/")
    } else {
        s.into_owned()
    }
}

// ---------------------------------------------------------------------------
// File walk (depth 2)
// ---------------------------------------------------------------------------

struct WalkResult {
    lang_counts: HashMap<Lang, usize>,
    /// Top-level dirs → per-language file counts
    module_lang_counts: HashMap<String, HashMap<Lang, usize>>,
    /// Code files found: (path relative to root, size in bytes)
    code_files: Vec<(String, u64)>,
    /// Whether specific test dirs exist
    has_tests_dir: bool,
    has_test_dir: bool,
    has_dunder_tests: bool,
    has_spec_dir: bool,
}

fn walk_files(root: &Path) -> WalkResult {
    let mut lang_counts: HashMap<Lang, usize> = HashMap::new();
    let mut module_lang_counts: HashMap<String, HashMap<Lang, usize>> = HashMap::new();
    let mut code_files: Vec<(String, u64)> = Vec::new();
    let mut has_tests_dir = false;
    let mut has_test_dir = false;
    let mut has_dunder_tests = false;
    let mut has_spec_dir = false;

    // Walk depth 0 (root itself)
    walk_dir(
        root,
        root,
        0,
        2,
        &mut lang_counts,
        &mut module_lang_counts,
        &mut code_files,
        &mut has_tests_dir,
        &mut has_test_dir,
        &mut has_dunder_tests,
        &mut has_spec_dir,
    );

    WalkResult {
        lang_counts,
        module_lang_counts,
        code_files,
        has_tests_dir,
        has_test_dir,
        has_dunder_tests,
        has_spec_dir,
    }
}

#[allow(clippy::too_many_arguments)]
fn walk_dir(
    dir: &Path,
    root: &Path,
    depth: usize,
    max_depth: usize,
    lang_counts: &mut HashMap<Lang, usize>,
    module_lang_counts: &mut HashMap<String, HashMap<Lang, usize>>,
    code_files: &mut Vec<(String, u64)>,
    has_tests_dir: &mut bool,
    has_test_dir: &mut bool,
    has_dunder_tests: &mut bool,
    has_spec_dir: &mut bool,
) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };

        let Ok(ft) = entry.file_type() else {
            continue;
        };

        if ft.is_dir() {
            if SKIP_DIRS.contains(&name) {
                continue;
            }

            // Track test directories at any depth
            match name {
                "tests" => *has_tests_dir = true,
                "test" => *has_test_dir = true,
                "__tests__" => *has_dunder_tests = true,
                "spec" => *has_spec_dir = true,
                _ => {}
            }

            if depth < max_depth {
                walk_dir(
                    &path,
                    root,
                    depth + 1,
                    max_depth,
                    lang_counts,
                    module_lang_counts,
                    code_files,
                    has_tests_dir,
                    has_test_dir,
                    has_dunder_tests,
                    has_spec_dir,
                );
            }
        } else if ft.is_file() {
            if let FileType::Code(lang) = detect_file_type(&path) {
                *lang_counts.entry(lang).or_insert(0) += 1;

                // Track size for hot files
                let size = entry.metadata().map_or(0, |m| m.len());
                if let Ok(rel) = path.strip_prefix(root) {
                    // Forward slashes, so every consumer of `code_files` — the `test_*.py`
                    // probe, the `hot` line — behaves identically on Windows and Unix.
                    let rel_str = rel_display(rel);

                    code_files.push((rel_str, size));

                    // Track module — use up to 2 path components as the key,
                    // but only for files nested at least one level deep.
                    // e.g. src/diff/mod.rs → key "src/diff", lib.rs → skipped
                    {
                        let mut comps = rel.components();
                        if let Some(c1) = comps.next() {
                            let remaining: Vec<_> = comps.collect();
                            if !remaining.is_empty() {
                                let key = if remaining.len() >= 2 {
                                    // File is at depth 3+: use first two components
                                    format!(
                                        "{}/{}",
                                        c1.as_os_str().to_string_lossy(),
                                        remaining[0].as_os_str().to_string_lossy()
                                    )
                                } else {
                                    // File is at depth 2: use first component only
                                    c1.as_os_str().to_string_lossy().to_string()
                                };
                                *module_lang_counts
                                    .entry(key)
                                    .or_default()
                                    .entry(lang)
                                    .or_insert(0) += 1;
                            }
                        }
                    }
                }
            }

            // Check test file patterns
            if name.contains(".test.") || name.contains(".spec.") {
                // These contribute to test style but we detect in test_style()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Manifest parsing
// ---------------------------------------------------------------------------

fn find_manifest(root: &Path) -> Option<String> {
    const MANIFESTS: &[&str] = &["Cargo.toml", "package.json", "go.mod", "pyproject.toml"];
    for m in MANIFESTS {
        if root.join(m).exists() {
            return Some((*m).to_string());
        }
    }
    None
}

struct ManifestInfo {
    name: Option<String>,
    version: Option<String>,
    deps: Vec<String>,
    /// How many dependencies the manifest declared, before the cap. The renderer needs this
    /// to say what it dropped.
    dep_total: usize,
}

/// Sort, record the true count, then cap.
///
/// One place for all four manifest formats, so the count and the cap cannot drift apart —
/// which is exactly how `dir_count` came to disagree with the `dirs:` line it introduces.
fn cap_deps(mut deps: Vec<String>) -> (Vec<String>, usize) {
    deps.sort();
    let total = deps.len();
    deps.truncate(MAX_LISTED_DEPS);
    (deps, total)
}

fn parse_manifest(root: &Path, manifest: &str) -> Option<ManifestInfo> {
    match manifest {
        "Cargo.toml" => parse_cargo_toml(root),
        "package.json" => parse_package_json(root),
        "go.mod" => parse_go_mod(root),
        "pyproject.toml" => parse_pyproject_toml(root),
        _ => None,
    }
}

fn parse_cargo_toml(root: &Path) -> Option<ManifestInfo> {
    #[derive(Deserialize)]
    struct CargoToml {
        package: Option<Package>,
        dependencies: Option<toml::Table>,
    }
    #[derive(Deserialize)]
    struct Package {
        name: Option<String>,
        version: Option<String>,
    }

    let content = fs::read_to_string(root.join("Cargo.toml")).ok()?;
    let parsed: CargoToml = toml::from_str(&content).ok()?;
    let (name, version) = parsed.package.map_or((None, None), |p| (p.name, p.version));
    let (deps, dep_total) = cap_deps(
        parsed
            .dependencies
            .map(|d| d.into_iter().map(|(k, _)| k).collect())
            .unwrap_or_default(),
    );
    Some(ManifestInfo {
        name,
        version,
        deps,
        dep_total,
    })
}

fn parse_package_json(root: &Path) -> Option<ManifestInfo> {
    #[derive(Deserialize)]
    struct PackageJson {
        name: Option<String>,
        version: Option<String>,
        dependencies: Option<serde_json::Map<String, serde_json::Value>>,
    }

    let content = fs::read_to_string(root.join("package.json")).ok()?;
    let parsed: PackageJson = serde_json::from_str(&content).ok()?;
    let (deps, dep_total) = cap_deps(
        parsed
            .dependencies
            .map(|d| d.into_iter().map(|(k, _)| k).collect())
            .unwrap_or_default(),
    );
    Some(ManifestInfo {
        name: parsed.name,
        version: parsed.version,
        deps,
        dep_total,
    })
}

fn parse_go_mod(root: &Path) -> Option<ManifestInfo> {
    let content = fs::read_to_string(root.join("go.mod")).ok()?;
    let mut name = None;
    let mut deps: Vec<String> = Vec::new();
    let mut in_require = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("module ") {
            name = Some(rest.trim().to_string());
        }
        if trimmed == "require (" {
            in_require = true;
            continue;
        }
        if trimmed == ")" {
            in_require = false;
            continue;
        }
        if in_require {
            // e.g. "github.com/gin-gonic/gin v1.9.0"
            if let Some(dep) = trimmed.split_whitespace().next() {
                if !dep.starts_with("//") {
                    // Use short name (last segment of module path)
                    let short = dep.rsplit('/').next().unwrap_or(dep);
                    deps.push(short.to_string());
                }
            }
        }
    }

    let (deps, dep_total) = cap_deps(deps);

    Some(ManifestInfo {
        name,
        version: None,
        deps,
        dep_total,
    })
}

fn parse_pyproject_toml(root: &Path) -> Option<ManifestInfo> {
    #[derive(Default, Deserialize)]
    struct PyProject {
        project: Option<Project>,
    }
    #[derive(Default, Deserialize)]
    struct Project {
        name: Option<String>,
        version: Option<String>,
        dependencies: Option<Vec<String>>,
    }

    let content = fs::read_to_string(root.join("pyproject.toml")).ok()?;
    let parsed: PyProject = toml::from_str(&content).ok()?;
    let project = parsed.project.unwrap_or_default();
    let (deps, dep_total) = cap_deps(
        project
            .dependencies
            .unwrap_or_default()
            .into_iter()
            .filter_map(|spec| {
                let bare = spec
                    .split(&['>', '<', '=', '~', '!', ';', '[', ' '][..])
                    .next()?
                    .trim();
                (!bare.is_empty()).then(|| bare.to_string())
            })
            .collect(),
    );
    Some(ManifestInfo {
        name: project.name,
        version: project.version,
        deps,
        dep_total,
    })
}

// ---------------------------------------------------------------------------
// Git context
// ---------------------------------------------------------------------------

/// Run a git command with a 200ms timeout. Returns None if it fails or times out.
/// Best-effort: every git failure (spawn error, non-zero exit, timeout, I/O error)
/// is intentionally swallowed into None — git context is a cosmetic fingerprint
/// and must never break the primary read/search path.
///
/// The 200ms deadline below is a **deliberate** residual source of variation, and is not
/// the same trade as the wall-clock budget removed from `hot_files`. That one decided which
/// of a known list of files contributed, so identical runs disagreed about work they were
/// both perfectly able to do. This one bounds an external process that may hang, and it is
/// all-or-nothing: `git` either answers or the line is omitted. Under heavy load a slow
/// `git` can therefore still drop the `git:` line from an otherwise identical fingerprint.
/// Removing the deadline would trade that for hanging the MCP handshake, which is worse.
fn git_output(root: &Path, args: &[&str]) -> Option<String> {
    let mut child = Command::new("git")
        .args(args)
        .current_dir(root)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;

    let deadline = Instant::now() + Duration::from_millis(200);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    return None;
                }
                let out = child.stdout.take()?;
                let s = std::io::read_to_string(out).ok()?;
                let trimmed = s.trim().to_string();
                return if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed)
                };
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(_) => return None,
        }
    }
}

fn git_context(root: &Path) -> Option<String> {
    let branch = git_output(root, &["branch", "--show-current"])
        .or_else(|| git_output(root, &["rev-parse", "--short", "HEAD"]))?;

    let dirty_count = git_output(root, &["status", "--porcelain"]).map_or(0, |s| s.lines().count());

    Some(format!("branch {branch}, {}", dirty_summary(dirty_count)))
}

/// "1 uncommitted files" — the same defect as the header's "1 directories", one line away,
/// so corrected with it rather than left as the only ungrammatical count in the payload.
///
/// Split out from `git_context` purely so it is testable: `git_context` shells out to `git`,
/// and this is the one behaviour change here that would otherwise have no coverage.
fn dirty_summary(dirty_count: usize) -> String {
    match dirty_count {
        0 => "clean".to_string(),
        1 => "1 uncommitted file".to_string(),
        n => format!("{n} uncommitted files"),
    }
}

// ---------------------------------------------------------------------------
// Test style detection
// ---------------------------------------------------------------------------

fn test_style(root: &Path, walk: &WalkResult, primary_lang: Option<Lang>) -> Option<String> {
    let mut styles: Vec<String> = Vec::new();

    // Directory-based test detection
    if walk.has_tests_dir {
        styles.push("tests/".to_string());
    }
    if walk.has_test_dir {
        styles.push("test/".to_string());
    }
    if walk.has_dunder_tests {
        styles.push("__tests__/".to_string());
    }
    if walk.has_spec_dir {
        styles.push("spec/".to_string());
    }

    // File pattern detection
    let has_test_files = walk
        .code_files
        .iter()
        .any(|(path, _)| path.contains(".test.") || path.contains(".spec."));
    let has_go_tests = walk
        .code_files
        .iter()
        .any(|(path, _)| path.ends_with("_test.go"));
    let has_py_tests = walk
        .code_files
        .iter()
        .any(|(path, _)| path.starts_with("test_") || path.contains("/test_"));

    if has_test_files && !walk.has_dunder_tests {
        styles.push("*.test/spec files".to_string());
    }
    if has_go_tests {
        styles.push("_test.go".to_string());
    }
    if has_py_tests {
        styles.push("test_*.py".to_string());
    }

    // Rust in-source test detection.
    //
    // This sampled `take(5)` over `code_files`, which is in `fs::read_dir` order — so it
    // decided *content*, not merely ordering: whether the `tests:` line mentions in-source
    // tests at all. Adding five unrelated source files that happened to be visited first
    // silently deleted a true fact from the fingerprint. Verified on two trees differing
    // only in which file held the test module.
    //
    // Sorting alone fixes the determinism but not the wrongness — with a five-file sample,
    // five earlier-sorting files still hide a real test module, just reproducibly. So the
    // candidates are ordered smallest-first (with a path tie-break for a total order, since
    // same-size files are common) and the sample is widened to `MAX_TEST_STYLE_PROBES`.
    // `any` short-circuits, so a crate that does use in-source tests almost always stops
    // after one read; the budget only binds on crates that do not, where it caps the work at
    // a fixed number of the smallest files rather than the whole tree.
    if primary_lang == Some(Lang::Rust) {
        let mut rs_files: Vec<&(String, u64)> = walk
            .code_files
            .iter()
            .filter(|(path, _)| {
                Path::new(path)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("rs"))
            })
            .collect();
        rs_files.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        let has_cfg_test = rs_files
            .iter()
            .take(MAX_TEST_STYLE_PROBES)
            .any(|(path, _)| {
                let full = root.join(path);
                fs::read_to_string(&full)
                    .ok()
                    .is_some_and(|content| content.contains("#[cfg(test)]"))
            });
        if has_cfg_test {
            styles.push("in-source #[cfg(test)]".to_string());
        }
    }

    if styles.is_empty() {
        None
    } else {
        Some(styles.join(", "))
    }
}

// ---------------------------------------------------------------------------
// Hot files — most imported local files
// ---------------------------------------------------------------------------

/// How many Rust files `test_style` reads looking for an in-source `#[cfg(test)]`.
///
/// Bounds the work when a crate has none — `any` short-circuits on the first hit, so a crate
/// that does use them stops almost immediately. Candidates are taken smallest-first, so this
/// is a bounded number of cheap reads rather than a bounded number of arbitrary ones.
const MAX_TEST_STYLE_PROBES: usize = 50;

/// Work budget for the `hot_files` import scan, counted in import lines. See the note at
/// the loop below for why the budget is in import lines rather than files or bytes, and for
/// the measurement that set it.
const MAX_IMPORT_LINES: usize = 500;

fn hot_files(root: &Path, walk: &WalkResult, primary_lang: Option<Lang>) -> Option<String> {
    let lang = primary_lang?; // require a detected language

    // Sort by size (smallest first), then by path, and take the first 100.
    //
    // The path tie-break matters because same-size files are common — a stable sort on
    // size alone left them in `code_files` order, i.e. `fs::read_dir` order, and
    // `truncate(100)` then picked a set that depends on it. `read_dir` order is at least
    // stable for a fixed tree on one filesystem, so this was the milder of the two
    // problems here, but it is free to remove and it is not guaranteed by the API.
    let mut files: Vec<&(String, u64)> = walk.code_files.iter().collect();
    files.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    files.truncate(100);

    // Use resolve_related_files to get real file paths for imports.
    // Count how many files import each target path.
    let mut path_counts: HashMap<std::path::PathBuf, usize> = HashMap::new();
    // Also collect all import source lines for symbol extraction later
    let mut all_import_sources: Vec<String> = Vec::new();

    // Budget the scan by *work*, not by wall clock.
    //
    // There used to be a `start.elapsed() > 100ms` break here. A time budget decides which
    // files contribute, so under load or on a cold cache a different prefix was processed
    // and the reported hot files changed — the one bound shape that cannot be made
    // deterministic. `files` above is now a total-ordered prefix, which is what makes a
    // count-based budget over it deterministic: the same tree always yields the same
    // candidates in the same order, so stopping after a fixed amount of work always stops
    // in the same place.
    //
    // The budget is in import *lines*, because that is the actual cost driver.
    // `resolve_related_files_with_content` is uncached and, for an unresolvable C/C++
    // include, probes up to 16 candidate paths — so cost scales with imports per file, not
    // with file count or with file size. Measured on a 100-file C++ tree of 40 unresolvable
    // includes each, whole-`fingerprint` wall time:
    //
    //   no budget at all   338-377ms warm, 705ms cold — `>250ms budget` warning every run
    //   MAX_IMPORT_LINES   112-180ms, no warning
    //
    // Those figures were taken when the probe count was 13, before #17 added hop 0 to the
    // include-root walk (one `dir/…` plus one per `CONVENTIONAL_INCLUDE_ROOTS` name). Treat
    // them as a lower bound: the shape of the argument is unchanged, but a file four or more
    // directories below the containment root now costs ~23% more probes than measured.
    // Shallower files are unaffected — the walk stops at the root either way.
    //
    // 100 files of 41 lines is a *small* tree, and unbudgeted it breaks the 250ms soft
    // budget `fingerprint` sets for itself by well over a factor of two.
    //
    // No claim here that the removed wall-clock cutoff "cost nothing". An earlier version of
    // this comment asserted the scan measured well under 100ms even on a large tree; that
    // was never measured, and it was wrong. The tree that motivated #25 is C#, where
    // `is_import_line` returns false for every line, so it never exercised this path at all.
    let mut import_budget = MAX_IMPORT_LINES;

    for (rel_path, _) in &files {
        if import_budget == 0 {
            break;
        }
        let full = root.join(rel_path);
        let Ok(content) = fs::read_to_string(&full) else {
            continue;
        };

        // Charge this file to the budget before resolving, so the expensive step is what the
        // budget actually governs. `max(1)` makes an import-free file cost something, which
        // bounds the reads as well as the resolutions. A file straddling the limit is
        // processed in full and the next iteration stops, so the real ceiling is
        // `MAX_IMPORT_LINES` plus one file's worth — bounded, and a function of the tree.
        let import_lines = content
            .lines()
            .filter(|line| is_import_line(line, lang))
            .count();
        import_budget = import_budget.saturating_sub(import_lines.max(1));

        // Resolve imports to actual file paths using the proven import resolver.
        //
        // Deliberately no boundary, unlike the read hint and callee resolution (#15). The
        // obvious candidate is `root`, but `fingerprint` is called as `fingerprint(&cwd)`
        // and every file here is `root.join(rel_path)`, so `root` would contain every
        // candidate and always outrank the `.git` fallback: launched above a checkout the
        // scan could count a hot file in a different repository, launched inside a
        // subdirectory it would stop counting headers above the launch dir. Both change
        // what the initialize fingerprint claims, and #29/#31 went to some trouble to make
        // it say true things. So C/C++ include-root resolution stays `.git`-gated here
        // until someone measures the alternative.
        let resolved =
            crate::read::imports::resolve_related_files_with_content(&full, &content, None);
        for target_path in resolved {
            *path_counts.entry(target_path).or_insert(0) += 1;
        }

        // Collect import source strings for symbol extraction
        for line in content.lines() {
            if is_import_line(line, lang) {
                let source = crate::lang::outline::extract_import_source(line, Some(lang));
                if !source.is_empty() && !crate::read::imports::is_external(&source, lang) {
                    all_import_sources.push(source);
                }
            }
        }
    }

    if path_counts.is_empty() {
        return None;
    }

    // Sort by import count descending, take top 5
    let mut sorted: Vec<(std::path::PathBuf, usize)> = path_counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    sorted.truncate(5);

    if sorted[0].1 < 2 {
        return None;
    }

    // For each hot file, find the most commonly imported symbol by scanning
    // import sources that reference this file's module name.
    let parts: Vec<String> = sorted
        .iter()
        .filter(|(_, count)| *count >= 2)
        .map(|(path, count)| {
            let rel = path.strip_prefix(root).unwrap_or(path);
            let rel_str = rel_display(rel);

            // Derive the module name from the file path
            // src/types.rs → "types", src/lang/mod.rs → "lang", src/error.rs → "error"
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            let module_name = if stem == "mod" || stem == "index" || stem == "__init__" {
                path.parent()
                    .and_then(|p| p.file_name())
                    .and_then(|n| n.to_str())
                    .unwrap_or(stem)
            } else {
                stem
            };

            // Count symbols imported from this module across all import sources
            let mut symbol_counts: HashMap<String, usize> = HashMap::new();
            for source in &all_import_sources {
                // Match imports that reference this module
                // e.g., for module "types": "crate::types::OutlineEntry" matches
                let segments: Vec<&str> = source.split("::").collect();
                if let Some(mod_pos) = segments.iter().position(|s| *s == module_name) {
                    // Everything after the module name is a symbol path
                    for &sym in segments.iter().skip(mod_pos + 1) {
                        if !sym.is_empty()
                            && !sym.contains('*')
                            && !sym.contains('{')
                            && sym != "self"
                        {
                            *symbol_counts.entry(sym.to_string()).or_insert(0) += 1;
                        }
                    }
                }
            }

            // Pick the most frequently imported symbol (break ties alphabetically for determinism)
            let top_sym = symbol_counts
                .into_iter()
                .max_by(|(a_sym, a_c), (b_sym, b_c)| a_c.cmp(b_c).then(b_sym.cmp(a_sym)))
                .map(|(sym, _)| sym);

            if let Some(sym) = top_sym {
                format!("{rel_str}({sym}) ×{count}")
            } else {
                format!("{rel_str} ×{count}")
            }
        })
        .collect();

    if parts.is_empty() {
        None
    } else {
        Some(parts.join(", "))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Directories in the tie fixture, and files in each.
    ///
    /// **Every directory has the same count**, which is the whole point: the bug was a
    /// stable `sort_by_key` over `HashMap` iteration order, so it is invisible unless
    /// counts tie. A fixture whose counts are all distinct passes with the bug present.
    ///
    /// More than 10 directories, because `truncate(10)` is what turned an unstable order
    /// into unstable *membership* — with 10 or fewer, every directory is listed whatever
    /// the order, and only the ordering half of the bug shows.
    const TIE_DIRS: usize = 15;
    const TIE_FILES_PER_DIR: usize = 3;

    /// `dirs` directories, each holding `files_per_dir` Rust files, so every module has an
    /// identical primary-language count.
    fn write_dirs_fixture(root: &Path, dirs: usize, files_per_dir: usize) {
        for d in 0..dirs {
            let sub = root.join(format!("m{d:02}"));
            std::fs::create_dir_all(&sub).unwrap();
            for f in 0..files_per_dir {
                std::fs::write(sub.join(format!("f{f}.rs")), "pub fn a() {}\n").unwrap();
            }
        }
    }

    /// A tree of `TIE_DIRS` directories, each holding `TIE_FILES_PER_DIR` Rust files, so
    /// every module has an identical primary-language count.
    fn write_tie_fixture(root: &Path) {
        write_dirs_fixture(root, TIE_DIRS, TIE_FILES_PER_DIR);
    }

    fn header_of(out: &str) -> String {
        out.lines().next().unwrap_or_default().to_string()
    }

    fn line_starting(out: &str, prefix: &str) -> String {
        out.lines()
            .find(|l| l.trim_start().starts_with(prefix))
            .unwrap_or_else(|| panic!("no {prefix} line in fingerprint:\n{out}"))
            .to_string()
    }

    /// The MCP `initialize` payload must be the same on every run for an unchanged tree.
    ///
    /// It was not: six runs on a large tree produced six distinct fingerprints, differing
    /// in *which* directory the `dirs:` line named, not merely the order. Cause was a
    /// stable sort over `HashMap` iteration order followed by `truncate(10)`.
    #[test]
    fn fingerprint_is_byte_identical_across_repeated_runs() {
        let dir = tempfile::tempdir().unwrap();
        write_tie_fixture(dir.path());

        let runs: Vec<String> = (0..8).map(|_| fingerprint(dir.path())).collect();

        assert!(
            !runs[0].is_empty(),
            "fixture produced no fingerprint, so this test proves nothing"
        );
        assert!(
            runs.windows(2).all(|w| w[0] == w[1]),
            "fingerprint varied across 8 identical runs:\n{}",
            runs.join("\n---\n")
        );
    }

    /// Stronger than "stable": with every count tied, the ten listed directories must be
    /// the alphabetically first ten. A consistently *wrong* selection would satisfy the
    /// stability test above but not this one.
    #[test]
    fn tied_module_counts_are_broken_by_name_not_hash_order() {
        let dir = tempfile::tempdir().unwrap();
        write_tie_fixture(dir.path());

        let out = fingerprint(dir.path());
        let dirs_line = out
            .lines()
            .find(|l| l.trim_start().starts_with("dirs:"))
            .unwrap_or_else(|| panic!("no dirs: line in fingerprint:\n{out}"));

        // Sorted by (count desc, name asc) then capped → m00..m09.
        for d in 0..MAX_LISTED_DIRS {
            assert!(
                dirs_line.contains(&format!("m{d:02}/")),
                "expected m{d:02}/ within the cap:\n{dirs_line}"
            );
        }
        for d in MAX_LISTED_DIRS..TIE_DIRS {
            assert!(
                !dirs_line.contains(&format!("m{d:02}/")),
                "m{d:02}/ is past the cap and must not be listed:\n{dirs_line}"
            );
        }
    }

    /// The `hot` line must be present and stable.
    ///
    /// Scoped honestly, because the first version of this docstring claimed more than the
    /// fixture delivers. With only five candidates against `truncate(100)` no truncation
    /// occurs, so the candidate sort is **unobservable** here — the rendered order comes
    /// from the later total-ordered sort of `path_counts`. Mutation testing confirms it:
    /// reversing the candidate sort entirely still passes. Covering the candidate tie-break
    /// would need a fixture of >100 same-size importers.
    ///
    /// What this does cover, verified by mutation: the line is produced at all, and is
    /// stable across runs. Reinstating a wall-clock break with a zero budget removes the
    /// line and fails here. A *partial* prefix cutoff is not testable — a test cannot
    /// reliably make the machine slow — and what pins that is the loop no longer reading a
    /// clock, plus `MAX_IMPORT_LINES` bounding the work by a count instead.
    #[test]
    fn hot_files_line_is_stable_with_tied_file_sizes() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"hf\"\n").unwrap();
        std::fs::write(src.join("shared.rs"), "pub struct Thing;\n").unwrap();
        // Single-character names keep these four files exactly the same length.
        for n in ["a", "b", "c", "d"] {
            std::fs::write(
                src.join(format!("{n}.rs")),
                format!("use crate::shared::Thing;\npub fn {n}(_t: Thing) {{}}\n"),
            )
            .unwrap();
        }

        let runs: Vec<String> = (0..8).map(|_| fingerprint(dir.path())).collect();

        assert!(
            runs[0].contains("hot ("),
            "fixture produced no hot line, so this test proves nothing:\n{}",
            runs[0]
        );
        assert!(
            runs.windows(2).all(|w| w[0] == w[1]),
            "hot line varied across 8 identical runs:\n{}",
            runs.join("\n---\n")
        );
    }

    /// Adding unrelated source files must not delete the in-source-tests fact.
    ///
    /// `test_style` sampled the first five `.rs` files in `fs::read_dir` order, so this was
    /// a truncation over an unpinned order that decided *content*: five unrelated files
    /// visited first, and the `tests:` line lost its `#[cfg(test)]` entry. The two trees here
    /// differ only by five files that contain no tests and sort before the one that does.
    #[test]
    fn unrelated_files_do_not_hide_in_source_tests() {
        let render = |with_filler: bool| -> String {
            let dir = tempfile::tempdir().unwrap();
            let src = dir.path().join("src");
            std::fs::create_dir_all(&src).unwrap();
            std::fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"ts\"\n").unwrap();
            // Sorts last by name, and is the largest, so it loses on either ordering.
            std::fs::write(
                src.join("zz_late.rs"),
                "pub fn a() {}\n#[cfg(test)]\nmod tests { #[test] fn t() {} }\n",
            )
            .unwrap();
            if with_filler {
                for n in 0..5 {
                    std::fs::write(src.join(format!("aa{n}.rs")), "pub fn b() {}\n").unwrap();
                }
            }
            fingerprint(dir.path())
        };

        let without = render(false);
        let with = render(true);

        assert!(
            without.contains("#[cfg(test)]"),
            "baseline tree must report in-source tests, or this proves nothing:\n{without}"
        );
        assert!(
            with.contains("#[cfg(test)]"),
            "five unrelated files must not hide a real test module:\n{with}"
        );
    }

    /// `primary_lang`'s tie-break assumes display names are unique — lock that in.
    ///
    /// The key is `(count, Reverse(lang_tiebreak_rank(lang)), Reverse(lang_display_name(lang)))`.
    /// The rank does not carry the guarantee — it is deliberately near-flat, so most pairs
    /// reach the name. It is a total order *only* because no two `Lang` variants share a
    /// display string, which is what makes `max_by_key`'s return-the-last-maximum behaviour
    /// unreachable. Add
    /// `Lang::Hpp => "C++"` or `Lang::Mjs => "JavaScript"` and the key silently stops being
    /// total, restoring hash-order dependence with nothing else failing. This is that
    /// nothing-else.
    #[test]
    fn lang_display_names_are_unique() {
        use std::collections::HashSet;

        // Every variant `detect_file_type` can produce. Kept explicit rather than derived so
        // adding a `Lang` forces a decision here.
        let all = [
            Lang::Rust,
            Lang::TypeScript,
            Lang::Tsx,
            Lang::JavaScript,
            Lang::Python,
            Lang::Go,
            Lang::Java,
            Lang::Scala,
            Lang::C,
            Lang::Cpp,
            Lang::Ruby,
            Lang::Php,
            Lang::CSharp,
            Lang::Swift,
            Lang::Kotlin,
            Lang::Elixir,
            Lang::Bash,
            Lang::Dockerfile,
            Lang::Make,
        ];

        let mut seen: HashSet<&str> = HashSet::new();
        for lang in all {
            let name = lang_display_name(lang);
            assert!(
                seen.insert(name),
                "two Lang variants share the display name {name:?}; \
                 primary_lang's tie-break is no longer a total order"
            );
        }
    }

    /// A tie on *language* counts must not change the primary language.
    ///
    /// `primary_lang` is a `max_by_key` over a `HashMap`, and `max_by_key` returns the last
    /// maximum, so a tie resolved by hash order. That choice cascades: it sets the displayed
    /// language and file count, decides which directories count as modules, and gates
    /// `hot_files` entirely.
    #[test]
    fn tied_language_counts_resolve_deterministically() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("src");
        std::fs::create_dir_all(&sub).unwrap();
        // Equal numbers of Rust and Python files, so `lang_counts` ties.
        for f in 0..4 {
            std::fs::write(sub.join(format!("a{f}.rs")), "pub fn a() {}\n").unwrap();
            std::fs::write(sub.join(format!("b{f}.py")), "def a():\n    pass\n").unwrap();
        }

        let runs: Vec<String> = (0..8).map(|_| fingerprint(dir.path())).collect();
        assert!(
            runs.windows(2).all(|w| w[0] == w[1]),
            "tied language counts produced a varying fingerprint:\n{}",
            runs.join("\n---\n")
        );
        // And the tie must resolve to the alphabetically-first display name, not whichever
        // the hash happened to visit last.
        assert!(
            runs[0].contains("Python project"),
            "expected the name tie-break to pick Python over Rust, got:\n{}",
            runs[0].lines().next().unwrap_or_default()
        );
    }

    /// The header's directory count must be what qualified, not what the list had room for.
    ///
    /// `dir_count` was `modules.len()` read *after* `truncate(10)`, so a 15-directory tree
    /// announced "10 directories" — a wrong number in a payload pushed once per session and
    /// questioned by no one. Two tree sizes, because below the cap the bug is invisible: a
    /// four-directory tree reports correctly with the defect fully present.
    #[test]
    fn header_dir_count_is_taken_before_the_list_is_capped() {
        for dirs in [4_usize, TIE_DIRS] {
            let dir = tempfile::tempdir().unwrap();
            write_dirs_fixture(dir.path(), dirs, TIE_FILES_PER_DIR);

            let out = fingerprint(dir.path());
            let header = header_of(&out);
            assert!(
                header.contains(&format!("{dirs} directories")),
                "header must count all {dirs} qualifying directories, got:\n{header}"
            );

            let dirs_line = line_starting(&out, "dirs:");
            let listed = dirs_line
                .split_whitespace()
                .filter(|t| t.ends_with('/'))
                .count();
            assert_eq!(
                listed,
                dirs.min(MAX_LISTED_DIRS),
                "the dirs: line must show min(qualifying, cap) entries:\n{dirs_line}"
            );

            // Whatever the cap hides must be accounted for, or the header and the list it
            // introduces contradict each other.
            let hidden = dirs.saturating_sub(MAX_LISTED_DIRS);
            if hidden > 0 {
                assert!(
                    dirs_line.contains(&format!("+{hidden} more")),
                    "the {hidden} capped directories must be declared:\n{dirs_line}"
                );
            } else {
                assert!(
                    !dirs_line.contains("more"),
                    "nothing was hidden, so nothing should be declared hidden:\n{dirs_line}"
                );
            }
        }
    }

    /// The file count's label must say which files it counted.
    ///
    /// It was `lang_counts[primary_lang]` under a label reading "source files", so this tied
    /// 4-Rust/4-Python tree of 8 files reported "4 source files" — a number that is neither
    /// the languages' total nor described by its own noun. Naming the language is the fix
    /// rather than summing, because the header's other number (directories) is
    /// primary-language-scoped too, and one sentence should report one population.
    ///
    /// So this pins both halves: reverting to the old label fails on "4 Python files", and
    /// switching to a cross-language sum fails on the same assertion from the other side.
    #[test]
    fn file_count_is_labelled_with_the_language_it_counted() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        for f in 0..4 {
            std::fs::write(src.join(format!("a{f}.rs")), "pub fn a() {}\n").unwrap();
            std::fs::write(src.join(format!("b{f}.py")), "def a():\n    pass\n").unwrap();
        }

        // The tie resolves to Python by the name tie-break, per
        // `tied_language_counts_resolve_deterministically`.
        let header = header_of(&fingerprint(dir.path()));
        assert!(
            header.contains("4 Python files"),
            "the count is the primary language's, so the label must name it:\n{header}"
        );
        assert!(
            !header.contains("source files"),
            "an unqualified 'source files' is the label that was wrong:\n{header}"
        );
    }

    /// With no language detected there is nothing to name, and the count is provably zero:
    /// `primary_lang` is `None` only when `lang_counts` is empty. The old fallback summed
    /// here, which is how one label came to mean two things.
    #[test]
    fn a_tree_with_no_code_falls_back_to_the_generic_noun() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("README.md"), "# nothing to see\n").unwrap();

        let header = header_of(&fingerprint(dir.path()));
        assert!(
            header.contains("0 source files"),
            "no language means no count to attribute:\n{header}"
        );
    }

    /// `[tilth] Rust project — 6 source files, 1 directories` — neither noun pluralised.
    ///
    /// The zero case is checked as well as the one case, because the obvious wrong fix is
    /// `if n <= 1`, which reads correctly for one and breaks "0 directories".
    #[test]
    fn header_pluralises_both_nouns() {
        // One code file at the root: one file, and nothing nested to qualify as a module.
        let flat = tempfile::tempdir().unwrap();
        std::fs::write(flat.path().join("main.rs"), "pub fn a() {}\n").unwrap();
        let header = header_of(&fingerprint(flat.path()));
        assert!(
            header.contains("1 Rust file,"),
            "singular for one file:\n{header}"
        );
        assert!(
            header.contains("0 directories"),
            "plural for zero directories:\n{header}"
        );

        // Exactly one qualifying directory, holding several files.
        let nested = tempfile::tempdir().unwrap();
        write_dirs_fixture(nested.path(), 1, TIE_FILES_PER_DIR);
        let header = header_of(&fingerprint(nested.path()));
        assert!(
            header.contains("1 directory"),
            "singular for one directory:\n{header}"
        );
        assert!(
            header.contains(&format!("{TIE_FILES_PER_DIR} Rust files")),
            "plural for several files:\n{header}"
        );
    }

    /// The `deps:` line drops more than it shows on a real manifest, and said nothing about
    /// it — the same defect as the header's directory count, one line below it.
    ///
    /// On tilth that is 10 of 40, alphabetical, so every `tree-sitter-*` grammar falls off
    /// the end: the truncation deletes the most identifying fact about the project from a
    /// payload whose whole job is orientation.
    #[test]
    fn deps_line_declares_what_the_cap_hid() {
        let over = tempfile::tempdir().unwrap();
        let mut manifest = String::from("[package]\nname = \"d\"\n\n[dependencies]\n");
        let declared = MAX_LISTED_DEPS + 4;
        for n in 0..declared {
            writeln!(manifest, "dep{n:02} = \"1\"").unwrap();
        }
        std::fs::write(over.path().join("Cargo.toml"), &manifest).unwrap();
        std::fs::write(over.path().join("main.rs"), "pub fn a() {}\n").unwrap();

        let deps_line = line_starting(&fingerprint(over.path()), "deps:");
        // Count entries, not occurrences of "dep" — the `deps:` label contains it too.
        let listed = deps_line
            .trim_start()
            .trim_start_matches("deps:")
            .split(", ")
            .count();
        assert_eq!(
            listed, MAX_LISTED_DEPS,
            "the deps: line must show the cap:\n{deps_line}"
        );
        assert!(
            deps_line.contains(&format!("+{} more", declared - MAX_LISTED_DEPS)),
            "and declare the rest:\n{deps_line}"
        );

        // Under the cap, nothing is hidden and nothing should be claimed hidden.
        let under = tempfile::tempdir().unwrap();
        std::fs::write(
            under.path().join("Cargo.toml"),
            "[package]\nname = \"d\"\n\n[dependencies]\nonly = \"1\"\n",
        )
        .unwrap();
        std::fs::write(under.path().join("main.rs"), "pub fn a() {}\n").unwrap();

        let deps_line = line_starting(&fingerprint(under.path()), "deps:");
        assert!(
            !deps_line.contains("more"),
            "nothing was hidden:\n{deps_line}"
        );
    }

    /// `git_context` shells out, so the pluralisation lives in a helper that does not.
    #[test]
    fn dirty_summary_pluralises() {
        assert_eq!(dirty_summary(0), "clean");
        assert_eq!(dirty_summary(1), "1 uncommitted file");
        assert_eq!(dirty_summary(2), "2 uncommitted files");
    }

    /// The `tests:` and `hot` lines must describe the same tree identically on every platform.
    ///
    /// Both are built from `strip_prefix(root)`, which is backslash-separated on Windows.
    /// `has_py_tests` probes for `"/test_"`, so a nested `test_*.py` was invisible there — a
    /// true fact silently lost on one platform — and the `hot` line rendered `src\shared.rs`,
    /// making fingerprints incomparable across platforms.
    ///
    /// This fails on Windows before the fix and passes on Linux either way, which is exactly
    /// why it is worth writing down: CI here is Linux-only while development is on Windows,
    /// so this class of bug cannot be caught by CI noticing a regression.
    #[test]
    fn path_bearing_lines_are_identical_across_platforms() {
        // Python: a nested test_*.py must be detected.
        let py = tempfile::tempdir().unwrap();
        let pkg = py.path().join("pkg");
        std::fs::create_dir_all(&pkg).unwrap();
        std::fs::write(
            py.path().join("pyproject.toml"),
            "[project]\nname = \"pp\"\n",
        )
        .unwrap();
        std::fs::write(pkg.join("test_thing.py"), "def test_a():\n    pass\n").unwrap();
        std::fs::write(pkg.join("thing.py"), "def a():\n    pass\n").unwrap();

        let out = fingerprint(py.path());
        let tests_line = line_starting(&out, "tests:");
        assert!(
            tests_line.contains("test_*.py"),
            "a nested test_*.py must be detected whatever the path separator:\n{tests_line}"
        );
        assert!(
            !out.contains('\\'),
            "no backslash separators anywhere in the fingerprint:\n{out}"
        );

        // Rust: the hot line renders a nested path.
        let rs = tempfile::tempdir().unwrap();
        let src = rs.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(rs.path().join("Cargo.toml"), "[package]\nname = \"hf\"\n").unwrap();
        std::fs::write(src.join("shared.rs"), "pub struct Thing;\n").unwrap();
        for n in ["a", "b", "c", "d"] {
            std::fs::write(
                src.join(format!("{n}.rs")),
                format!("use crate::shared::Thing;\npub fn {n}(_t: Thing) {{}}\n"),
            )
            .unwrap();
        }

        let out = fingerprint(rs.path());
        let hot_line = line_starting(&out, "hot (");
        assert!(
            hot_line.contains("src/shared.rs"),
            "the hot line must use forward slashes:\n{hot_line}"
        );
        assert!(
            !out.contains('\\'),
            "no backslash separators anywhere in the fingerprint:\n{out}"
        );
    }

    /// A `.ts`/`.tsx` tie is a TypeScript project, not a "TSX" one.
    ///
    /// The tie-break was the display name, so an even React split resolved to "TSX" purely
    /// because `'S' < 'y'`. Deterministic but misleading — and `primary_lang` also decides
    /// which directories qualify as modules and whether `hot_files` runs, so the choice
    /// reaches further than the label.
    #[test]
    fn ts_tsx_tie_resolves_to_typescript() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        for f in 0..4 {
            std::fs::write(src.join(format!("a{f}.ts")), "export const a = 1;\n").unwrap();
            std::fs::write(
                src.join(format!("b{f}.tsx")),
                "export const B = () => null;\n",
            )
            .unwrap();
        }

        let runs: Vec<String> = (0..4).map(|_| fingerprint(dir.path())).collect();
        assert!(
            runs.windows(2).all(|w| w[0] == w[1]),
            "tied ts/tsx counts produced a varying fingerprint:\n{}",
            runs.join("\n---\n")
        );
        assert!(
            runs[0].contains("TypeScript project"),
            "an even .ts/.tsx split is a TypeScript project:\n{}",
            header_of(&runs[0])
        );
    }

    #[test]
    fn test_fingerprint_on_tilth() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let output = fingerprint(root);

        assert!(!output.is_empty(), "fingerprint should not be empty");
        assert!(
            output.contains("Rust"),
            "should detect Rust as primary language"
        );
        assert!(output.contains("Cargo.toml"), "should detect manifest");
        assert!(output.contains("tilth"), "should find project name");

        // Token budget: output should be compact
        let estimated_tokens = output.len() / 4;
        assert!(
            estimated_tokens < 300,
            "fingerprint should be <300 tokens, got {estimated_tokens}"
        );
    }

    #[test]
    fn test_fingerprint_empty_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let output = fingerprint(tmp.path());

        // Empty dir: should produce minimal output or empty
        // With 0 files and 0 modules, the header will say "0 source files"
        // but that's fine — it's still useful context
        assert!(
            output.is_empty() || output.contains("0 source files"),
            "empty dir should produce empty or minimal output, got: {output}"
        );
    }

    #[test]
    fn test_manifest_parsing() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let info = parse_cargo_toml(root).expect("should parse Cargo.toml");

        assert_eq!(info.name.as_deref(), Some("tilth"));
        assert!(info.version.is_some(), "should have a version");
        assert!(
            info.deps.iter().any(|d| d == "clap"),
            "deps should include clap: {:?}",
            info.deps
        );
        assert!(
            info.deps.iter().any(|d| d == "dashmap"),
            "deps should include dashmap: {:?}",
            info.deps
        );
    }
}
