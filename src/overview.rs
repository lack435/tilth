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
    //
    // `hot`, `git` and `tests` used to be emitted twice: once inside the parse-success arm
    // and again in the no-manifest arm, with nothing in between. So a manifest that was
    // found but could not be parsed took all three down with it and left a one-line
    // fingerprint — strictly worse than having no manifest at all, and the actual shape of
    // the #43 report. None of the three depends on the manifest, so they are emitted once,
    // unconditionally. Line order is unchanged for both cases that already worked.
    let parsed = find_manifest(root).map(|manifest| (parse_manifest(root, &manifest), manifest));

    // Deps line. Same disclosure as `dirs:` above — this line drops more than it shows on a
    // real manifest (tilth: 10 of 40, alphabetical, so every `tree-sitter-*` grammar falls
    // off the end), and saying nothing about that in an orientation payload is a defect.
    if let Some((Ok(info), _)) = &parsed {
        if !info.deps.is_empty() {
            let dep_str = info.deps.join(", ");
            let mut deps_line = format!("  deps: {dep_str}");
            let hidden = info.dep_total.saturating_sub(info.deps.len());
            if hidden > 0 {
                write!(deps_line, " +{hidden} more").unwrap();
            }
            lines.push(deps_line);
        }
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
    if let Some((info, manifest)) = &parsed {
        let mut manifest_line = format!("  manifest: {manifest}");
        match info {
            Ok(info) => match &info.name {
                // Byte-identical to before #80, deliberately and by construction: the named
                // branch is untouched. That is the property that makes the change safe, and
                // `a_named_manifest_line_is_unchanged` pins it against the spellings that
                // already worked.
                Some(name) => {
                    write!(manifest_line, " ({name}").unwrap();
                    if let Some(version) = &info.version {
                        write!(manifest_line, " v{version}").unwrap();
                    }
                    manifest_line.push(')');
                }
                // The nameless branch used to write nothing at all, leaving a bare
                // `manifest: Cargo.toml` — which reads as "this project has no name", the same
                // claim-by-silence #43 removed from the unreadable case (#80). It never means
                // that: overwhelmingly it means "this is a Cargo workspace root, and the names
                // are one level down". Now it says which.
                None => write!(manifest_line, " ({})", info.nameless.label()).unwrap(),
            },
            // Say why, rather than dropping the line. The damage in #35, #39, #41 and #43
            // was in every case the silence, not the parse failure — an agent that can see
            // "not UTF-8" can fix its manifest or stop trusting the block, whereas an absent
            // block reads as "this project has no name", which is a claim tilth cannot make.
            //
            // "unusable", not "unreadable": half the reasons are parse failures on a file
            // that read perfectly well, and telling an agent a syntax error is a read error
            // sends it to re-save the encoding instead of to the broken line. "Unusable" is
            // true across all of them — unreadable, unparseable, unsupported — and stays a
            // single greppable marker.
            Err(reason) => write!(manifest_line, " — unusable: {reason}").unwrap(),
        }
        lines.push(manifest_line);
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
    /// What the manifest *is*, for the case where it parses but declares no name (#80).
    ///
    /// Not an `Option`: every parser must decide, and the renderer reads it unconditionally
    /// whenever `name` is `None`. That is the whole point — the bug was a bare
    /// `manifest: Cargo.toml` line produced by having nothing to say and saying it silently, so
    /// "nothing to say" is made unrepresentable rather than merely discouraged. A parser that
    /// gains a new nameless path has to name it here or it will not compile.
    nameless: Nameless,
}

/// What a manifest that parsed cleanly is, when it carries no name.
///
/// The bare line this replaces read as "this project has no name", which is a claim tilth cannot
/// make — the same objection `overview`'s own manifest comment raises about the *unreadable* case
/// that #43 fixed with `— unusable: <reason>`. The nameless-but-readable case takes the same shape
/// and was never covered.
///
/// **Rendered as a parenthetical, never as `— unusable:`.** That distinction is the trap #80 named
/// explicitly: a workspace root is a completely valid manifest, and labelling it with the
/// failure marker would swap a vague line for a wrong one. The parenthetical is the slot the name
/// already occupies, so this says what was found in the place a reader already looks for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Nameless {
    /// Nothing at all: empty, or only comments and whitespace.
    ///
    /// Its own variant because an empty `Cargo.toml` is *valid* TOML and so parses and reports
    /// identically to a legitimate workspace root — two very different situations that the
    /// fingerprint could not tell apart. Only reachable for the TOML formats: an empty
    /// `package.json` is `malformed JSON` and an empty `go.mod` is `no module directive`, both
    /// caught before they get here.
    Empty,
    /// A Cargo workspace root: `[workspace]`, with the number of members it declares.
    ///
    /// The common case by far, and the reason #80 mattered more than it looked. **A workspace-root
    /// `Cargo.toml` has no `[package]` section at all**, so every Cargo workspace hit the bare
    /// line — at the repository root, which is exactly where an agent is most likely to launch.
    /// The truth is "this is a workspace root and the names are one level down", and the member
    /// count is sitting right there in the manifest.
    CargoWorkspace(usize),
    /// A `pyproject.toml` declaring a build backend and no packaging metadata: the ordinary
    /// Python layout with the name in `setup.cfg` or `setup.py`.
    ///
    /// "No packaging metadata" means both `[project]` (PEP 621) and `[tool.poetry]` absent. Other
    /// `[tool.*]` sections do not count and must not suppress this — `[tool.black]` is formatter
    /// settings, `[tool.setuptools]` is build configuration, and a project keeping its name in
    /// `setup.cfg` commonly has one or both. Requiring *nothing* but `[build-system]` would push
    /// those onto [`Nameless::NoNameDeclared`], which says strictly less about the same file.
    BuildSystemOnly,
    /// Parses, holds real content, and still declares no name.
    ///
    /// The honest fallback, and deliberately not silence. It states the observation rather than
    /// implying it by omission, and it is what separates "a manifest that simply has no name"
    /// from [`Nameless::Empty`] — the distinction #80's acceptance asks for.
    NoNameDeclared,
}

impl Nameless {
    /// The parenthetical body. Renders in the slot a name would have occupied.
    fn label(self) -> String {
        match self {
            Nameless::Empty => "empty".to_string(),
            // No count to report. Three inputs land here and all three deserve the same line: no
            // `members` key at all, `members = []`, and a `members` that is not an array (which
            // is read leniently rather than failing the manifest — see `CargoToml::workspace`).
            // "0 members" would be worse than silence for the first and the third, since neither
            // actually declares zero of anything.
            Nameless::CargoWorkspace(0) => "workspace".to_string(),
            // Declared *entries*, which is what the manifest says rather than what Cargo would
            // resolve — `members = ["crates/*"]` is one entry and may expand to many crates. The
            // count orients; it is not a crate census, and reporting the manifest's own number is
            // the only figure available without globbing the tree.
            Nameless::CargoWorkspace(1) => "workspace, 1 member".to_string(),
            Nameless::CargoWorkspace(n) => format!("workspace, {n} members"),
            Nameless::BuildSystemOnly => "build-system only".to_string(),
            Nameless::NoNameDeclared => "no name declared".to_string(),
        }
    }
}

/// True when TOML content declares nothing at all — empty, whitespace, or comments only.
///
/// Only the TOML formats need this; see [`Nameless::Empty`] for why the other two cannot reach it.
fn toml_declares_nothing(content: &str) -> bool {
    content
        .lines()
        .all(|l| l.trim().is_empty() || l.trim_start().starts_with('#'))
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

fn parse_manifest(root: &Path, manifest: &str) -> Result<ManifestInfo, String> {
    match manifest {
        "Cargo.toml" => parse_cargo_toml(root),
        "package.json" => parse_package_json(root),
        "go.mod" => parse_go_mod(root),
        "pyproject.toml" => parse_pyproject_toml(root),
        // Unreachable: `find_manifest` only ever returns one of the four above. Stated as a
        // reason rather than a silent `None` so that adding a name there and forgetting to
        // add it here shows up in the output instead of deleting the manifest block.
        other => Err(format!("no parser for {other}")),
    }
}

/// Read a manifest, or say why not.
///
/// `fs::read_to_string(..).ok()?` was the shape in all four parsers, and it turned every
/// read failure into an absent manifest block — no error, no warning, no partial result
/// (#43). A UTF-16 manifest is the case that motivated this: `read_to_string` requires
/// valid UTF-8 and PowerShell 5.1 writes UTF-16 LE by default on several paths, so it is
/// unusual but not exotic. `overview` exists to orient an agent *without* a tool call, so
/// this is the one place a silent failure is both invisible and load-bearing.
///
/// The reasons are fixed strings rather than `io::Error`'s `Display`, which is free to vary
/// with platform and locale. Nothing currently pins fingerprint text — the two byte-lock
/// tests cover `SERVER_INSTRUCTIONS` and `EDIT_MODE_EXTRA`, not this — so the guard here is
/// `unreadable_reasons_do_not_vary_by_platform` plus the deterministic-output property #28
/// established, not a golden file.
///
/// Fixed strings alone are **not** sufficient, because `ErrorKind` itself diverges for the
/// same logical failure. A directory named `package.json` — which `find_manifest` accepts,
/// since `Path::exists()` is true for directories — surfaces as `PermissionDenied` on
/// Windows and `IsADirectory` on Linux. Reporting "permission denied" for that would send an
/// agent chasing file modes for a problem that is "this is a directory", so the kind is
/// tested explicitly up front rather than inferred from the error.
fn read_manifest(path: &Path) -> Result<String, String> {
    // Ahead of the read, so the answer does not depend on which errno the platform picks.
    if path.is_dir() {
        return Err("not a file".to_string());
    }
    // `fs::read` plus an explicit `String::from_utf8`, rather than `read_to_string`, which is
    // those two steps welded together. Splitting them is what lets the UTF-16 decode see the
    // bytes (#65) — and it does so **without reading twice**. A first version kept
    // `read_to_string` and re-read the file on the `InvalidData` arm, which paid for a second
    // read, opened a window in which the bytes decoded were not the bytes that failed
    // validation, and collapsed a `NotFound` on that second read into "not UTF-8" instead of
    // "disappeared during the scan". One read has none of those.
    //
    // The success path costs no more than before: `String::from_utf8` takes the `Vec` by value
    // and validates in place, so a UTF-8 manifest is one read and no copy, exactly as
    // `read_to_string` was.
    let bytes = fs::read(path).map_err(|e| {
        match e.kind() {
            std::io::ErrorKind::PermissionDenied => "permission denied",
            // `find_manifest` stat'd the file moments ago, so this is a genuine race.
            std::io::ErrorKind::NotFound => "disappeared during the scan",
            // Deliberately not the word "unreadable": that is the line's own prefix in the
            // old spelling, and "unreadable: unreadable" told the reader nothing at all.
            _ => "read failed",
        }
        .to_string()
    })?;

    // "not UTF-8" now comes from `String::from_utf8` rather than from `io::ErrorKind::InvalidData`,
    // and the reason that string was platform-stable is unchanged and now load-bearing rather than
    // merely reassuring: the verdict is std's own UTF-8 validation over a buffer already in memory,
    // never an errno the OS picked. The code branches on it to decide whether to try a decode.
    match String::from_utf8(bytes) {
        // Valid UTF-8 is not the same as valid *text* here (#79). UTF-16 holding ASCII is valid
        // UTF-8 — NUL is a legal UTF-8 byte — so a BOM-less UTF-16 manifest sails through
        // `from_utf8` and reaches the parser as NUL-interleaved text, which then reports
        // `malformed TOML` / `malformed JSON`: an encoding problem wearing a syntax error's
        // clothes, sending the reader after a missing brace in a file whose braces are fine.
        //
        // The observation is certain and the cause is inferred, so the wording hedges only the
        // cause. **No manifest format permits a raw NUL** — TOML and JSON both forbid unescaped
        // control characters in strings, and JSON spells one as a six-character escape — so
        // "there are NUL bytes in here" is a fact, while "probably UTF-16 without a BOM" is the
        // overwhelmingly likely explanation and is marked as a guess.
        //
        // This is the check `decode_utf16_with_bom` deliberately would not move up to this arm,
        // and the reason it could not is that it reports `not UTF-8` — which would be **false**
        // about a file that just passed UTF-8 validation. A reason of its own has no such
        // problem. The two NUL tests are therefore different claims: that one rejects a *decode*
        // that produced NULs (so the BOM lied about the encoding, and `not UTF-8` is true), this
        // one rejects bytes that really are UTF-8 and really are not text.
        //
        // Ordinary malformed input is untouched, which is the property this trades against: a
        // truncated `{"name":` holds no NUL and still reports `malformed JSON`.
        Ok(text) if text.contains('\0') => {
            Err("NUL bytes, probably UTF-16 without a BOM".to_string())
        }
        Ok(text) => Ok(text),
        // #43 stopped at reporting this. Reporting is the wrong end state: the agent learns the
        // file is unusable at the one moment `overview` exists to tell it the project's name,
        // version and dependencies without a tool call. So try to decode first (#65).
        Err(not_utf8) => {
            decode_utf16_with_bom(not_utf8.as_bytes()).ok_or_else(|| "not UTF-8".to_string())
        }
    }
}

/// Decode BOM-prefixed UTF-16 to a `String`, or `None` for anything else.
///
/// `String::from_utf16` is in std, so this costs no dependency — #43 estimated the decode as
/// costing `encoding_rs` or a hand-rolled decoder and that was wrong, which is what reopened
/// the trade-off as #65.
///
/// **BOM-driven only, and deliberately so.** A BOM makes the encoding and the byte order
/// unambiguous. Without one, detection is a heuristic — and worse, UTF-16 LE holding ASCII is
/// *valid UTF-8*, since NUL is a legal UTF-8 byte, so a BOM-less file never reaches this at all:
/// `String::from_utf8` succeeds and hands NUL-interleaved text to the parser.
///
/// That case is #79, and it is now handled — but in `read_manifest`, on the `from_utf8` **success**
/// arm, not here. The reasoning this comment used to record still stands and is exactly why it
/// could not be closed from inside this function: reaching it means reporting `not UTF-8`, which
/// would be a false statement about a file that just passed UTF-8 validation. A reason of its own
/// (`NUL bytes, probably UTF-16 without a BOM`) has no such problem, so all four formats now name
/// the encoding rather than reporting `malformed TOML` / `malformed JSON` / `no module directive`.
/// See `a_bomless_utf16_manifest_reports_the_encoding_not_the_syntax`.
///
/// The BOM is consumed rather than translated, so the returned text starts at the first real
/// character. A *doubled* BOM therefore leaves one U+FEFF, which the TOML and JSON parsers strip
/// for themselves and `parse_go_mod` handles with `trim_start_bom_aware` — the doubled-BOM handling
/// the rest of the codebase settled on. Note that consuming it here versus leaving it to them is
/// not observable through any parser, so it is tidiness rather than correctness.
fn decode_utf16_with_bom(bytes: &[u8]) -> Option<String> {
    let (rest, big_endian) = match bytes {
        [0xFF, 0xFE, rest @ ..] => (rest, false),
        [0xFE, 0xFF, rest @ ..] => (rest, true),
        _ => return None,
    };
    // An odd tail cannot be UTF-16, and `chunks_exact` would silently drop the stray byte.
    if rest.len() % 2 != 0 {
        return None;
    }

    // Both of these allocate infallibly — `Vec` and `String` grow through `handle_alloc_error`,
    // which aborts, whereas `fs::read` reserves through `try_reserve_exact` and surfaces
    // `OutOfMemory` as the "read failed" reason. So peak for an N-byte manifest is ~3.5N on this
    // path (N still owned by the `FromUtf8Error`, N for the units, up to 1.5N for the `String`)
    // against N before, and the top of that range aborts rather than reporting. Nothing caps
    // manifest size anywhere — `find_manifest` gates on `Path::exists()` alone — so this is a
    // pre-existing exposure that the decode multiplies rather than a new one, and it needs a cap
    // chosen deliberately rather than invented here. Reachable only by a UTF-16 manifest sized
    // near available memory.
    let units: Vec<u16> = rest
        .chunks_exact(2)
        .map(|c| {
            if big_endian {
                u16::from_be_bytes([c[0], c[1]])
            } else {
                u16::from_le_bytes([c[0], c[1]])
            }
        })
        .collect();
    // `from_utf16`, not `from_utf16_lossy`: an unpaired surrogate means the guess was wrong, and
    // a manifest silently studded with U+FFFD is the kind of confidently-wrong output the
    // renderer's own comments warn is worse than a useless one.
    let text = String::from_utf16(&units).ok()?;

    // A NUL in the decoded text means these bytes were not UTF-16 text, whatever the BOM claimed.
    // UTF-32 LE is the case that matters: it opens `FF FE 00 00`, whose first two bytes *are* the
    // UTF-16 LE BOM, so it matches above and decodes — NUL is a valid code point — into NUL-riddled
    // text that then fails the TOML/JSON parser. That turns "not UTF-8" into "malformed JSON",
    // sending the reader after a syntax error that is really an encoding.
    //
    // Tested on the decoded text rather than by matching an `FF FE 00 00` prefix, which was the
    // first version: the prefix only catches a *declared* UTF-32 LE file, while `FF FE` followed by
    // a UTF-32-shaped payload with no UTF-32 BOM walks past it and produces the same wrong reason.
    // No manifest format has a legitimate raw NUL — JSON writes one as a six-character escape — so
    // rejecting all of them costs nothing real.
    //
    // **This covers the decode path only**, and the reason that is not the whole story is worth
    // keeping straight: a BOM-*less* UTF-16 file is valid UTF-8 and never arrives here. It is
    // caught on the `from_utf8` success arm in `read_manifest` instead, under a different reason,
    // because the two situations are genuinely different claims. Here the BOM lied about the
    // encoding, so `not UTF-8` is true. There the file really is UTF-8 and really is not text, so
    // it says so (#79).
    if text.contains('\0') {
        return None;
    }
    Some(text)
}

fn parse_cargo_toml(root: &Path) -> Result<ManifestInfo, String> {
    #[derive(Deserialize)]
    struct CargoToml {
        package: Option<Package>,
        dependencies: Option<toml::Table>,
        // `toml::Value` rather than a typed `Workspace`, and that is not laziness. A typed field
        // makes the **whole manifest** fail to parse when its shape is unexpected, which moves the
        // failure into a branch #80 promised not to touch: measured against the parent commit,
        // `workspace = ".."` and `members = "a"` alongside a perfectly good `[package]` rendered
        // `manifest: Cargo.toml (widget)` before and `— unusable: malformed TOML` after — a named
        // manifest losing its whole block, name, version and `deps:` line together. Those inputs
        // are invalid Cargo either way, but "reading a field for presence must not change whether
        // an unrelated manifest parses" is the rule that keeps the change confined where the
        // safety argument says it is.
        workspace: Option<toml::Value>,
    }
    #[derive(Deserialize)]
    struct Package {
        name: Option<String>,
        version: Option<String>,
    }
    let content = read_manifest(&root.join("Cargo.toml"))?;
    // The `toml` crate strips exactly *one* BOM, so a doubled one still fails — and
    // `.ok()?` turned that into the same silent loss of the whole manifest block as the
    // `package.json` bug. Cheap to close, and leaving it would make the repeat-stripping
    // both BOM helpers already do inconsistent with the two parsers that skip them.
    let stripped = crate::lang::outline::strip_bom(&content);
    let parsed: CargoToml = toml::from_str(stripped).map_err(|_| "malformed TOML".to_string())?;
    let (name, version) = parsed.package.map_or((None, None), |p| (p.name, p.version));
    // Order matters: `[workspace]` beats emptiness, because a manifest holding a `[workspace]`
    // table is not empty. Emptiness is only consulted once nothing else has anything to say.
    let nameless = match &parsed.workspace {
        Some(ws) => Nameless::CargoWorkspace(
            ws.get("members")
                .and_then(toml::Value::as_array)
                .map_or(0, Vec::len),
        ),
        None if toml_declares_nothing(stripped) => Nameless::Empty,
        None => Nameless::NoNameDeclared,
    };
    let (deps, dep_total) = cap_deps(
        parsed
            .dependencies
            .map(|d| d.into_iter().map(|(k, _)| k).collect())
            .unwrap_or_default(),
    );
    Ok(ManifestInfo {
        name,
        version,
        deps,
        dep_total,
        nameless,
    })
}

fn parse_package_json(root: &Path) -> Result<ManifestInfo, String> {
    #[derive(Deserialize)]
    struct PackageJson {
        name: Option<String>,
        version: Option<String>,
        dependencies: Option<serde_json::Map<String, serde_json::Value>>,
    }

    let content = read_manifest(&root.join("package.json"))?;
    // `.ok()?` on a BOM'd file discarded the whole manifest block — name, version and the
    // dependency list — from the fingerprint injected at MCP initialize, with no error
    // anywhere. `overview` exists to orient an agent without a tool call, so failing this
    // way is failing silently at exactly the wrong moment.
    let parsed: PackageJson = serde_json::from_str(crate::lang::outline::strip_bom(&content))
        .map_err(|_| "malformed JSON".to_string())?;
    let (deps, dep_total) = cap_deps(
        parsed
            .dependencies
            .map(|d| d.into_iter().map(|(k, _)| k).collect())
            .unwrap_or_default(),
    );
    Ok(ManifestInfo {
        name: parsed.name,
        version: parsed.version,
        deps,
        dep_total,
        // JSON has no comment syntax and an empty file is `malformed JSON`, so `Nameless::Empty`
        // is unreachable here — `{}` is the nameless case, and it is not empty.
        nameless: Nameless::NoNameDeclared,
    })
}

fn parse_go_mod(root: &Path) -> Result<ManifestInfo, String> {
    let content = read_manifest(&root.join("go.mod"))?;
    let mut name = None;
    let mut deps: Vec<String> = Vec::new();
    let mut in_require = false;

    for line in content.lines() {
        // `line.trim()` is not BOM-aware, so a BOM'd `go.mod` reported no module name at
        // all — the same mistake as the import-detection bug, in a file that fix did not
        // visit. `require` entries were unaffected only because they never sit on line 1.
        let trimmed = crate::lang::outline::trim_start_bom_aware(line).trim_end();
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

    // `go.mod` is the one format with no parser to fail, so nothing here could ever report
    // a reason — a file that yields neither a module name nor a require entry produced a
    // bare `manifest: go.mod` and looked like an unnamed module rather than a broken file.
    //
    // The input this was written for — BOM-less UTF-16, valid UTF-8 because NUL is a legal UTF-8
    // byte — no longer reaches here: `read_manifest` rejects a NUL-bearing file with a reason that
    // names the encoding (#79), which is the more accurate of the two, since such a file is not
    // missing a module directive so much as unreadable as text. This guard still covers what
    // remains: a NUL-free `go.mod` that genuinely declares neither a module nor a require, which
    // would otherwise render a bare `manifest: go.mod` and read as an unnamed module rather than a
    // broken file. `module` is mandatory in a real `go.mod`, so requiring one signal or the other
    // costs nothing on a valid file. Pinned by
    // `a_go_mod_with_no_module_line_still_reports_its_own_reason`.
    if name.is_none() && deps.is_empty() {
        return Err("no module directive".to_string());
    }

    let (deps, dep_total) = cap_deps(deps);

    Ok(ManifestInfo {
        name,
        version: None,
        deps,
        dep_total,
        // Reachable only by a `go.mod` with `require` entries but no `module` line — the guard
        // above already rejects the file that has neither. Emptiness cannot reach here for the
        // same reason.
        nameless: Nameless::NoNameDeclared,
    })
}

fn parse_pyproject_toml(root: &Path) -> Result<ManifestInfo, String> {
    #[derive(Default, Deserialize)]
    struct PyProject {
        project: Option<Project>,
        // `toml::Value`, not a typed table: this field exists only to be tested for presence, and
        // a typed one would make the *whole manifest* fail to parse for a spelling that previously
        // parsed fine. See `parse_cargo_toml`'s `workspace` for the measurement.
        #[serde(rename = "build-system")]
        build_system: Option<toml::Value>,
        /// Poetry keeps its metadata under `[tool.poetry]` rather than `[project]`.
        tool: Option<toml::Value>,
    }
    #[derive(Default, Deserialize)]
    struct Project {
        name: Option<String>,
        version: Option<String>,
        dependencies: Option<Vec<String>>,
    }

    let content = read_manifest(&root.join("pyproject.toml"))?;
    // Doubled BOM — see `parse_cargo_toml`.
    let stripped = crate::lang::outline::strip_bom(&content);
    let parsed: PyProject = toml::from_str(stripped).map_err(|_| "malformed TOML".to_string())?;

    // Poetry declares its name under `[tool.poetry]`, and `[build-system]` is effectively
    // mandatory in a Poetry project — so testing `build-system` alone reported `(build-system
    // only)` for a manifest carrying a perfectly good name and version. That is worse than the
    // bare line #80 set out to replace: vague became *wrong*, on every Poetry 1.x project. Read
    // the name instead of labelling its absence.
    //
    // Only `name` and `version` are read here. `[tool.poetry.dependencies]` is deliberately not
    // folded into `deps` — its keys include `python`, which is an interpreter constraint rather
    // than a package, so merging it needs a judgement this change does not need to make. A Poetry
    // manifest therefore renders its name with no `deps:` line, exactly as it did before.
    let poetry = parsed.tool.as_ref().and_then(|t| t.get("poetry"));
    let poetry_str = |key: &str| {
        poetry
            .and_then(|p| p.get(key))
            .and_then(toml::Value::as_str)
            .map(str::to_string)
    };

    // `[build-system]` only means "build-system only" when there is genuinely nothing else. A
    // `[project]` table that happens to omit `name` is a different situation and must not claim
    // to be this one.
    let nameless = if parsed.build_system.is_some() && parsed.project.is_none() && poetry.is_none()
    {
        Nameless::BuildSystemOnly
    } else if toml_declares_nothing(stripped) {
        Nameless::Empty
    } else {
        Nameless::NoNameDeclared
    };
    let project = parsed.project.unwrap_or_default();
    let name = project.name.or_else(|| poetry_str("name"));
    let version = project.version.or_else(|| poetry_str("version"));
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
    Ok(ManifestInfo {
        name,
        version,
        deps,
        dep_total,
        nameless,
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
    //
    // Remeasured for #45, which made the non-git case actually probe. Until then an
    // unresolvable include in a tree with no `.git` cost exactly one `enclosing_repo_root`
    // walk and then short-circuited, so that case was anomalously cheap *because* it was
    // broken. On a 1,852-file C++ module that is not a git checkout, whole-`fingerprint`
    // wall time, five warm runs each of a release build:
    //
    //   before #45   71-74ms — and no `hot` line at all, nothing resolved
    //   after #45    83-88ms — five hot files resolved through the include root
    //
    // Non-C/C++ projects pay none of it: the boundary is only computed for a file whose own
    // language reads one. Measured on this repository (Rust, a checkout), before and after are
    // 76-86ms and 77ms, and the fingerprint is byte-identical.
    //
    // So the probes cost ~12ms where there previously were none, and the result stays well
    // inside the 250ms soft budget. No new ceiling: the worst case is still a tree full of
    // *unresolvable* includes, which was already probing before #45 whenever a `.git` was
    // present, and that is what the figures above it measure.
    //
    // Git-tree *output* is unchanged, because `boundary_from_file` hands back the identical
    // path `resolve_c_include` would have computed for itself. Checked by diffing the
    // pre-#45 and post-#45 binaries at three launch directories on a 1,933-file C++
    // checkout — repository root, a subdirectory, and a parent above the checkout — all
    // three identical. The *syscall mix* does change, and in tilth's favour: one upward
    // `.exists()` walk per file replaces one per unresolvable include, at the cost of the
    // `is_within(dir, boundary)` canonicalize pair that the `Some` arm now reaches.
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

        // The boundary is derived from the *file*, not from `root` (#45).
        //
        // Passing `root` directly is the obvious patch and it is wrong. `fingerprint` is only
        // ever called as `fingerprint(&cwd)`, and every file here is `root.join(rel_path)`, so
        // `root` contains every candidate and would always outrank the `.git` fallback:
        // launched above a checkout the scan could count a hot file in a *different*
        // repository, launched inside a subdirectory it would stop counting headers above the
        // launch dir. Both change what the initialize fingerprint claims about a git tree,
        // which is the common case and the one #29/#31 went to some trouble to make truthful.
        //
        // `boundary_from_file` inverts that: the enclosing repository wins, and `root` is only
        // reached when there is no `.git` at all — the case that was broken, where resolution
        // gave up before probing and every project-relative include bucketed as external. In a
        // git tree it returns the identical path `resolve_c_include` computes for itself, so
        // this launch directory's fingerprint is unchanged from before #45. That was checked by
        // running both binaries at the repository root, a subdirectory, and a parent above the
        // checkout of a 1,933-file C++ checkout and diffing: identical at all three, each with a
        // substantive and *different* hot line. It is not expressible as a test — it compares
        // two implementations — so the three `hot_files_*` fixtures pin the *rule* instead, one
        // per way of getting it wrong.
        //
        // Refusing to cross into another project is a claim about **git** trees only. Where
        // there is no `.git` the tree root is the only boundary that exists, so a sibling
        // directory under the same root can be reached — `hot_files_in_a_non_git_tree_can_reach
        // _a_sibling_project` states that outcome rather than leaving it implied. It is the same
        // reach a declared scope has always had, and the alternative is the pre-#45 behaviour of
        // resolving nothing at all.
        //
        // Computed only for C/C++, because only `resolve_c_include` reads a boundary —
        // `resolve_import_to_file` dispatches on the file's own language, so for a Rust or
        // Python file this is an upward `.exists()` walk whose result is discarded, and in a
        // tree with no `.git` that walk runs to the filesystem root every time. At up to 100
        // files per fingerprint that is a few hundred pointless `stat`s on the majority
        // language case, which the C++ measurement above would never have shown.
        let boundary = matches!(detect_file_type(&full), FileType::Code(Lang::C | Lang::Cpp))
            .then(|| crate::read::imports::boundary_from_file(full.parent().unwrap_or(root), root));
        let resolved = crate::read::imports::resolve_related_files_with_content(
            &full,
            &content,
            boundary.as_deref(),
        );
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

    /// A UTF-8 BOM, written as bytes. A manifest fixture built from a `&str` literal
    /// cannot see any of this, which is how the class of bug kept coming back.
    const UTF8_BOM: &[u8] = &[0xEF, 0xBB, 0xBF];

    fn write_with_bom(path: &Path, prefix: &[u8], body: &str) {
        let mut bytes = prefix.to_vec();
        bytes.extend_from_slice(body.as_bytes());
        std::fs::write(path, &bytes).unwrap();
    }

    /// `serde_json` rejects a leading BOM, and `parse_package_json` swallowed that with
    /// `.ok()?` — so a BOM'd `package.json` dropped name, version *and* the dependency
    /// list from the fingerprint injected at MCP initialize, with no error anywhere.
    ///
    /// This is the worst of the BOM parse failures: `overview` exists to orient an agent
    /// without a tool call, so failing silently here fails at exactly the wrong moment.
    #[test]
    fn bom_package_json_keeps_the_manifest_block() {
        let body = "{\"name\":\"demo-app\",\"version\":\"2.3.4\",\
                    \"dependencies\":{\"react\":\"^18\",\"lodash\":\"^4\"}}";

        let mut outs = Vec::new();
        for prefix in [&[][..], UTF8_BOM] {
            let dir = tempfile::tempdir().unwrap();
            write_with_bom(&dir.path().join("package.json"), prefix, body);
            std::fs::write(dir.path().join("index.js"), "export const x = 1;\n").unwrap();
            outs.push(fingerprint(dir.path()));
        }

        // Pinned against literals first: a comparison alone would be satisfied by both
        // runs losing the manifest, which is precisely the bug.
        for needle in ["demo-app", "2.3.4", "react", "lodash"] {
            assert!(
                outs[0].contains(needle),
                "fixture is broken: unmarked package.json lost {needle}:\n{}",
                outs[0]
            );
            assert!(
                outs[1].contains(needle),
                "a BOM'd package.json lost {needle} from the fingerprint:\n{}",
                outs[1]
            );
        }
        assert_eq!(outs[1], outs[0], "a BOM changed the fingerprint");
    }

    /// UTF-16 LE with BOM, written as bytes. Like `UTF8_BOM`, a `&str` literal cannot
    /// express this — and it is the encoding PowerShell 5.1 produces by default on several
    /// paths, which is where the reports come from.
    fn utf16le_with_bom(body: &str) -> Vec<u8> {
        let mut bytes = vec![0xFF, 0xFE];
        for unit in body.encode_utf16() {
            bytes.extend_from_slice(&unit.to_le_bytes());
        }
        bytes
    }

    fn utf16be_with_bom(body: &str) -> Vec<u8> {
        let mut bytes = vec![0xFE, 0xFF];
        for unit in body.encode_utf16() {
            bytes.extend_from_slice(&unit.to_be_bytes());
        }
        bytes
    }

    /// Bytes that are neither valid UTF-8 nor BOM-prefixed UTF-16, so `read_manifest` has no
    /// decode to attempt and must still report `not UTF-8`.
    ///
    /// `0x80` is a UTF-8 continuation byte with nothing to continue, and the file does not open
    /// with either UTF-16 BOM — both halves matter, since `FF FE` would now be *decoded*.
    fn not_utf8_and_not_utf16(body: &str) -> Vec<u8> {
        let mut bytes = vec![0x80];
        bytes.extend_from_slice(body.as_bytes());
        bytes
    }

    /// A manifest that cannot be read must say so, not vanish.
    ///
    /// `fs::read_to_string` requires valid UTF-8, so a non-UTF-8 manifest returned
    /// `Err(InvalidData)` and `.ok()?` discarded it — taking the entire manifest block out
    /// of the fingerprint injected at MCP initialize with no error anywhere (#43). Worse
    /// than the report captured: `hot`, `git` and `tests` were nested inside the
    /// parse-success arm, so they went too and the whole fingerprint collapsed to its
    /// header line.
    ///
    /// Both halves are asserted. That the reason is visible is the fix; that the unrelated
    /// lines survive is what makes it better than "no manifest" rather than merely
    /// different from it.
    ///
    /// The fixture used to be UTF-16 LE, which #65 now decodes — so it would have stopped being
    /// unusable and this test would have gone green while asserting nothing. It is now bytes with
    /// no decode available, which is the case the reason string still exists for.
    #[test]
    fn an_unreadable_manifest_says_why_and_keeps_the_rest_of_the_block() {
        let body = "{\"name\":\"utf16-app\",\"version\":\"9.9.9\"}";

        // Both testable rescued lines are made reachable, not just `tests:`. With only one
        // live, re-nesting the other back under the parse-success arm would leave this green.
        //
        // `git:` is the third and is deliberately uncovered: `git_context` shells out, which
        // is why `dirty_summary` was split out of it "purely so it is testable". It sits in
        // the same unconditional block as these two, so what pins the structure pins it too.
        let build = |dir: &Path, manifest: &[u8]| {
            std::fs::write(dir.join("package.json"), manifest).unwrap();
            // -> tests:
            std::fs::write(dir.join("index.test.js"), "test('x', () => {});\n").unwrap();
            // Two importers of one module -> hot:
            std::fs::write(dir.join("util.js"), "export const u = 1;\n").unwrap();
            for f in ["a.js", "b.js"] {
                std::fs::write(dir.join(f), "import { u } from './util';\n").unwrap();
            }
        };

        let dir = tempfile::tempdir().unwrap();
        build(dir.path(), &not_utf8_and_not_utf16(body));
        let out = fingerprint(dir.path());

        assert!(
            out.contains("manifest: package.json — unusable: not UTF-8"),
            "an unusable manifest must name itself and say why:\n{out}"
        );
        for line in ["hot (× = importers):", "tests:"] {
            assert!(
                out.contains(line),
                "`{line}` does not depend on the manifest and must survive it:\n{out}"
            );
        }

        // The control: the identical content as UTF-8 reports name and version, so the
        // assertions above are about the encoding and not about a broken fixture.
        let ok_dir = tempfile::tempdir().unwrap();
        build(ok_dir.path(), body.as_bytes());
        let ok_out = fingerprint(ok_dir.path());
        assert!(
            ok_out.contains("utf16-app") && ok_out.contains("9.9.9"),
            "fixture is broken: the UTF-8 spelling must parse:\n{ok_out}"
        );
        assert!(
            !ok_out.contains("unusable"),
            "a readable manifest must not carry the note:\n{ok_out}"
        );

        // Line order is claimed unchanged by the restructure, so pin it. Without this,
        // moving the manifest block above hot/git/tests leaves every test in the file green.
        let idx = |hay: &str, needle: &str| {
            hay.find(needle)
                .unwrap_or_else(|| panic!("missing {needle}:\n{hay}"))
        };
        for text in [&out, &ok_out] {
            assert!(
                idx(text, "hot (× = importers):") < idx(text, "tests:")
                    && idx(text, "tests:") < idx(text, "manifest:"),
                "emission order must stay hot -> [git] -> tests -> manifest:\n{text}"
            );
        }
    }

    /// Every manifest format must **decode** a BOM'd UTF-16 file, in both byte orders (#65).
    ///
    /// This test used to assert the opposite — that each of the four reported
    /// `unusable: not UTF-8` — which was #43 taking the "fail visibly" option and leaving the
    /// decode open. Reporting fixed the silence but still told the agent nothing about the
    /// project at the one moment `overview` exists to say what it is called and what it depends
    /// on. So the expectation inverts: name, version and dependency list, from the encoding
    /// PowerShell 5.1 writes by default on several paths.
    ///
    /// All four formats, because they reach the decode through one `read_manifest` but each
    /// parses independently — and both byte orders, because a decoder that ignores the BOM it
    /// matched on passes LE and produces nothing but mojibake on BE.
    ///
    /// Fixtures write the UTF-16 bytes explicitly, per #35/#41: a `&str` literal cannot express
    /// this, and a fixture that is secretly UTF-8 proves nothing.
    #[test]
    fn every_manifest_format_decodes_a_bommed_utf16_file() {
        // `(manifest, body, expected substrings)`. A dependency is included per format so the
        // decode is shown to survive the whole parse, not just the name line.
        let cases: &[(&str, &str, &[&str])] = &[
            (
                "Cargo.toml",
                "[package]\nname = \"crate-c\"\nversion = \"0.1.0\"\n\n[dependencies]\nserde = \"1\"\n",
                &["crate-c", "0.1.0", "serde"],
            ),
            (
                "package.json",
                "{\"name\":\"pkg-p\",\"version\":\"0.2.0\",\"dependencies\":{\"left-pad\":\"1.0.0\"}}",
                &["pkg-p", "0.2.0", "left-pad"],
            ),
            (
                "go.mod",
                "module example.com/mod-g\n\nrequire (\n\tgithub.com/acme/widget v1.2.3\n)\n",
                &["example.com/mod-g", "widget"],
            ),
            (
                "pyproject.toml",
                "[project]\nname = \"proj-y\"\nversion = \"0.3.0\"\ndependencies = [\"requests\"]\n",
                &["proj-y", "0.3.0", "requests"],
            ),
        ];

        for (manifest, body, expected) in cases {
            for (order, encode) in [
                ("LE", utf16le_with_bom as fn(&str) -> Vec<u8>),
                ("BE", utf16be_with_bom as fn(&str) -> Vec<u8>),
            ] {
                let dir = tempfile::tempdir().unwrap();
                std::fs::write(dir.path().join(manifest), encode(body)).unwrap();
                std::fs::write(dir.path().join("main.go"), "package main\n").unwrap();

                let out = fingerprint(dir.path());
                assert!(
                    !out.contains("unusable"),
                    "{manifest} in UTF-16 {order} must decode, not report a reason:\n{out}"
                );
                for want in *expected {
                    assert!(
                        out.contains(want),
                        "{manifest} in UTF-16 {order} lost `{want}` from the fingerprint:\n{out}"
                    );
                }
            }

            // The control, once per format rather than once per byte order — it does not depend on
            // the order. The same body as UTF-8 must carry the same claims, so a failure above is
            // about the encoding and not about a fixture the parser never liked. Equality of the
            // rendered lines is asserted separately, by
            // `a_utf16_manifest_renders_what_its_utf8_spelling_renders`.
            let utf8_dir = tempfile::tempdir().unwrap();
            std::fs::write(utf8_dir.path().join(manifest), body).unwrap();
            std::fs::write(utf8_dir.path().join("main.go"), "package main\n").unwrap();
            let utf8_out = fingerprint(utf8_dir.path());
            for want in *expected {
                assert!(
                    utf8_out.contains(want),
                    "fixture is broken: the UTF-8 spelling of {manifest} lacks `{want}`:\n\
                     {utf8_out}"
                );
            }
        }
    }

    /// The reasons #43 established must all survive the decode being added (#65).
    ///
    /// The decode sits on the `Err` arm of `read_manifest`'s `String::from_utf8`, so everything that
    /// is *not* a UTF-8 validation failure has to reach its old reason untouched, and a validation
    /// failure with no usable BOM has to keep reporting `not UTF-8` rather than falling through to a
    /// parse error. Each row is a distinct way to fail, and the two UTF-32-shaped ones are the cases
    /// a naive BOM match gets wrong.
    #[test]
    fn a_file_with_no_usable_utf16_bom_still_reports_not_utf8() {
        let json = "{\"name\":\"n\",\"version\":\"1.0.0\"}";

        // UTF-32 LE opens `FF FE 00 00`, whose first two bytes *are* the UTF-16 LE BOM. Decoding
        // it as UTF-16 succeeds and yields NUL-riddled text, so without the NUL check in
        // `decode_utf16_with_bom` this reports `malformed JSON` — a syntax error for what is
        // really an encoding.
        let mut utf32le = vec![0xFF, 0xFE, 0x00, 0x00];
        for ch in json.chars() {
            utf32le.extend_from_slice(&(ch as u32).to_le_bytes());
        }

        // The same payload shape with **no** UTF-32 BOM, so only the first character's high bytes
        // supply the `00 00`. A guard matching the four-byte `FF FE 00 00` prefix — the first
        // version of this — lets it through and reports `malformed JSON` again. Testing the decoded
        // text for NUL catches both, which is why the check moved off the prefix.
        let mut utf32le_bomless = vec![0xFF, 0xFE];
        for ch in json.chars() {
            utf32le_bomless.extend_from_slice(&(ch as u32).to_le_bytes());
        }

        // An odd byte count cannot be UTF-16; `chunks_exact` would drop the stray byte silently.
        let mut odd_tail = utf16le_with_bom(json);
        odd_tail.push(0x21);

        // A BOM followed by an unpaired high surrogate: well-formed UTF-16 units, not valid
        // UTF-16 text. `from_utf16_lossy` would accept this and hand the parser U+FFFD.
        let unpaired_surrogate = vec![0xFF, 0xFE, 0x00, 0xD8, 0x21, 0x00];

        // A perfectly well-formed UTF-16 document whose *content* holds a raw NUL. This is the one
        // class where the two encodings deliberately disagree — as UTF-8 the same body reaches the
        // parser and reports `malformed JSON` — and pinning it here is what keeps the NUL test from
        // being "fixed" by moving it to the `from_utf8` success arm, where it would call a file that
        // really is UTF-8 "not UTF-8". `a_utf16_manifest_renders_what_its_utf8_spelling_renders`
        // names the same bound from the other side.
        let nul_in_content = utf16le_with_bom("{\"name\":\"a\0b\",\"version\":\"1.0.0\"}");

        let cases: &[(&str, Vec<u8>)] = &[
            ("a lone continuation byte", not_utf8_and_not_utf16(json)),
            ("UTF-32 LE", utf32le),
            ("a BOM-less UTF-32 LE payload", utf32le_bomless),
            ("an odd trailing byte", odd_tail),
            ("an unpaired surrogate", unpaired_surrogate),
            ("a raw NUL in the content", nul_in_content),
        ];

        for (label, bytes) in cases {
            assert!(
                std::str::from_utf8(bytes).is_err(),
                "{label}: fixture must not be valid UTF-8, or it never reaches the decode"
            );

            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join("package.json"), bytes).unwrap();
            let out = fingerprint(dir.path());
            assert!(
                out.contains("manifest: package.json — unusable: not UTF-8"),
                "{label} must report `not UTF-8`, not a parse error or a decode:\n{out}"
            );
        }
    }

    /// A decoded manifest must still be able to fail its *parser*, with the parser's own reason.
    ///
    /// The decode is additive: it turns bytes into text and hands them on. If it started
    /// swallowing failures — returning the reason string for a decode problem where the content
    /// is simply malformed — an agent would be told to re-save a file whose real problem is a
    /// missing brace.
    #[test]
    fn a_decoded_utf16_manifest_still_reports_a_parse_failure() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("package.json"),
            utf16le_with_bom("{\"name\": \"broken\","),
        )
        .unwrap();

        let out = fingerprint(dir.path());
        assert!(
            out.contains("manifest: package.json — unusable: malformed JSON"),
            "a decoded but malformed manifest must report the parse failure:\n{out}"
        );
    }

    /// A doubled BOM must not survive the decode into the parsed values.
    ///
    /// `decode_utf16_with_bom` consumes the BOM it matched on, so a file written with two leaves
    /// one U+FEFF at the start of the decoded text — the doubled-BOM shape #35/#41/#42/#51 kept
    /// finding. Here it is `strip_bom` inside the parser that has to absorb it, so this asserts
    /// the two mechanisms compose rather than each assuming the other ran.
    ///
    /// **`package.json`, not `Cargo.toml`, and the choice is the whole test.** `serde_json` rejects
    /// any leading BOM, so `parse_package_json` genuinely depends on its `strip_bom` call. The
    /// `toml` crate strips one itself — `strip_bom`'s own doc records that calling it there is "a
    /// harmless no-op" — so a `Cargo.toml` fixture passes even with `strip_bom` deleted from the
    /// parser, and tests nothing. Verified by mutation: it did.
    #[test]
    fn a_doubled_bom_utf16_manifest_still_parses() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("package.json"),
            utf16le_with_bom("\u{FEFF}{\"name\":\"doubled\",\"version\":\"4.5.6\"}"),
        )
        .unwrap();

        let out = fingerprint(dir.path());
        assert!(
            out.contains("doubled") && out.contains("4.5.6"),
            "a doubled BOM must be absorbed by the parser's own strip:\n{out}"
        );
    }

    /// A decoded UTF-16 manifest must render exactly what the same NUL-free content renders in UTF-8.
    ///
    /// This is the property the change is really for, and the one an intuition about it gets
    /// wrong. Review of this work read the empty and nameless cases as regressions, because
    /// `FF FE` alone used to report `unusable: not UTF-8` and now reports a bare
    /// `manifest: Cargo.toml`. But an *empty UTF-8* `Cargo.toml` has always reported that bare
    /// line, and an empty UTF-8 `package.json` has always reported `malformed JSON` — measured on
    /// the parent commit, not assumed. So the change makes the two spellings agree, and a special
    /// case keeping UTF-16 on "not UTF-8" for an empty file would be both a new divergence and a
    /// false statement about a file that decoded fine.
    ///
    /// Asserted as equality of the whole `manifest:` line rather than as a substring, so a decode
    /// that mangles a value is caught as well as one that loses it. Rows are included whose UTF-8
    /// rendering is itself a failure or a blank, since those are exactly where "agrees with UTF-8"
    /// and "looks successful" come apart.
    ///
    /// **"NUL-free" in the name is a real bound, not hedging.** A body holding a raw NUL is the one
    /// class where the two encodings legitimately diverge: as UTF-8 it reaches the parser and
    /// reports `malformed TOML`/`malformed JSON`, while as UTF-16 `decode_utf16_with_bom` refuses it
    /// and reports `not UTF-8`. That refusal is deliberate — see the NUL test there — and the
    /// divergence class is exactly it, because the decode is otherwise byte-for-byte:
    /// `String::from_utf16` cannot fail on `str::encode_utf16` output, and the leading-BOM match is
    /// unambiguous in both orders even for a body that itself starts with U+FEFF. No manifest format
    /// permits a raw NUL, so nothing real sits in the excluded class.
    #[test]
    fn a_utf16_manifest_renders_what_its_utf8_spelling_renders() {
        let cases: &[(&str, &str)] = &[
            // Ordinary success.
            (
                "Cargo.toml",
                "[package]\nname = \"ordinary\"\nversion = \"1.0.0\"\n",
            ),
            // Valid TOML with no `[package]` — a workspace root. This rendered a bare line in both
            // encodings when #65 wrote the row, which is what made review read it as a regression;
            // #80 has since replaced that with `(workspace, 1 member)`. The row is unaffected
            // either way, and that is the point of asserting *parity* rather than a literal: it
            // pins the two encodings to each other, so a change to what the line says changes both
            // sides together and this keeps holding.
            ("Cargo.toml", "[workspace]\nmembers = [\"a\"]\n"),
            // Empty, and comment-only: both parse as valid, nameless TOML.
            ("Cargo.toml", ""),
            ("Cargo.toml", "# nothing here\n"),
            // Empty JSON is malformed JSON in either encoding.
            ("package.json", ""),
            // A `pyproject.toml` whose metadata lives in `setup.cfg`.
            (
                "pyproject.toml",
                "[build-system]\nrequires = [\"setuptools\"]\n",
            ),
        ];

        let manifest_line = |out: &str| {
            out.lines()
                .find(|l| l.trim_start().starts_with("manifest:"))
                .unwrap_or("<no manifest line>")
                .to_string()
        };

        for (manifest, body) in cases {
            for (order, encode) in [
                ("LE", utf16le_with_bom as fn(&str) -> Vec<u8>),
                ("BE", utf16be_with_bom as fn(&str) -> Vec<u8>),
            ] {
                let utf16_dir = tempfile::tempdir().unwrap();
                std::fs::write(utf16_dir.path().join(manifest), encode(body)).unwrap();
                std::fs::write(utf16_dir.path().join("m.rs"), "fn main() {}\n").unwrap();

                let utf8_dir = tempfile::tempdir().unwrap();
                std::fs::write(utf8_dir.path().join(manifest), body).unwrap();
                std::fs::write(utf8_dir.path().join("m.rs"), "fn main() {}\n").unwrap();

                assert_eq!(
                    manifest_line(&fingerprint(utf16_dir.path())),
                    manifest_line(&fingerprint(utf8_dir.path())),
                    "{manifest} in UTF-16 {order} must render what its NUL-free UTF-8 spelling \
                     renders (body {body:?})"
                );
            }
        }
    }

    /// UTF-16 LE *without* a BOM is valid UTF-8 — NUL is a legal UTF-8 byte — so it sails
    /// past the read check and reaches the parser as NUL-interleaved text.
    ///
    /// **All four formats now report the encoding rather than their parser's reason (#79.)**
    /// Before, each blamed whatever it happened to notice: `malformed TOML` / `malformed JSON`
    /// for three of them, and `no module directive` for `go.mod` — which escaped only because
    /// #43 had given it a guard of its own, for exactly this input. Every one of those sends the
    /// reader after a syntax error in a file whose syntax is fine.
    ///
    /// `go.mod` was brought onto the shared reason rather than keeping its own, which #79 left as
    /// a decision. It is the more accurate of the two: the file is not missing a module directive,
    /// it is in the wrong encoding, and the directive is right there once decoded. `no module
    /// directive` still covers the case it was written for — a NUL-free `go.mod` that genuinely
    /// declares nothing — which `a_go_mod_with_no_module_line_still_reports_its_own_reason` pins
    /// so the guard cannot be quietly absorbed by this one.
    ///
    /// PowerShell's `-Encoding Unicode` always writes a BOM so this exact spelling needs a writer
    /// that deliberately omits one, but any manifest carrying a stray NUL takes the same path.
    #[test]
    fn a_bomless_utf16_manifest_reports_the_encoding_not_the_syntax() {
        for (manifest, body, sibling, sibling_body) in [
            (
                "go.mod",
                "module github.com/acme/widget\n",
                "main.go",
                "package main\n\nfunc main() {}\n",
            ),
            (
                "Cargo.toml",
                "[package]\nname = \"w\"\nversion = \"1.0.0\"\n",
                "m.rs",
                "fn main() {}\n",
            ),
            (
                "package.json",
                "{\"name\":\"w\",\"version\":\"1.0.0\"}",
                "index.js",
                "export const x = 1;\n",
            ),
            (
                "pyproject.toml",
                "[project]\nname = \"w\"\n",
                "m.py",
                "def a():\n    pass\n",
            ),
        ] {
            // Written as explicit UTF-16 LE bytes with **no** BOM, per #35/#41.
            let mut bytes = Vec::new();
            for unit in body.encode_utf16() {
                bytes.extend_from_slice(&unit.to_le_bytes());
            }
            // Load-bearing: if this failed, the fixture would exercise #65's decode path instead
            // of this one, and the test would prove something else entirely.
            assert!(
                std::str::from_utf8(&bytes).is_ok(),
                "{manifest} fixture must be valid UTF-8, or it proves the wrong thing"
            );

            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join(manifest), &bytes).unwrap();
            std::fs::write(dir.path().join(sibling), sibling_body).unwrap();

            let out = fingerprint(dir.path());
            assert!(
                out.contains(&format!(
                    "manifest: {manifest} — unusable: NUL bytes, probably UTF-16 without a BOM"
                )),
                "{manifest} in BOM-less UTF-16 must name the encoding, not the syntax:\n{out}"
            );
        }
    }

    /// The reason #79 traded against: ordinary malformed input must keep its parser's reason.
    ///
    /// This is the common case, and it is what a NUL check applied too broadly would have cost.
    /// A truncated `{"name":` holds no NUL, so it is still a syntax error and still says so —
    /// if this ever reports the encoding reason, the check has started guessing.
    #[test]
    fn a_malformed_but_valid_utf8_manifest_keeps_its_parser_reason() {
        for (manifest, body, reason) in [
            ("package.json", "{\"name\":", "malformed JSON"),
            ("Cargo.toml", "[package\nname = \"x\"\n", "malformed TOML"),
            (
                "pyproject.toml",
                "[project\nname = \"x\"\n",
                "malformed TOML",
            ),
        ] {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join(manifest), body).unwrap();
            assert!(
                std::str::from_utf8(body.as_bytes()).is_ok() && !body.contains('\0'),
                "the fixture must be valid UTF-8 with no NUL, or it tests the other branch"
            );

            let out = fingerprint(dir.path());
            assert!(
                out.contains(&format!("manifest: {manifest} — unusable: {reason}")),
                "a genuinely malformed {manifest} must keep `{reason}`:\n{out}"
            );
        }
    }

    /// #43's `go.mod` guard still covers the case it was written for.
    ///
    /// #79 moved BOM-less UTF-16 onto the shared encoding reason, which was most of what this
    /// guard used to catch. Pinned separately so that move cannot quietly absorb it: a NUL-free
    /// `go.mod` that declares neither a module nor a require is a different failure and keeps a
    /// different reason.
    #[test]
    fn a_go_mod_with_no_module_line_still_reports_its_own_reason() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("go.mod"), "// just a comment\n").unwrap();
        std::fs::write(
            dir.path().join("main.go"),
            "package main\n\nfunc main() {}\n",
        )
        .unwrap();

        let out = fingerprint(dir.path());
        assert!(
            out.contains("manifest: go.mod — unusable: no module directive"),
            "a NUL-free go.mod declaring nothing must keep its own reason:\n{out}"
        );
    }

    /// #80: a manifest that parses but names nothing must say what it *is*.
    ///
    /// The bare `manifest: Cargo.toml` this replaces reads as "this project has no name", which
    /// is a claim tilth cannot make — and it was almost never what the file meant. The first two
    /// rows are the ones that matter: **a workspace-root `Cargo.toml` has no `[package]` section
    /// at all**, so every Cargo workspace hit this at its repository root, which is exactly where
    /// an agent launches.
    ///
    /// Asserted as whole-line equality rather than `contains`, because the defect being fixed is
    /// the *absence* of a suffix — a substring check against `manifest: Cargo.toml` matches the
    /// broken output and every fixed one equally, and would pass with the change reverted.
    #[test]
    fn a_nameless_manifest_says_what_it_is() {
        let cases: &[(&str, &str, &str)] = &[
            // A virtual workspace root: the single most common way to reach this.
            (
                "Cargo.toml",
                "[workspace]\nmembers = [\"a\", \"b\", \"c\"]\n",
                "  manifest: Cargo.toml (workspace, 3 members)",
            ),
            // `members` may be absent; still worth saying it is a workspace.
            (
                "Cargo.toml",
                "[workspace]\n",
                "  manifest: Cargo.toml (workspace)",
            ),
            // Singular, because "1 members" is the kind of detail that makes output look generated.
            (
                "Cargo.toml",
                "[workspace]\nmembers = [\"only\"]\n",
                "  manifest: Cargo.toml (workspace, 1 member)",
            ),
            // Empty and comment-only are *valid* TOML, so they parse and would otherwise render
            // identically to the workspace root above. Distinguishing them is #80's second half.
            ("Cargo.toml", "", "  manifest: Cargo.toml (empty)"),
            (
                "Cargo.toml",
                "# nothing here\n\n   \n",
                "  manifest: Cargo.toml (empty)",
            ),
            // Parses, holds real content, still nameless — the third category, which must be
            // distinguishable from both of the above.
            (
                "Cargo.toml",
                "[dependencies]\nserde = \"1\"\n",
                "  manifest: Cargo.toml (no name declared)",
            ),
            // The ordinary Python layout with metadata in `setup.cfg`.
            (
                "pyproject.toml",
                "[build-system]\nrequires = [\"setuptools\"]\n",
                "  manifest: pyproject.toml (build-system only)",
            ),
            ("pyproject.toml", "", "  manifest: pyproject.toml (empty)"),
            // Valid JSON, no name. An *empty* package.json is `malformed JSON`, so `(empty)` is
            // unreachable for this format — see `Nameless::Empty`.
            (
                "package.json",
                "{}",
                "  manifest: package.json (no name declared)",
            ),
        ];

        for (manifest, body, expected) in cases {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join(manifest), body).unwrap();

            let out = fingerprint(dir.path());
            let line = out
                .lines()
                .find(|l| l.trim_start().starts_with("manifest:"))
                .unwrap_or("<no manifest line>");
            assert_eq!(
                line, *expected,
                "nameless {manifest} (body {body:?}) rendered the wrong line"
            );
        }
    }

    /// A manifest that declares a name must never be labelled by what it lacks.
    ///
    /// `(build-system only)` was a *false* claim on every Poetry 1.x project: `[build-system]` is
    /// effectively mandatory there, Poetry keeps its metadata under `[tool.poetry]` rather than
    /// `[project]`, and testing `build-system` alone matched both. That is worse than the bare line
    /// #80 replaced — vague became wrong — and it is exactly the trade the issue warned against
    /// when it argued the failure marker would mislabel a valid manifest.
    ///
    /// The `[project]`-without-`name` rows are the same defect from the other side: a manifest that
    /// has a `[project]` table is not "build-system only", whatever it does or does not name.
    #[test]
    fn a_pyproject_is_not_called_build_system_only_when_it_is_not() {
        let cases: &[(&str, &str)] = &[
            // The ordinary Poetry layout: name and version live under `[tool.poetry]`.
            (
                "[tool.poetry]\nname = \"widget\"\nversion = \"0.1.0\"\n\n\
                 [build-system]\nrequires = [\"poetry-core\"]\n",
                "  manifest: pyproject.toml (widget v0.1.0)",
            ),
            // Poetry without a version still resolves a name.
            (
                "[tool.poetry]\nname = \"widget\"\n\n[build-system]\nrequires = [\"poetry-core\"]\n",
                "  manifest: pyproject.toml (widget)",
            ),
            // `[project]` wins when both are present — PEP 621 is the standard spelling.
            (
                "[project]\nname = \"standard\"\n\n[tool.poetry]\nname = \"legacy\"\n",
                "  manifest: pyproject.toml (standard)",
            ),
            // A `[project]` table with no name is not "build-system only" either.
            (
                "[project]\nversion = \"1.0\"\n\n[build-system]\nrequires = [\"setuptools\"]\n",
                "  manifest: pyproject.toml (no name declared)",
            ),
            // A `[tool.*]` section that is not poetry must not be mistaken for one — and must not
            // stop the build-system verdict either. Black config is formatter settings, not
            // packaging metadata, so this really is a `setup.cfg` project and the label holds.
            (
                "[tool.black]\nline-length = 88\n\n[build-system]\nrequires = [\"setuptools\"]\n",
                "  manifest: pyproject.toml (build-system only)",
            ),
            // And the case the variant is actually for: nothing but a build backend.
            (
                "[build-system]\nrequires = [\"setuptools\"]\n",
                "  manifest: pyproject.toml (build-system only)",
            ),
        ];

        for (body, expected) in cases {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join("pyproject.toml"), body).unwrap();

            let out = fingerprint(dir.path());
            let line = out
                .lines()
                .find(|l| l.trim_start().starts_with("manifest:"))
                .unwrap_or("<no manifest line>");
            assert_eq!(line, *expected, "pyproject body {body:?}");
        }
    }

    /// Reading a field for its presence must not change whether an unrelated manifest parses.
    ///
    /// #80 added `workspace` to `CargoToml` and `build-system` to `PyProject` purely to test for
    /// presence. Typed as tables, they made the **whole manifest** fail for a spelling that used to
    /// parse — measured against the parent commit, each of these rendered its name before and
    /// `— unusable: malformed TOML` after, losing the name, version and `deps:` line together.
    ///
    /// The inputs are invalid per their own format specs, so nothing real broke; the point is that
    /// the safety argument for #80 is "the change lives in the `name: None` branch", and a typed
    /// field moves failures *outside* that branch. Read as `toml::Value` they cannot.
    #[test]
    fn a_presence_only_field_cannot_make_a_named_manifest_fail_to_parse() {
        let cases: &[(&str, &str, &str)] = &[
            (
                "Cargo.toml",
                "workspace = \"..\"\n[package]\nname = \"widget\"\n",
                "  manifest: Cargo.toml (widget)",
            ),
            (
                "Cargo.toml",
                "[package]\nname = \"widget\"\n[workspace]\nmembers = \"a\"\n",
                "  manifest: Cargo.toml (widget)",
            ),
            (
                "Cargo.toml",
                "[package]\nname = \"widget\"\n[workspace]\nmembers = [1, 2]\n",
                "  manifest: Cargo.toml (widget)",
            ),
            (
                "pyproject.toml",
                "build-system = \"setuptools\"\n[project]\nname = \"widget\"\n",
                "  manifest: pyproject.toml (widget)",
            ),
        ];

        for (manifest, body, expected) in cases {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join(manifest), body).unwrap();

            let out = fingerprint(dir.path());
            let line = out
                .lines()
                .find(|l| l.trim_start().starts_with("manifest:"))
                .unwrap_or("<no manifest line>");
            assert_eq!(line, *expected, "{manifest} body {body:?}");
        }
    }

    /// The property that makes #80 safe: a manifest that *has* a name renders exactly as before.
    ///
    /// The whole change lives in the `name: None` branch, so this is true by construction — and
    /// pinned anyway, because "by construction" is the claim, not the evidence. `go.mod` is
    /// included even though it cannot reach the nameless rendering, so the row set covers all
    /// four formats rather than only the three that can.
    #[test]
    fn a_named_manifest_line_is_unchanged() {
        let cases: &[(&str, &str, &str)] = &[
            (
                "Cargo.toml",
                "[package]\nname = \"widget\"\nversion = \"1.2.3\"\n",
                "  manifest: Cargo.toml (widget v1.2.3)",
            ),
            // Name without version: the parenthetical closes straight after the name.
            (
                "Cargo.toml",
                "[package]\nname = \"widget\"\n",
                "  manifest: Cargo.toml (widget)",
            ),
            // A workspace root that *also* declares a package still renders the package — the
            // `[workspace]` table must not hijack a manifest that has a name.
            (
                "Cargo.toml",
                "[package]\nname = \"widget\"\nversion = \"1.2.3\"\n[workspace]\nmembers = [\"a\"]\n",
                "  manifest: Cargo.toml (widget v1.2.3)",
            ),
            (
                "package.json",
                "{\"name\":\"widget\",\"version\":\"1.2.3\"}",
                "  manifest: package.json (widget v1.2.3)",
            ),
            (
                "pyproject.toml",
                "[project]\nname = \"widget\"\nversion = \"1.2.3\"\n[build-system]\nrequires = [\"setuptools\"]\n",
                "  manifest: pyproject.toml (widget v1.2.3)",
            ),
            (
                "go.mod",
                "module github.com/acme/widget\n",
                "  manifest: go.mod (github.com/acme/widget)",
            ),
        ];

        for (manifest, body, expected) in cases {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join(manifest), body).unwrap();

            let out = fingerprint(dir.path());
            let line = out
                .lines()
                .find(|l| l.trim_start().starts_with("manifest:"))
                .unwrap_or("<no manifest line>");
            assert_eq!(
                line, *expected,
                "named {manifest} (body {body:?}) must render exactly as it did before #80"
            );
        }
    }

    /// No manifest line may ever be bare — the invariant, rather than the nine cases above.
    ///
    /// `Nameless` is not an `Option` precisely so this cannot regress, but the type only forces a
    /// parser to *decide*; it cannot stop a future variant rendering an empty label. This asserts
    /// the observable property directly: whatever a manifest turns out to be, the line says more
    /// than the file name.
    #[test]
    fn no_manifest_line_is_ever_bare() {
        let bodies: &[(&str, &str)] = &[
            ("Cargo.toml", ""),
            ("Cargo.toml", "[workspace]\n"),
            ("Cargo.toml", "# comment\n"),
            ("Cargo.toml", "[dependencies]\nserde = \"1\"\n"),
            ("Cargo.toml", "[package]\nname = \"n\"\n"),
            ("Cargo.toml", "[package\nbroken\n"),
            ("package.json", "{}"),
            ("package.json", "{\"name\":\"n\"}"),
            ("package.json", "{oops"),
            ("pyproject.toml", ""),
            ("pyproject.toml", "[build-system]\nrequires = []\n"),
            ("pyproject.toml", "[tool.black]\nline-length = 88\n"),
            ("go.mod", "module m\n"),
            ("go.mod", "// nothing\n"),
        ];

        for (manifest, body) in bodies {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join(manifest), body).unwrap();

            let out = fingerprint(dir.path());
            let line = out
                .lines()
                .find(|l| l.trim_start().starts_with("manifest:"))
                .unwrap_or("<no manifest line>")
                .trim()
                .to_string();
            assert_ne!(
                line,
                format!("manifest: {manifest}"),
                "a bare manifest line reads as `this project has no name` (body {body:?})"
            );
        }
    }

    /// The reason must not depend on which errno the platform chose.
    ///
    /// `find_manifest` uses `Path::exists()`, which is true for directories, so a *directory*
    /// named `package.json` reaches `read_manifest`. `fs::read` reports that as
    /// `PermissionDenied` on Windows and `IsADirectory` on Linux — so deriving the reason
    /// from `ErrorKind` alone both diverged across platforms and told a Windows agent to go
    /// fix file modes for a problem that is "this is a directory". CI is Linux-only while
    /// development here is Windows, which is why
    /// `path_bearing_lines_are_identical_across_platforms` exists; this is the same hazard
    /// in the same payload.
    ///
    /// Only the `ErrorKind`-derived reasons need a row here — those are the ones an OS can spell
    /// differently. #79's `NUL bytes, probably UTF-16 without a BOM` is decided by
    /// `str::contains` over bytes already in memory, so no errno reaches it; it is pinned as an
    /// exact string by `a_bomless_utf16_manifest_reports_the_encoding_not_the_syntax` instead.
    #[test]
    fn unreadable_reasons_do_not_vary_by_platform() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("package.json")).unwrap();
        std::fs::write(dir.path().join("index.js"), "export const x = 1;\n").unwrap();

        let out = fingerprint(dir.path());
        assert!(
            out.contains("manifest: package.json — unusable: not a file"),
            "a directory named package.json must report the same reason everywhere:\n{out}"
        );
    }

    /// The same visibility rule for a manifest that reads fine but does not parse.
    ///
    /// This shared the `.ok()?` shape and so shared the silence. Kept distinct from the
    /// encoding case because the reasons differ and an agent acts on them differently —
    /// "not UTF-8" is a re-save, "malformed JSON" is a syntax error to go and find.
    #[test]
    fn a_malformed_manifest_says_so_rather_than_vanishing() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("package.json"), "{\"name\": oops,,,}").unwrap();
        std::fs::write(dir.path().join("index.js"), "export const x = 1;\n").unwrap();

        let out = fingerprint(dir.path());
        assert!(
            out.contains("manifest: package.json — unusable: malformed JSON"),
            "a malformed manifest must say so:\n{out}"
        );
    }

    /// The `toml` crate strips exactly *one* BOM, so a doubled one still fails to parse and
    /// `.ok()?` drops the entire manifest block — the same silent loss as the `package.json`
    /// bug, reached through the format that was supposed to be immune.
    ///
    /// A doubled BOM is what a tool that prepends one without checking for an existing one
    /// produces; both BOM helpers already strip repeats for exactly this reason, so leaving
    /// these two parsers out would have been an inconsistency rather than a scope decision.
    #[test]
    fn doubled_bom_toml_manifests_keep_their_manifest_block() {
        let doubled = [UTF8_BOM, UTF8_BOM].concat();

        // (manifest, body, needles that must survive)
        let cases: &[(&str, &str, &[&str])] = &[
            (
                "Cargo.toml",
                "[package]\nname = \"demo-crate\"\nversion = \"0.4.2\"\n\n\
                 [dependencies]\nserde = \"1\"\n",
                &["demo-crate", "0.4.2", "serde"],
            ),
            (
                "pyproject.toml",
                "[project]\nname = \"demo-pkg\"\nversion = \"1.2.3\"\n\
                 dependencies = [\"httpx\"]\n",
                &["demo-pkg", "1.2.3", "httpx"],
            ),
        ];

        for (manifest, body, needles) in cases {
            let mut outs = Vec::new();
            for prefix in [&[][..], &doubled] {
                let dir = tempfile::tempdir().unwrap();
                write_with_bom(&dir.path().join(manifest), prefix, body);
                let src = dir.path().join("src");
                std::fs::create_dir_all(&src).unwrap();
                std::fs::write(src.join("main.rs"), "pub fn a() {}\n").unwrap();
                outs.push(fingerprint(dir.path()));
            }

            for needle in *needles {
                assert!(
                    outs[0].contains(needle),
                    "fixture is broken: unmarked {manifest} lost {needle}:\n{}",
                    outs[0]
                );
                assert!(
                    outs[1].contains(needle),
                    "a doubled BOM on {manifest} lost {needle} from the fingerprint:\n{}",
                    outs[1]
                );
            }
            assert_eq!(
                outs[1], outs[0],
                "a doubled BOM changed the fingerprint for {manifest}"
            );
        }
    }

    /// `line.trim()` is not BOM-aware, so `parse_go_mod` reported no module name at all for
    /// a BOM'd `go.mod` — literally the import-detection bug, in a file that fix never
    /// visited. `require` entries were unaffected only because they never sit on line 1,
    /// which is why the fixture asserts on both.
    #[test]
    fn bom_go_mod_keeps_the_module_name() {
        let body = "module github.com/acme/widget\n\ngo 1.21\n\n\
                    require (\n\tgithub.com/gin-gonic/gin v1.9.0\n)\n";

        let mut outs = Vec::new();
        for prefix in [&[][..], UTF8_BOM] {
            let dir = tempfile::tempdir().unwrap();
            write_with_bom(&dir.path().join("go.mod"), prefix, body);
            std::fs::write(
                dir.path().join("main.go"),
                "package main\n\nfunc main() {}\n",
            )
            .unwrap();
            outs.push(fingerprint(dir.path()));
        }

        for needle in ["github.com/acme/widget", "gin"] {
            assert!(
                outs[0].contains(needle),
                "fixture is broken: unmarked go.mod lost {needle}:\n{}",
                outs[0]
            );
            assert!(
                outs[1].contains(needle),
                "a BOM'd go.mod lost {needle} from the fingerprint:\n{}",
                outs[1]
            );
        }
        assert_eq!(outs[1], outs[0], "a BOM changed the fingerprint");
    }

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

    /// A C++ tree whose headers are reached through an include root: `include/proj/util.h`
    /// included as `"proj/util.h"` from sources in `src/`. `importers` sources include it, so
    /// it clears `hot_files`' "at least two importers" bar.
    ///
    /// Returns nothing — the caller decides where `.git` goes, which is the whole variable.
    fn write_include_root_cpp_tree(root: &Path, importers: usize) {
        let inc = root.join("include/proj");
        let src = root.join("src");
        std::fs::create_dir_all(&inc).unwrap();
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(inc.join("util.h"), "#pragma once\nvoid util();\n").unwrap();
        for i in 0..importers {
            std::fs::write(
                src.join(format!("m{i}.cc")),
                "#include \"proj/util.h\"\n\nvoid m() { util(); }\n",
            )
            .unwrap();
        }
    }

    fn hot_line(out: &str) -> Option<&str> {
        out.lines().find(|l| l.trim_start().starts_with("hot ("))
    }

    /// A tree that is not a git checkout must still resolve include-root headers (#45).
    ///
    /// This is the reported bug. `hot_files` passed no boundary at all, so `resolve_c_include`
    /// fell to `enclosing_repo_root(dir)?` — which returns `None` outside a checkout and
    /// short-circuits before probing anything. Every project-relative include bucketed as
    /// external and the hot-file counts silently under-reported, the same blindness #15
    /// removed from the post-read hint, callee resolution, `deps`, `grok` and expanded search
    /// and that #44 deliberately left here pending measurement.
    ///
    /// Deliberately no `.git` anywhere above the fixture either: `tempfile::tempdir` lands in
    /// the system temp directory, so there is no repository to accidentally rescue it. That is
    /// asserted rather than assumed, because if a `.git` ever did sit above the temp root this
    /// test would pass without the fix and prove nothing.
    #[test]
    fn hot_files_in_a_non_git_tree_resolves_include_root_headers() {
        let dir = tempfile::tempdir().unwrap();
        write_include_root_cpp_tree(dir.path(), 3);

        let mut probe = Some(dir.path());
        while let Some(d) = probe {
            assert!(
                !d.join(".git").exists(),
                "a .git at {} makes this fixture a git tree, so it cannot test the non-git path",
                d.display()
            );
            probe = d.parent();
        }

        let out = fingerprint(dir.path());
        let hot = hot_line(&out).unwrap_or_else(|| {
            panic!("no hot line in a fingerprint of a 3-importer include-root tree:\n{out}")
        });
        assert!(
            hot.contains("util"),
            "the include-root header was not counted in a non-git tree — resolution gave up \
             before probing (#45):\n{hot}"
        );
    }

    /// Where there is no `.git`, a sibling project under the same root *can* be reached — and
    /// that is stated rather than left implied.
    ///
    /// This is the honest limit of the fix, and the counterpart to
    /// `hot_files_launched_above_two_checkouts_does_not_cross_repositories`: refusing to cross
    /// into another project is a claim about **git** trees, where the repository supplies a
    /// boundary. Outside one the tree root is the only boundary that exists, so `proj_a` can
    /// resolve an include that lands in `proj_b`. Review flagged this as an unscoped claim, so
    /// it gets a fixture: the behaviour is deliberate, it is the same reach a declared scope has
    /// always had, and the alternative is #45's reported bug of resolving nothing at all.
    ///
    /// If this ever becomes unacceptable, the fix is a real project boundary — a manifest, a
    /// build file — not a narrower default, because narrowing it is what #45 undid.
    #[test]
    fn hot_files_in_a_non_git_tree_can_reach_a_sibling_project() {
        let parent = tempfile::tempdir().unwrap();
        let proj_a = parent.path().join("proj_a/src");
        std::fs::create_dir_all(&proj_a).unwrap();
        std::fs::create_dir_all(parent.path().join("proj_b/shared")).unwrap();
        std::fs::write(
            parent.path().join("proj_b/shared/util.h"),
            "#pragma once\nvoid other();\n",
        )
        .unwrap();
        for i in 0..3 {
            std::fs::write(
                proj_a.join(format!("m{i}.cc")),
                "#include \"proj_b/shared/util.h\"\n\nvoid m() {}\n",
            )
            .unwrap();
        }

        let out = fingerprint(parent.path());
        let hot = hot_line(&out)
            .unwrap_or_else(|| panic!("no hot line in a non-git two-project tree:\n{out}"));
        assert!(
            hot.contains("proj_b"),
            "documented behaviour changed: with no `.git` the tree root is the only boundary, \
             so a sibling project is reachable. If this is now refused, say so here and in the \
             note at the `boundary_from_file` call site:\n{hot}"
        );
    }

    /// Launched *inside* a subdirectory, headers above the launch directory must still count.
    ///
    /// This is the first reason `root` is not simply the fix. `fingerprint` is only ever called
    /// with the process cwd, so every candidate is `root.join(rel)` and `root` would contain all
    /// of them — the declared-scope arm of `resolve_c_include`'s composition rule would always
    /// win and `.git` would never apply. Launched at `<repo>/src`, `<repo>/include/proj/util.h`
    /// sits *above* the boundary and every resolution would be refused: a silent change to
    /// fingerprint output in git trees, which are the common case.
    ///
    /// `boundary_from_file` gives the repository instead, so the answer does not depend on where
    /// the server happened to be started.
    #[test]
    fn hot_files_launched_in_a_subdirectory_still_reaches_headers_above_it() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".git")).unwrap();
        write_include_root_cpp_tree(dir.path(), 3);

        let out = fingerprint(&dir.path().join("src"));
        let hot = hot_line(&out)
            .unwrap_or_else(|| panic!("no hot line when launched inside src/:\n{out}"));
        // Matched loosely on purpose: a target *above* the launch directory fails
        // `strip_prefix(root)` in the renderer and falls through to the absolute path, so this
        // line names a host path rather than a relative one. Pre-existing and unrelated to the
        // boundary — before #45 the same launch resolved to the same place via `.git` — but this
        // is the first test to reach it, so it is recorded rather than silently ratified.
        assert!(
            hot.contains("util"),
            "a header above the launch directory stopped being counted, so the boundary came \
             from the launch dir rather than from the file (#45):\n{hot}"
        );
    }

    /// Launched *above* two **checkouts**, resolution must not cross from one into the other.
    ///
    /// Scoped to git trees on purpose — `hot_files_in_a_non_git_tree_can_reach_a_sibling_project`
    /// is the same shape without the `.git` directories, and states the opposite outcome.
    ///
    /// The second reason `root` is not the fix, and the one worth caring about: a parent
    /// directory holding several repositories is ordinary with worktrees. `repo_a/src/*.cc`
    /// includes `"repo_b/shared/util.h"`, which exists — so with the launch directory as the
    /// boundary the ancestor walk reaches it at hop `parent` and the fingerprint counts a hot
    /// file in a *different project*. Given #29/#31 went to some trouble to make the initialize
    /// fingerprint say true things, that is the failure mode to refuse.
    ///
    /// The `repo_a`-local arm is the control: the same launch must still count `repo_a`'s own
    /// include-root header, or this test would also pass against an implementation that simply
    /// resolved nothing.
    #[test]
    fn hot_files_launched_above_two_checkouts_does_not_cross_repositories() {
        let parent = tempfile::tempdir().unwrap();
        let repo_a = parent.path().join("repo_a");
        let repo_b = parent.path().join("repo_b");
        std::fs::create_dir_all(repo_a.join(".git")).unwrap();
        std::fs::create_dir_all(repo_b.join(".git")).unwrap();
        write_include_root_cpp_tree(&repo_a, 0);
        std::fs::create_dir_all(repo_b.join("shared")).unwrap();
        std::fs::write(
            repo_b.join("shared/util.h"),
            "#pragma once\nvoid other();\n",
        )
        .unwrap();

        // Each source includes repo_a's own header (must count) and repo_b's (must not).
        for i in 0..3 {
            std::fs::write(
                repo_a.join("src").join(format!("m{i}.cc")),
                "#include \"proj/util.h\"\n#include \"repo_b/shared/util.h\"\n\nvoid m() {}\n",
            )
            .unwrap();
        }

        let out = fingerprint(parent.path());
        let hot = hot_line(&out)
            .unwrap_or_else(|| panic!("no hot line when launched above two checkouts:\n{out}"));
        assert!(
            hot.contains("include") && hot.contains("proj"),
            "control failed: repo_a's own include-root header was not counted, so the \
             cross-repository assertion below proves nothing:\n{hot}"
        );
        assert!(
            !hot.contains("repo_b"),
            "resolution crossed a repository boundary — the fingerprint is claiming a hot file \
             in a different project (#45):\n{hot}"
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
