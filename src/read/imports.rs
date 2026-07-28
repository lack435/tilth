//! Resolve import statements to local file paths.
//! Used by the MCP layer to hint related files after an outlined read.

use std::fs;
use std::path::{Path, PathBuf};

use crate::lang::detect_file_type;
use crate::types::{FileType, Lang};

const MAX_SUGGESTIONS: usize = 8;

/// Extract import sources from a code file and resolve them to existing local file paths.
/// Returns empty Vec for non-code files, files with no imports, or when all imports are external.
pub fn resolve_related_files(file_path: &Path) -> Vec<PathBuf> {
    let Ok(content) = fs::read_to_string(file_path) else {
        return Vec::new();
    };
    resolve_related_files_with_content(file_path, &content)
}

/// Same as `resolve_related_files` but takes pre-read content to avoid a redundant file read.
pub fn resolve_related_files_with_content(file_path: &Path, content: &str) -> Vec<PathBuf> {
    let FileType::Code(lang) = detect_file_type(file_path) else {
        return Vec::new();
    };

    let Some(dir) = file_path.parent() else {
        return Vec::new();
    };

    let mut results = Vec::new();
    for line in content.lines() {
        if results.len() >= MAX_SUGGESTIONS {
            break;
        }
        if !is_import_line(line, lang) {
            continue;
        }
        let source = crate::lang::outline::extract_import_source(line, Some(lang));
        if source.is_empty() || is_external(&source, lang) {
            continue;
        }
        if let Some(path) = resolve_import_to_file(dir, &source, lang) {
            if !results.contains(&path) {
                results.push(path);
            }
        }
    }
    results
}

pub(crate) fn is_import_line(line: &str, lang: Lang) -> bool {
    let trimmed = line.trim_start();
    match lang {
        Lang::Rust => trimmed.starts_with("use "),
        Lang::TypeScript | Lang::Tsx | Lang::JavaScript => {
            trimmed.starts_with("import ") || trimmed.starts_with("import{")
        }
        Lang::Python => trimmed.starts_with("import ") || trimmed.starts_with("from "),
        Lang::Go | Lang::Java | Lang::Scala | Lang::Kotlin => trimmed.starts_with("import "),
        Lang::C | Lang::Cpp => trimmed.starts_with("#include"),
        Lang::Elixir => {
            trimmed.starts_with("alias ")
                || trimmed.starts_with("import ")
                || trimmed.starts_with("use ")
                || trimmed.starts_with("require ")
        }
        Lang::Bash => trimmed
            .strip_prefix("source")
            .or_else(|| trimmed.strip_prefix('.'))
            .is_some_and(|rest| rest.starts_with(char::is_whitespace)),
        _ => false,
    }
}

pub(crate) fn is_external(source: &str, lang: Lang) -> bool {
    match lang {
        Lang::Rust => {
            !(source.starts_with("crate::")
                || source.starts_with("self::")
                || source.starts_with("super::"))
        }
        Lang::TypeScript | Lang::Tsx | Lang::JavaScript => {
            !(source.starts_with('.') || source.starts_with("@/") || source.starts_with("~/"))
        }
        // Bash: dot-relative paths are local; anything else (bare name, /abs/path) is external.
        Lang::Python | Lang::Bash => !source.starts_with('.'),
        Lang::C | Lang::Cpp => !source.starts_with('"'),
        // Elixir, Go, Java, Scala, Kotlin — can't resolve without build system knowledge.
        _ => true,
    }
}

/// Resolve an import source to an existing file on disk, or `None` when it does not
/// name one. Shared with `search::deps`, which needs to distinguish "resolved to a
/// local file" from "names something we cannot see" rather than only collecting hits.
pub(crate) fn resolve_import_to_file(dir: &Path, source: &str, lang: Lang) -> Option<PathBuf> {
    let raw = match lang {
        Lang::Rust => resolve_rust(dir, source),
        Lang::TypeScript | Lang::Tsx | Lang::JavaScript => resolve_js(dir, source),
        Lang::Python => resolve_python(dir, source),
        Lang::C | Lang::Cpp => resolve_c_include(dir, source),
        Lang::Bash => resolve_bash(dir, source),
        // Elixir, Go, Java, etc. — module-to-file mapping requires build system conventions.
        _ => None,
    };
    raw.map(|p| normalize_path(&p))
}

/// Lexically collapse `.` and `..` components without touching the filesystem.
/// `dir.join("../foo")` returns a `PathBuf` containing literal `..`; without this,
/// distinct spellings of the same target file produce distinct `PathBuf`s and
/// downstream callers (dedup loops, `HashMap` keys) treat them as different files.
fn normalize_path(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                let pop_ok = matches!(
                    out.components().next_back(),
                    Some(Component::Normal(_) | Component::Prefix(_))
                );
                if pop_ok {
                    out.pop();
                } else {
                    out.push(comp);
                }
            }
            _ => out.push(comp),
        }
    }
    out
}

// --- Rust ---

fn resolve_rust(dir: &Path, source: &str) -> Option<PathBuf> {
    if let Some(rest) = source.strip_prefix("crate::") {
        let src_dir = find_src_ancestor(dir)?;
        try_rust_path(src_dir, rest)
    } else if let Some(rest) = source.strip_prefix("self::") {
        try_rust_path(dir, rest)
    } else if let Some(rest) = source.strip_prefix("super::") {
        try_rust_path(dir.parent()?, rest)
    } else {
        None
    }
}

/// Try progressively shorter paths until one resolves.
/// `cache::OutlineCache` → try cache/OutlineCache.rs (no) → cache.rs (yes).
/// `read::imports` → try read/imports.rs (yes) → stop.
fn try_rust_path(base: &Path, rest: &str) -> Option<PathBuf> {
    let segments: Vec<&str> = rest.split("::").collect();
    for len in (1..=segments.len()).rev() {
        let rel: PathBuf = segments[..len].iter().collect();
        if let Some(found) = try_rust_module(&base.join(&rel)) {
            return Some(found);
        }
    }
    None
}

fn try_rust_module(base: &Path) -> Option<PathBuf> {
    let with_rs = base.with_extension("rs");
    if with_rs.exists() {
        return Some(with_rs);
    }
    let mod_rs = base.join("mod.rs");
    if mod_rs.exists() {
        return Some(mod_rs);
    }
    None
}

fn find_src_ancestor(start: &Path) -> Option<&Path> {
    let mut current = start;
    loop {
        if current.file_name().and_then(|n| n.to_str()) == Some("src") {
            return Some(current);
        }
        current = current.parent()?;
    }
}

// --- JS/TS ---

fn resolve_js(dir: &Path, source: &str) -> Option<PathBuf> {
    let base = dir.join(source);
    // Try with extensions
    for ext in &[".ts", ".tsx", ".js", ".jsx"] {
        let candidate = PathBuf::from(format!("{}{ext}", base.display()));
        if candidate.exists() {
            return Some(candidate);
        }
    }
    // Already has extension
    if base.exists() && base.is_file() {
        return Some(base);
    }
    // Index files
    for name in &["index.ts", "index.tsx", "index.js", "index.jsx"] {
        let candidate = base.join(name);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

// --- Python ---

fn resolve_python(dir: &Path, source: &str) -> Option<PathBuf> {
    let dots = source.bytes().take_while(|&b| b == b'.').count();
    if dots == 0 {
        return None;
    }
    // Each dot beyond the first goes up one directory.
    let mut base = dir.to_path_buf();
    for _ in 1..dots {
        base = base.parent()?.to_path_buf();
    }
    let module_part = &source[dots..];
    if module_part.is_empty() {
        // Bare `from . import X`
        let init = base.join("__init__.py");
        return if init.exists() { Some(init) } else { None };
    }
    let rel = module_part.replace('.', "/");
    let as_file = base.join(format!("{rel}.py"));
    if as_file.exists() {
        return Some(as_file);
    }
    let as_pkg = base.join(&rel).join("__init__.py");
    if as_pkg.exists() {
        return Some(as_pkg);
    }
    None
}

// --- C/C++ ---

/// How far up to look for an include root. Bounded so a stray include cannot walk a
/// whole filesystem, and generous enough for the nesting real module layouts use
/// (`Source/<Module>/A/B/C/File.h` including `A/Other.h` is four hops).
const MAX_INCLUDE_ROOT_HOPS: usize = 8;

/// Resolve a quoted `#include` to a file, trying the including directory first and
/// then ancestor directories as candidate include roots.
///
/// A quoted include is only *sometimes* relative to the including file. Far more often
/// it is written relative to an include root — a module source directory, or a project's
/// `include/` — which a compiler learns from `-I` flags that live in the build system.
/// tilth has no build metadata, so it approximates by testing ancestors, nearest first.
///
/// This is what the two real cases look like, and one hop resolves both:
///   `include/leveldb/env.h`            includes `leveldb/status.h`
///     → `include/` + `leveldb/status.h`
///   `Source/Game/Character/Comp.h`     includes `Character/PawnComponent.h`
///     → `Source/Game/` + `Character/PawnComponent.h`
///
/// The ancestor walk only applies to multi-segment includes. A bare `"config.h"` is
/// either a sibling — already found by the first check — or something outside the
/// project, and walking up for it is how you match an unrelated same-named file several
/// directories away. Requiring a separator keeps the guess specific.
///
/// The walk also stops once it has tested a directory holding `.git`, so resolution
/// cannot wander out of the project.
fn resolve_c_include(dir: &Path, source: &str) -> Option<PathBuf> {
    let clean = source.trim_matches('"');

    // The standard `""` search: relative to the including file.
    let direct = dir.join(clean);
    if direct.is_file() {
        return Some(direct);
    }

    if !clean.contains('/') && !clean.contains('\\') {
        return None;
    }

    let mut base = dir;
    for _ in 0..MAX_INCLUDE_ROOT_HOPS {
        // Stop *before* stepping above the repo root. The root itself is still reachable
        // as a `parent` on an earlier iteration, so an include root at the top of the
        // repo resolves; what this prevents is testing paths outside the project.
        if base.join(".git").exists() {
            break;
        }
        let Some(parent) = base.parent() else { break };
        let candidate = parent.join(clean);
        if candidate.is_file() {
            return Some(candidate);
        }
        base = parent;
    }
    None
}

// --- Bash ---

fn resolve_bash(dir: &Path, source: &str) -> Option<PathBuf> {
    // Only resolve literal relative paths — no extension inference. A single
    // metadata() stat avoids the exists()+is_file() two-call TOCTOU; resolution
    // is best-effort, so a stale result only ever costs a related-file hint.
    let candidate = dir.join(source);
    std::fs::metadata(&candidate)
        .is_ok_and(|m| m.is_file())
        .then_some(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn normalize_collapses_dot_and_parent_components() {
        let p = Path::new("temporal/workflows/../utils/activityProxies.ts");
        assert_eq!(
            normalize_path(p),
            PathBuf::from("temporal/utils/activityProxies.ts")
        );
        let p = Path::new("app/db/./db.ts");
        assert_eq!(normalize_path(p), PathBuf::from("app/db/db.ts"));
    }

    #[test]
    fn normalize_preserves_leading_parent_when_unresolvable() {
        // No prior Normal component to pop, so ".." is kept.
        let p = Path::new("../outside.ts");
        assert_eq!(normalize_path(p), PathBuf::from("../outside.ts"));
    }

    #[test]
    fn js_resolve_returns_normalized_path_for_parent_import() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        fs::create_dir_all(root.join("temporal/workflows")).unwrap();
        fs::create_dir_all(root.join("temporal/utils")).unwrap();
        fs::write(root.join("temporal/utils/activityProxies.ts"), "").unwrap();

        let resolved = resolve_js(&root.join("temporal/workflows"), "../utils/activityProxies")
            .expect("should resolve");
        let normalized = normalize_path(&resolved);
        assert_eq!(normalized, root.join("temporal/utils/activityProxies.ts"));
        // No "../" component should survive normalization.
        assert!(
            !normalized
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir)),
            "normalized path still contains '..': {normalized:?}"
        );
    }

    #[test]
    fn js_resolve_dedups_different_spellings_of_same_file() {
        // Two importers of the same file via different relative paths must
        // produce the same PathBuf so that hot-file counting aggregates them.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        fs::create_dir_all(root.join("a/b")).unwrap();
        fs::create_dir_all(root.join("a/c")).unwrap();
        fs::write(root.join("a/b/target.ts"), "").unwrap();

        let from_sibling = resolve_import_to_file(&root.join("a/b"), "./target", Lang::TypeScript)
            .expect("sibling");
        let from_cousin =
            resolve_import_to_file(&root.join("a/c"), "../b/target", Lang::TypeScript)
                .expect("cousin");

        assert_eq!(
            from_sibling, from_cousin,
            "different spellings should normalize to the same PathBuf"
        );
    }

    /// A quoted include is usually written relative to an *include root*, not to the
    /// including file: `include/leveldb/env.h` includes `"leveldb/status.h"`, which lives
    /// at `include/leveldb/status.h`. Compilers learn those roots from `-I` flags; with
    /// no build metadata, ancestors are tested nearest-first.
    ///
    /// Before this, such an include resolved to nothing, so a header's project-local
    /// dependencies were reported as external.
    #[test]
    fn c_include_resolves_against_an_ancestor_include_root() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::create_dir_all(root.join("include/lib")).unwrap();
        fs::write(root.join("include/lib/status.h"), "struct S {};\n").unwrap();

        // Included from include/lib/env.h as "lib/status.h" — one hop up.
        let from = root.join("include/lib");
        let got = resolve_c_include(&from, "\"lib/status.h\"").expect("should resolve");
        assert_eq!(got, root.join("include/lib/status.h"));
    }

    #[test]
    fn c_include_prefers_the_including_directory() {
        // The standard `""` search still wins when the file is a sibling, even if an
        // ancestor also has a same-named path.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        fs::create_dir_all(root.join("a/b/a")).unwrap();
        fs::write(root.join("a/b/a/x.h"), "// near\n").unwrap();
        fs::write(root.join("a/x.h"), "// far\n").unwrap();

        let got = resolve_c_include(&root.join("a/b"), "\"a/x.h\"").expect("resolves");
        assert_eq!(got, root.join("a/b/a/x.h"), "nearest match must win");
    }

    #[test]
    fn c_include_does_not_walk_for_a_bare_filename() {
        // A single-segment include is either a sibling (found directly) or outside the
        // project. Walking up for it is how an unrelated same-named file several
        // directories away gets picked up, so it is deliberately not attempted.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        fs::create_dir_all(root.join("deep/nested")).unwrap();
        fs::write(root.join("config.h"), "// unrelated\n").unwrap();

        assert!(
            resolve_c_include(&root.join("deep/nested"), "\"config.h\"").is_none(),
            "a bare filename must not resolve to a same-named file up the tree"
        );
    }

    #[test]
    fn c_include_walk_stops_at_the_repo_root() {
        // Resolution must not wander out of the project.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        fs::create_dir_all(root.join("proj/.git")).unwrap();
        fs::create_dir_all(root.join("proj/src")).unwrap();
        // The only match is *outside* the repo, one level above `proj/`.
        fs::create_dir_all(root.join("outside")).unwrap();
        fs::write(root.join("outside/leak.h"), "// outside\n").unwrap();

        assert!(
            resolve_c_include(&root.join("proj/src"), "\"outside/leak.h\"").is_none(),
            "the walk must stop once it has tested the directory holding .git"
        );
    }

    #[test]
    fn bash_is_import_line_tab_separated() {
        // Tab between `source` and path is valid bash and must be detected.
        assert!(
            is_import_line("source\t./lib.sh", Lang::Bash),
            "source<TAB>./lib.sh should be detected as an import line"
        );
        // False positives: `sourcefile=1` looks like it starts with `source` but
        // has no whitespace separator.
        assert!(
            !is_import_line("sourcefile=1", Lang::Bash),
            "sourcefile=1 must not be detected as an import line"
        );
        // `./script.sh` is a script execution, not a source directive.
        assert!(
            !is_import_line("./script.sh", Lang::Bash),
            "./script.sh must not be detected as an import line"
        );
        // `.bashrc` — dot followed by non-whitespace, not a source directive.
        assert!(
            !is_import_line(".bashrc", Lang::Bash),
            ".bashrc must not be detected as an import line"
        );
    }
}
