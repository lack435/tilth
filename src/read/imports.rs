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
///
/// Capped at `MAX_SUGGESTIONS` — this feeds the "related files" hint after a read, where
/// a short list is the point. Callers that need the *complete* set (dependency analysis)
/// must use `resolve_local_imports`; truncating there loses dependencies entirely, since
/// an import that resolved is also excluded from the external bucket.
///
/// Passes no boundary, so C/C++ include-root resolution here still depends on finding a
/// `.git` ancestor. A read has no declared scope to use instead. The consequence is
/// narrow and known: in a non-git tree the post-read "Related:" hint, and the callee
/// resolution in `search::callees`, miss include-root-relative headers that `tilth_deps`
/// now finds. deps still lists the file — the import-based merge registers it — just
/// without per-symbol detail. Fixing it means threading a scope through both call paths.
pub fn resolve_related_files_with_content(file_path: &Path, content: &str) -> Vec<PathBuf> {
    let mut resolved = resolve_local_imports(file_path, content, None);
    resolved.truncate(MAX_SUGGESTIONS);
    resolved
}

/// Every import in `content` that resolves to a local file. Uncapped.
///
/// `resolve_related_files_with_content` is the capped view for display. Dependency
/// analysis needs all of them: it classifies an import as external only when it does
/// *not* resolve, so an import dropped by a cap here would appear in neither the local
/// nor the external list and vanish from the report.
///
/// `boundary` is the caller's declared search scope, used to confine C/C++ include-root
/// resolution. It must already be canonical — build it with `canonical_boundary` — because
/// a caller that also classifies non-resolving imports has to ask the *same* question here
/// and there. `search::deps` does exactly that, and an include that resolves under one
/// boundary but not the other lands in both of its buckets or in neither. Pass `None` when
/// there is no scope; see `resolve_c_include`.
pub(crate) fn resolve_local_imports(
    file_path: &Path,
    content: &str,
    boundary: Option<&Path>,
) -> Vec<PathBuf> {
    let FileType::Code(lang) = detect_file_type(file_path) else {
        return Vec::new();
    };

    let Some(dir) = file_path.parent() else {
        return Vec::new();
    };

    let mut results = Vec::new();
    for line in content.lines() {
        if !is_import_line(line, lang) {
            continue;
        }
        let source = crate::lang::outline::extract_import_source(line, Some(lang));
        if source.is_empty() || is_external(&source, lang) {
            continue;
        }
        if let Some(path) = resolve_import_to_file(dir, &source, lang, boundary) {
            if !results.contains(&path) {
                results.push(path);
            }
        }
    }
    results
}

/// Canonicalize a declared scope so it can be compared against canonicalized candidate
/// paths. A scope that does not exist yields `None`, which falls resolution back to the
/// enclosing repository.
pub(crate) fn canonical_boundary(boundary: Option<&Path>) -> Option<PathBuf> {
    boundary.and_then(|b| b.canonicalize().ok())
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
        // Shares its judgement with `extract_import_source` so the two cannot disagree
        // about which lines are includes — see `c_include_directive_rest`.
        Lang::C | Lang::Cpp => crate::lang::outline::c_include_directive_rest(trimmed).is_some(),
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
pub(crate) fn resolve_import_to_file(
    dir: &Path,
    source: &str,
    lang: Lang,
    boundary: Option<&Path>,
) -> Option<PathBuf> {
    let raw = match lang {
        Lang::Rust => resolve_rust(dir, source),
        Lang::TypeScript | Lang::Tsx | Lang::JavaScript => resolve_js(dir, source),
        Lang::Python => resolve_python(dir, source),
        Lang::C | Lang::Cpp => resolve_c_include(dir, source, boundary),
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

/// How far up to look for an include root.
///
/// Sized from evidence rather than caution: over all 524 quoted includes in the leveldb
/// fixture, 310 resolve at one hop and 7 at two, none beyond. `Source/<Module>/A/B/C.h`
/// including `A/Other.h` — the deepest layout worth supporting — is three. Every hop
/// past that buys nothing measurable and only widens the surface for a wrong match.
///
/// Counts *ancestors*. The walk also tests the including directory itself, at hop 0, which
/// is what lets a file reach an `include/` sitting beside it rather than above it.
const MAX_INCLUDE_ROOT_HOPS: usize = 4;

/// Conventional include-root directory names, tried as *siblings* at each level of the walk.
///
/// The upward walk alone only finds roots that are ancestors of the including file.
/// The canonical C++ layout puts the root beside the sources instead — `include/leveldb/db.h`
/// included from `db/db_impl.cc` — which is `-Iinclude` to a compiler and invisible to a
/// pure upward walk. These two names cover that layout; anything more exotic needs real
/// build metadata.
const CONVENTIONAL_INCLUDE_ROOTS: &[&str] = &["include", "inc"];

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
/// The walk is confined to a containment root, and every candidate must normalise to a
/// path inside it. Without that, a `..` in the include could let resolution reach files
/// outside the project entirely.
///
/// Two things can supply that root, and it matters that they *compose* rather than one
/// replacing the other:
///
///   * `boundary` — the caller's declared search scope. Requiring `.git` was once the only
///     rule, and it meant include-root resolution did nothing whatsoever in a tree that is
///     not a git checkout: every project-relative include silently bucketed as external.
///     A caller that named a scope has already stated the boundary.
///   * the enclosing `.git` repository, derived from the *including file*.
///
/// The scope wins only when it actually contains the file. A declared scope is not
/// guaranteed to be related to the file being analysed — `tilth_deps` with an absolute
/// `path` and no `scope` resolves the scope to the server's process cwd, which may be a
/// different checkout entirely — and letting an unrelated root win rejected every
/// candidate and silently reclassified real local includes as external. That was a
/// regression against the `.git`-only rule, which at least always derived its root from
/// the file. So: the scope when it contains the file, the repository otherwise.
fn resolve_c_include(dir: &Path, source: &str, boundary: Option<&Path>) -> Option<PathBuf> {
    let clean = source.trim_matches('"');

    // The standard `""` search: relative to the including file. This is also the only
    // correct place for a `..`-relative include, which is by definition anchored to the
    // including file rather than to any include root.
    let direct = dir.join(clean);
    if direct.is_file() {
        return Some(direct);
    }

    if !is_include_root_relative(clean) {
        return None;
    }

    let root = match boundary.filter(|b| is_within(dir, b)) {
        Some(b) => b.to_path_buf(),
        None => enclosing_repo_root(dir)?,
    };

    // Hop 0 is the including directory itself, so that a file gets its *own* `include/`
    // sibling tried. Testing only ancestors meant a translation unit sitting directly at
    // the containment root could never reach `<root>/include/…` — there is no ancestor
    // left to hang the sibling off. That is an ordinary case since the declared scope
    // became a containment root, not an edge one.
    //
    // The cost is that hop 0 re-tests `dir.join(clean)`, which the direct check above
    // already covered: one extra stat on a path known not to be a file.
    let mut base = dir;
    for _ in 0..=MAX_INCLUDE_ROOT_HOPS {
        for candidate in candidates_at(base, clean) {
            let normalized = normalize_path(&candidate);
            if normalized.is_file() && is_within(&normalized, &root) {
                return Some(normalized);
            }
        }
        if base == root {
            break;
        }
        let Some(parent) = base.parent() else { break };
        base = parent;
    }
    None
}

/// Is `candidate` inside `root`?
///
/// Compared in canonical form. A lexical `starts_with` is only correct when both paths are
/// spelled the same way, and they need not be: `boundary` is canonicalized so it can be
/// compared at all, while `candidate` inherits the caller's spelling of the including
/// file's path. On Windows that is the difference between `C:\x` and `\\?\C:\x`, which
/// compares as "outside" and silently refuses every resolution. Falls back to the lexical
/// test if either path cannot be canonicalized.
fn is_within(candidate: &Path, root: &Path) -> bool {
    match (candidate.canonicalize(), root.canonicalize()) {
        (Ok(c), Ok(r)) => c.starts_with(&r),
        _ => candidate.starts_with(root),
    }
}

/// The include-root candidates to try at `base`: `base` itself, then the conventional
/// sibling roots beneath it (see `CONVENTIONAL_INCLUDE_ROOTS`).
fn candidates_at(base: &Path, clean: &str) -> Vec<PathBuf> {
    let mut out = Vec::with_capacity(1 + CONVENTIONAL_INCLUDE_ROOTS.len());
    out.push(base.join(clean));
    for root in CONVENTIONAL_INCLUDE_ROOTS {
        out.push(base.join(root).join(clean));
    }
    out
}

/// True when an include path is worth resolving against an include root.
///
/// Requires at least two real path components and no `..`:
///   * A single real component (`"config.h"`, and equally `"./config.h"`) is either a
///     sibling — already found by the direct check — or outside the project. Walking up
///     for it is how an unrelated same-named file several directories away gets matched.
///   * A `..` is anchored to the including file, so joining it onto an ancestor is
///     meaningless, and it can climb back out of the repository.
fn is_include_root_relative(clean: &str) -> bool {
    let mut real = 0;
    for part in clean.split(['/', '\\']) {
        match part {
            "" | "." => {}
            ".." => return false,
            _ => real += 1,
        }
    }
    real >= 2
}

/// Nearest ancestor of `dir` (inclusive) containing `.git`, if any.
///
/// `.git` is a directory in a normal clone and a *file* in a linked worktree or
/// submodule, so presence rather than kind is what matters.
fn enclosing_repo_root(dir: &Path) -> Option<PathBuf> {
    let mut cur = Some(dir);
    while let Some(d) = cur {
        if d.join(".git").exists() {
            return Some(d.to_path_buf());
        }
        cur = d.parent();
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

        let from_sibling =
            resolve_import_to_file(&root.join("a/b"), "./target", Lang::TypeScript, None)
                .expect("sibling");
        let from_cousin =
            resolve_import_to_file(&root.join("a/c"), "../b/target", Lang::TypeScript, None)
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
        let got = resolve_c_include(&from, "\"lib/status.h\"", None).expect("should resolve");
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

        let got = resolve_c_include(&root.join("a/b"), "\"a/x.h\"", None).expect("resolves");
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
            resolve_c_include(&root.join("deep/nested"), "\"config.h\"", None).is_none(),
            "a bare filename must not resolve to a same-named file up the tree"
        );
    }

    /// The canonical C++ layout puts the include root *beside* the sources —
    /// `include/lib/db.h` included from `src/db.cc` — which is `-Iinclude` to a compiler
    /// and invisible to a pure upward walk. Without this, 40% of leveldb's quoted
    /// includes still resolved to nothing and were reported as external.
    #[test]
    fn c_include_resolves_against_a_sibling_include_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::create_dir_all(root.join("include/lib")).unwrap();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("include/lib/db.h"), "struct DB {};\n").unwrap();

        let got = resolve_c_include(&root.join("src"), "\"lib/db.h\"", None).expect("resolves");
        assert_eq!(got, root.join("include/lib/db.h"));
    }

    /// A `..` include is anchored to the including file — `"../shared.h"` from `a/b`
    /// means `a/shared.h` and nothing else. Joining it onto an ancestor changes what it
    /// means, so the walk must refuse it: here `a/shared.h` does not exist, and walking
    /// would silently "resolve" the include to the unrelated `shared.h` at the repo root.
    /// It also let resolution climb back out of the repository despite the `.git` bound.
    ///
    /// The direct check still honours `..` — that is the include's real meaning, and a
    /// compiler resolves it the same way.
    #[test]
    fn c_include_refuses_parent_relative_paths_for_the_walk() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::create_dir_all(root.join("a/b")).unwrap();
        // Only the repo-root copy exists; `a/shared.h` — what the include actually names
        // — does not.
        fs::write(root.join("shared.h"), "// wrong target\n").unwrap();

        assert!(
            resolve_c_include(&root.join("a/b"), "\"../shared.h\"", None).is_none(),
            "a `..` include must not be re-anchored onto an ancestor"
        );

        // With the file where the include actually points, the direct check finds it.
        fs::write(root.join("a/shared.h"), "// right target\n").unwrap();
        let got = resolve_c_include(&root.join("a/b"), "\"../shared.h\"", None).expect("resolves");
        assert_eq!(normalize_path(&got), root.join("a/shared.h"));
    }

    /// `"./cfg.h"` is semantically identical to `"cfg.h"`, which the guard exists to
    /// refuse — counting separators rather than real components let it through.
    #[test]
    fn c_include_treats_dot_slash_as_a_bare_filename() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::create_dir_all(root.join("a/b")).unwrap();
        fs::write(root.join("cfg.h"), "// unrelated\n").unwrap();

        assert!(
            resolve_c_include(&root.join("a/b"), "\"./cfg.h\"", None).is_none(),
            "./name is a bare filename and must not trigger the walk"
        );
    }

    /// With no `.git` anywhere — a tarball checkout, a vendored drop — there is nothing to
    /// bound the walk, so it must not run at all. Previously it ran the full hop budget
    /// and could resolve a file outside the intended tree, then leak that absolute path
    /// into `tilth_deps` output.
    #[test]
    fn c_include_does_not_walk_without_a_repository_root() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        fs::create_dir_all(root.join("proj/a/b")).unwrap();
        fs::create_dir_all(root.join("x")).unwrap();
        fs::write(root.join("x/leak.h"), "// outside proj\n").unwrap();

        assert!(
            resolve_c_include(&root.join("proj/a/b"), "\"x/leak.h\"", None).is_none(),
            "without a .git bound the walk must not run"
        );
    }

    /// A declared scope is the containment boundary, and it does not need to be a git
    /// checkout. Requiring `.git` meant include-root resolution did nothing at all in a
    /// tree that is not a repository: every project-relative include silently bucketed as
    /// external, even when the target sat directly under the scope the caller named.
    ///
    /// The three forms in one tree, which is how the bug was reported:
    ///   `B/BThing.h` includes `"BThing2.h"`   — same directory
    ///   `B/BThing.h` includes `"C/CThing.h"`  — subpath below its own directory
    ///   `B/BThing.h` includes `"A/AThing.h"`  — relative to the include root
    /// Only the third needs the walk, and only the third was broken.
    #[test]
    fn c_include_resolves_against_a_declared_scope_without_a_git_root() {
        let tmp = tempfile::tempdir().unwrap();
        let module = tmp.path().join("ModuleRoot");
        for d in ["A", "B", "B/C"] {
            fs::create_dir_all(module.join(d)).unwrap();
        }
        fs::write(module.join("A/AThing.h"), "// include-root relative\n").unwrap();
        fs::write(module.join("B/BThing2.h"), "// sibling\n").unwrap();
        fs::write(module.join("B/C/CThing.h"), "// below own dir\n").unwrap();
        // Deliberately no `.git` anywhere.
        assert!(
            enclosing_repo_root(&module.join("B")).is_none(),
            "fixture must not sit inside a repository, or it proves nothing"
        );

        let from = module.join("B");
        // Canonicalized, as a real caller's scope is — and deliberately a *different
        // spelling* from `from`, which is not. Containment has to survive that.
        let scope = module.canonicalize().unwrap();
        for (include, expected) in [
            ("\"BThing2.h\"", "B/BThing2.h"),
            ("\"C/CThing.h\"", "B/C/CThing.h"),
            ("\"A/AThing.h\"", "A/AThing.h"),
        ] {
            let got = resolve_c_include(&from, include, Some(&scope))
                .unwrap_or_else(|| panic!("{include} must resolve under a declared scope"));
            assert_eq!(got, module.join(expected), "for {include}");
        }
    }

    /// The declared scope bounds the walk as strictly as `.git` did.
    ///
    /// The positive control is not optional. Without it this test passes for the wrong
    /// reason: reintroduce the `.git`-only rule and, since the fixture has no `.git`,
    /// resolution is disabled outright — so "nothing resolved" is satisfied by nothing
    /// being *attempted*. The in-scope header proves the walk was live and still refused
    /// the out-of-scope one.
    #[test]
    fn c_include_walk_stops_at_the_declared_scope() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        fs::create_dir_all(root.join("proj/src")).unwrap();
        fs::create_dir_all(root.join("proj/inside")).unwrap();
        fs::create_dir_all(root.join("outside")).unwrap();
        fs::write(root.join("proj/inside/ok.h"), "// in scope\n").unwrap();
        fs::write(root.join("outside/leak.h"), "// outside\n").unwrap();

        let from = root.join("proj/src");
        let scope = root.join("proj").canonicalize().unwrap();

        assert_eq!(
            resolve_c_include(&from, "\"inside/ok.h\"", Some(&scope)),
            Some(root.join("proj/inside/ok.h")),
            "positive control: the walk must be live inside the scope"
        );
        assert!(
            resolve_c_include(&from, "\"outside/leak.h\"", Some(&scope)).is_none(),
            "the walk must not resolve outside the declared scope"
        );
    }

    /// A declared scope that has nothing to do with the file must not disable resolution.
    ///
    /// `tilth_deps` given an absolute `path` and no `scope` resolves the scope to the
    /// server's process cwd, which can be an unrelated checkout. Letting that win as the
    /// containment root rejected every candidate and silently reclassified real local
    /// includes as external — a regression against the `.git`-only rule it replaced, which
    /// at least always derived its root from the file. The scope wins only when it contains
    /// the file; otherwise the enclosing repository does.
    #[test]
    fn c_include_falls_back_to_the_repo_when_the_scope_excludes_the_file() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        fs::create_dir_all(root.join("proj/.git")).unwrap();
        fs::create_dir_all(root.join("proj/Alpha")).unwrap();
        fs::create_dir_all(root.join("proj/Gamma")).unwrap();
        fs::create_dir_all(root.join("elsewhere")).unwrap();
        fs::write(root.join("proj/Alpha/Beta.h"), "// target\n").unwrap();

        let from = root.join("proj/Gamma");
        let unrelated = root.join("elsewhere").canonicalize().unwrap();

        assert_eq!(
            resolve_c_include(&from, "\"Alpha/Beta.h\"", Some(&unrelated)),
            Some(root.join("proj/Alpha/Beta.h")),
            "an unrelated scope must fall back to the file's repository, not veto resolution"
        );
    }

    /// An include root at the top of the repository must still resolve — the boundary is
    /// "do not go above the root", not "do not test the root".
    #[test]
    fn c_include_resolves_when_the_repo_root_is_the_include_root() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::create_dir_all(root.join("src/a")).unwrap();
        fs::write(root.join("src/shared.h"), "// at src/\n").unwrap();

        let got =
            resolve_c_include(&root.join("src/a"), "\"src/shared.h\"", None).expect("resolves");
        assert_eq!(got, root.join("src/shared.h"));
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
            resolve_c_include(&root.join("proj/src"), "\"outside/leak.h\"", None).is_none(),
            "the walk must not resolve outside the repository"
        );
    }

    /// A linked worktree and a submodule both have `.git` as a *file*, not a directory, so
    /// the root check must test presence rather than kind.
    #[test]
    fn c_include_treats_a_git_file_as_a_repository_root() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        fs::create_dir_all(root.join("wt/include/lib")).unwrap();
        fs::create_dir_all(root.join("wt/src")).unwrap();
        // `.git` as a file, the linked-worktree / submodule form.
        fs::write(root.join("wt/.git"), "gitdir: ../real/.git/worktrees/wt\n").unwrap();
        fs::write(root.join("wt/include/lib/db.h"), "struct DB {};\n").unwrap();

        let got = resolve_c_include(&root.join("wt/src"), "\"lib/db.h\"", None)
            .expect("a .git file must still establish the root");
        assert_eq!(got, root.join("wt/include/lib/db.h"));
    }

    /// A file sitting directly at the containment root must still get its *own* `include/`
    /// sibling tried. Candidates were only ever tested at ancestors, so `<root>/main.cpp`
    /// including `"lib/db.h"` never reached `<root>/include/lib/db.h` — there is no ancestor
    /// left inside the root to hang the sibling off, and the include bucketed as external.
    ///
    /// The tell was that moving the same file down one level made it resolve, which
    /// `c_include_resolves_against_a_sibling_include_directory` covers. Both spellings of
    /// the same layout are asserted here so the asymmetry cannot come back.
    ///
    /// Reachable as an ordinary case since the declared scope became a containment root: a
    /// translation unit at the scope root is now normal, not exotic.
    #[test]
    fn c_include_resolves_its_own_include_sibling_at_the_scope_root() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        fs::create_dir_all(root.join("include/lib")).unwrap();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("include/lib/db.h"), "struct DB {};\n").unwrap();
        // Deliberately no `.git`: the declared scope is what bounds the walk.
        assert!(
            enclosing_repo_root(root).is_none(),
            "fixture must not sit inside a repository, or it proves nothing"
        );
        let scope = root.canonicalize().unwrap();

        assert_eq!(
            resolve_c_include(root, "\"lib/db.h\"", Some(&scope)),
            Some(root.join("include/lib/db.h")),
            "a file at the scope root must try the include/ beside it"
        );
        assert_eq!(
            resolve_c_include(&root.join("src"), "\"lib/db.h\"", Some(&scope)),
            Some(root.join("include/lib/db.h")),
            "the ancestor spelling of the same layout must still resolve"
        );
    }

    /// `# include "X.h"` — legal C, and not rare in older codebases — was not treated as an
    /// import by any consumer, because detection required `#include` as a single token. It
    /// contributed to neither `uses_local` nor `uses_external`: a silently dropped
    /// dependency, the same failure mode as the trailing-comment bug.
    #[test]
    fn spaced_include_is_detected_and_lands_in_the_right_bucket() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        fs::write(root.join("Widget.h"), "struct W {};\n").unwrap();
        let main = root.join("main.cpp");
        let content = "#  include \"Widget.h\"\n\
                       # include <vector>\n\
                       int main() { return 0; }\n";
        fs::write(&main, content).unwrap();

        assert_eq!(
            resolve_local_imports(&main, content, None),
            vec![root.join("Widget.h")],
            "the spaced quoted include must resolve as a local dependency"
        );
        // The system header is detected too, and its delimiter survives extraction — which
        // is the only thing that keeps it out of the local bucket. Asserted on the fixture's
        // own line rather than on a literal, so reverting the fix fails this half as well.
        let vector_line = content.lines().nth(1).expect("fixture has a second line");
        assert!(is_import_line(vector_line, Lang::Cpp), "{vector_line}");
        let source = crate::lang::outline::extract_import_source(vector_line, Some(Lang::Cpp));
        assert_eq!(source, "<vector>");
        assert!(is_external(&source, Lang::Cpp));
    }

    /// `is_import_line` and `extract_import_source` are separate judgements, and a mismatch
    /// between them is precisely how an include gets dropped with no warning: one says
    /// "import", the other yields a name that resolves to nothing and carries no delimiter,
    /// so the line lands in neither the local nor the external bucket. #10's trailing-comment
    /// bug worked exactly that way. Pin that they agree on every form either accepts.
    #[test]
    fn c_include_detection_and_extraction_agree() {
        use crate::lang::outline::extract_import_source;

        // `None` means "not an include line at all".
        let cases: &[(&str, Option<&str>)] = &[
            ("#include \"X.h\"", Some("\"X.h\"")),
            ("# include \"X.h\"", Some("\"X.h\"")),
            ("#  include \"X.h\"", Some("\"X.h\"")),
            ("#\tinclude \"X.h\"", Some("\"X.h\"")),
            ("   # include \"X.h\"", Some("\"X.h\"")),
            ("#include\"X.h\"", Some("\"X.h\"")),
            ("#include \"X.h\" // note", Some("\"X.h\"")),
            ("#include /* why */ \"X.h\"", Some("\"X.h\"")),
            ("  #include <vector>", Some("<vector>")),
            ("# include <vector> // std", Some("<vector>")),
            ("#include_next <stdio.h>", Some("<stdio.h>")),
            ("# include_next \"limits.h\"", Some("\"limits.h\"")),
            ("#pragma once", None),
            ("# define INCLUDE_GUARD 1", None),
            ("#ifndef X_H", None),
            ("// #include \"X.h\"", None),
            ("int x = 0;", None),
        ];

        for (line, want) in cases {
            assert_eq!(
                is_import_line(line, Lang::Cpp),
                want.is_some(),
                "detection is wrong for: {line}"
            );
            let Some(header) = want else {
                // Extraction is only ever reached *through* detection, so its output for a
                // non-include line is unconstrained — the generic fallback would hand back
                // `"X.h"` for the commented-out row above. Detection saying no is the whole
                // guard, and that is what is asserted.
                continue;
            };
            let source = extract_import_source(line, Some(Lang::Cpp));
            assert_eq!(&source, header, "extraction is wrong for: {line}");
            // The delimiter has to survive extraction, because it is the only thing
            // `is_external` uses to tell a system header from a project-relative one.
            assert_eq!(
                is_external(&source, Lang::Cpp),
                header.starts_with('<'),
                "bucketed wrongly: {line}"
            );
        }
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
