pub mod format;
pub mod matching;
pub mod overlay;
pub mod parse;

use std::collections::HashSet;
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use rayon::prelude::*;

use crate::types::OutlineKind;

#[derive(Debug)]
pub enum DiffSource {
    GitUncommitted,
    GitStaged,
    GitRef(String),
    Files(PathBuf, PathBuf),
    Patch(PathBuf),
    Log(String),
}

#[derive(Debug)]
pub struct FileDiff {
    pub path: PathBuf,
    pub old_path: Option<PathBuf>,
    pub status: FileStatus,
    pub hunks: Vec<Hunk>,
    pub is_generated: bool,
    pub is_binary: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileStatus {
    Added,
    Modified,
    Deleted,
    Renamed,
}

#[derive(Debug)]
pub struct Hunk {
    pub old_start: u32,
    pub old_count: u32,
    pub new_start: u32,
    pub new_count: u32,
    pub lines: Vec<DiffLine>,
}

#[derive(Debug)]
pub struct DiffLine {
    pub kind: DiffLineKind,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffLineKind {
    Context,
    Added,
    Removed,
}

#[derive(Debug)]
pub struct DiffSymbol {
    pub entry: crate::types::OutlineEntry,
    pub identity: SymbolIdentity,
    pub content_hash: u64,
    pub structural_hash: u64,
    pub source_text: String,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct SymbolIdentity {
    pub kind: OutlineKind,
    pub parent_path: String,
    pub name: String,
}

#[derive(Debug)]
pub struct SymbolChange {
    pub name: String,
    pub kind: OutlineKind,
    pub change: ChangeType,
    pub match_confidence: MatchConfidence,
    pub line: u32,
    pub old_sig: Option<String>,
    pub new_sig: Option<String>,
    pub size_delta: Option<(u32, u32)>,
}

#[derive(Debug, Clone)]
pub enum ChangeType {
    Added,
    Deleted,
    BodyChanged,
    SignatureChanged,
    Renamed { old_name: String },
    Moved { old_path: PathBuf },
    RenamedAndMoved { old_name: String, old_path: PathBuf },
    Unchanged,
}

#[derive(Debug, Clone)]
pub enum MatchConfidence {
    Exact,
    Structural,
    Fuzzy(f32),
    Ambiguous(u32),
}

#[derive(Debug)]
pub struct FileOverlay {
    pub path: PathBuf,
    pub symbol_changes: Vec<SymbolChange>,
    pub attributed_hunks: Vec<(String, Vec<DiffLine>)>,
    pub conflicts: Vec<Conflict>,
    pub new_content: Option<String>,
    /// Why symbol analysis was abandoned for this file, if it was.
    ///
    /// An overlay with no symbols is otherwise indistinguishable from a file
    /// git reported as changed but whose structure did not move — so the
    /// formatters must say "could not analyze" rather than render a confident
    /// `+0/−0`. See issue #111.
    pub analysis_failed: Option<String>,
}

#[derive(Debug)]
pub struct Conflict {
    pub line: u32,
    pub ours: String,
    pub theirs: String,
    pub enclosing_fn: Option<String>,
}

#[derive(Debug)]
pub struct CommitSummary {
    pub hash: String,
    pub timestamp: i64,
    pub message: String,
    pub author: String,
    pub overlays: Vec<FileOverlay>,
}

/// Resolve the diff source from CLI/MCP parameters.
///
/// Priority: patch > log > a+b > source > default (uncommitted).
/// Returns an error if only one of `a` or `b` is provided.
pub fn resolve_source(
    source: Option<&str>,
    a: Option<&str>,
    b: Option<&str>,
    patch: Option<&str>,
    log: Option<&str>,
) -> Result<DiffSource, String> {
    if let Some(p) = patch {
        return Ok(DiffSource::Patch(PathBuf::from(p)));
    }
    if let Some(l) = log {
        return Ok(DiffSource::Log(l.to_string()));
    }
    match (a, b) {
        (Some(fa), Some(fb)) => return Ok(DiffSource::Files(PathBuf::from(fa), PathBuf::from(fb))),
        (Some(_), None) | (None, Some(_)) => {
            return Err("both --a and --b must be provided together".to_string());
        }
        (None, None) => {}
    }
    if let Some(s) = source {
        let ds = match s {
            "staged" => DiffSource::GitStaged,
            "uncommitted" | "working" => DiffSource::GitUncommitted,
            r => DiffSource::GitRef(r.to_string()),
        };
        return Ok(ds);
    }
    Ok(DiffSource::GitUncommitted)
}

/// Run one `git diff` invocation.
///
/// `uncommitted_base` is the rev the working tree is compared against for
/// `GitUncommitted`; every other source ignores it.
fn git_diff_once(source: &DiffSource, uncommitted_base: &str) -> Result<Output, String> {
    let mut cmd = Command::new("git");
    cmd.args(["-c", "core.quotePath=false"]);
    cmd.arg("diff");

    match source {
        DiffSource::GitUncommitted => {
            // working tree vs the base (unstaged + staged)
            cmd.arg(uncommitted_base);
        }
        DiffSource::GitStaged => {
            cmd.arg("--staged");
        }
        DiffSource::GitRef(r) => {
            cmd.arg(r);
        }
        DiffSource::Files(fa, fb) => {
            cmd.arg("--no-index").arg("--").arg(fa).arg(fb);
        }
        // Patch and Log are handled by the caller
        DiffSource::Patch(_) | DiffSource::Log(_) => unreachable!(),
    }

    cmd.output()
        .map_err(|e| format!("failed to run git diff: {e}"))
}

/// Is HEAD *unborn* — a symbolic ref to a branch that has no commit yet?
///
/// Two states make `git rev-parse --verify HEAD` fail, and `git diff HEAD`
/// emits an identical fatal for both: a repository with no commit, and one
/// whose HEAD ref is corrupt. Only the first may fall back to the empty tree.
/// Treating the second as unborn reports a damaged repository's entire working
/// tree as newly added — the silent-success shape #112 exists to prevent.
///
/// `git symbolic-ref` separates them: it resolves for an unborn branch (and for
/// an orphan branch, which is also legitimately unborn) but fails outright when
/// HEAD itself cannot be read.
///
/// Only consulted after a `git diff` has already failed, so the common path
/// never pays for either subprocess.
fn head_is_unborn() -> bool {
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .output()
            .is_ok_and(|o| o.status.success())
    };
    !git(&["rev-parse", "--verify", "--quiet", "HEAD"]) && git(&["symbolic-ref", "-q", "HEAD"])
}

/// Execute a git diff command and return raw unified diff output.
fn run_git_diff(source: &DiffSource) -> Result<String, String> {
    match source {
        DiffSource::Log(_) => {
            return Err("log mode should not call run_git_diff directly".to_string());
        }
        DiffSource::Patch(path) => {
            let content = std::fs::read_to_string(path)
                .map_err(|e| format!("failed to read patch file: {e}"))?;
            return Ok(content);
        }
        _ => {}
    }

    let mut output = git_diff_once(source, "HEAD")?;
    let mut stdout = String::from_utf8_lossy(&output.stdout).into_owned();

    // Before the first commit HEAD does not resolve, so `git diff HEAD` exits
    // 128 with nothing on stdout. The empty tree is what the repository looked
    // like then — the same fallback `diff_log` uses for a parentless commit —
    // and it makes every tracked file read as added, which is what they are.
    //
    // Retried rather than pre-checked so that the overwhelmingly common case,
    // a repo that has commits, never spends a subprocess proving it.
    if !output.status.success()
        && stdout.is_empty()
        && matches!(source, DiffSource::GitUncommitted)
        && head_is_unborn()
    {
        output = git_diff_once(source, &empty_tree())?;
        stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    }

    // git diff --no-index exits 1 when there are differences; that is normal,
    // and so is any non-zero exit that still produced a diff — the caller can
    // decide what to make of it.
    //
    // Failing with *nothing* on stdout is different: `diff()` reads empty
    // output as "No changes." and exits 0, so `tilth diff base..typo` reported
    // a clean tree for a ref that does not exist. Same danger as #111 — an
    // error dressed up as authoritative evidence — one layer up.
    if !output.status.success() && stdout.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let detail = stderr.trim();
        return Err(if detail.is_empty() {
            format!("git diff failed ({})", output.status)
        } else {
            format!("git diff failed: {detail}")
        });
    }

    Ok(stdout)
}

/// Full diff orchestrator — parse → overlay → format pipeline.
pub fn diff(
    source: &DiffSource,
    scope: Option<&str>,
    search: Option<&str>,
    blast: bool,
    _expand: usize,
    budget: Option<u64>,
) -> Result<String, String> {
    // Log mode has its own pipeline.
    if let DiffSource::Log(range) = source {
        return diff_log(range, scope, budget);
    }

    let raw = run_git_diff(source)?;
    if raw.is_empty() {
        return Ok("No changes.".to_string());
    }

    // 1. Parse raw unified diff.
    let file_diffs = parse::parse_unified_diff(&raw);
    if file_diffs.is_empty() {
        return Ok("No changes.".to_string());
    }

    // 2. Pin a symmetric range to its merge base once, up front.
    //
    // `get_old_content` runs inside the parallel map below, once per file, so
    // resolving `a...b` down there would spawn a `git merge-base` per file —
    // a subprocess apiece on a wide diff. Rewriting the range to `<sha>..b`
    // here costs one call and leaves every reader downstream on the plain-range
    // path. It also pins the answer: a concurrent ref move mid-diff can no
    // longer give two files different old sides.
    let overlay_source = overlay::pin_range_to_merge_base(source)?;
    let overlay_source = overlay_source.as_ref().unwrap_or(source);

    // 3. Build structural overlays in parallel — each FileDiff is independent
    // and `compute_overlay` constructs its own tree-sitter parser per call
    // (see `lang::outline::get_outline_entries`), so no shared mutable state
    // crosses worker boundaries.
    let mut overlays: Vec<FileOverlay> = file_diffs
        .par_iter()
        .map(|fd| overlay::compute_overlay(fd, overlay_source))
        .collect();

    // 3. Cross-file move detection.
    overlay::cross_file_matching(&mut overlays);

    // 4. Signature warnings, plus any file whose analysis was abandoned.
    // The latter are collected before the search filter below, which drops
    // symbol-less overlays and would otherwise make the failure disappear.
    let mut warnings = overlay::signature_warnings(&overlays);
    for overlay in &overlays {
        if let Some(reason) = &overlay.analysis_failed {
            warnings.push(format!(
                "could not analyze `{}` — {reason}; symbol counts for it are missing, not zero",
                overlay.path.display()
            ));
        }
    }

    // 5. Search filter.
    if let Some(term) = search {
        filter_by_search(&mut overlays, term);
        if overlays.is_empty() {
            // Collecting the warnings above is not enough on its own: an
            // abandoned overlay has no symbols and no hunks, so the filter
            // always drops it, and this early return is the one path that
            // never renders `warnings`. Without them "no match" would be the
            // #111 lie again — a file we could not read, reported as absent.
            let mut out = format!("No changes matching '{term}'.");
            for w in &warnings {
                let _ = write!(out, "\n⚠ {w}");
            }
            return Ok(out);
        }
    }

    // 6. Blast radius.
    if blast {
        let mut blast_warnings = compute_blast(&overlays);
        warnings.append(&mut blast_warnings);
    }

    // 7. Build file_meta parallel to overlays.
    let file_meta: Vec<(&Path, bool, bool)> = overlays
        .iter()
        .map(|o| {
            // Find the original FileDiff for this overlay to get is_generated/is_binary.
            let fd = file_diffs.iter().find(|fd| fd.path == o.path);
            let (is_generated, is_binary) =
                fd.map_or((false, false), |f| (f.is_generated, f.is_binary));
            (o.path.as_path(), is_generated, is_binary)
        })
        .collect();

    // 8. Format based on scope.
    let label = source_label(source);
    let paths: Vec<String> = overlays
        .iter()
        .map(|o| normalize_path(&o.path.to_string_lossy()))
        .collect();

    // Split the symbol off first: `<path>:<symbol>` scopes a path too, and the
    // conflict pass below must see the same selection the body was rendered
    // from, or it reports conflicts in files the header says are out of scope.
    let (scope_path, symbol) = match scope {
        None => (None, None),
        Some(s) => match split_path_symbol(s) {
            Some((path, sym)) => (Some(path), Some(sym)),
            None => (Some(s), None),
        },
    };

    let selection: Option<ScopeMatch> = match scope_path {
        None => None,
        Some(s) => match scope_request(s) {
            // An absolute scope that resolves to the repo root, or `.`, asks
            // the same question as no scope at all.
            ScopeRequest::Everything => None,
            ScopeRequest::Paths(candidates) => match select_scope(&paths, &candidates) {
                Some((m, _)) => Some(m),
                None => return Err(scope_miss_error(&paths, s)),
            },
        },
    };

    // Label with the form that matched, not the caller's absolute path.
    let scoped_label = |m: &ScopeMatch| -> String {
        match scope_request(scope_path.unwrap_or(".")) {
            ScopeRequest::Paths(c) => {
                let _ = m;
                format!("{label} ({})", c.last().cloned().unwrap_or_default())
            }
            ScopeRequest::Everything => label.clone(),
        }
    };

    let mut output = match (&selection, symbol) {
        // Unscoped, or scoped to the whole repo.
        (None, _) => {
            let all: Vec<&FileOverlay> = overlays.iter().collect();
            format::format_overview(&all, &file_meta, &warnings, &label, budget)
        }

        // `file:function` — one symbol inside exactly one file.
        (Some(m), Some(fn_name)) => match m.files.as_slice() {
            [i] if m.under.is_empty() => format::format_function_detail(&overlays[*i], fn_name),
            // A directory has no single overlay to render a symbol from.
            // Saying so beats "not found", which is false — it matched.
            [] => {
                return Err(format!(
                    "scope '{}' is a directory ({} changed files); the file:function form needs one file",
                    scope_path.unwrap_or_default(),
                    m.under.len()
                ));
            }
            _ => {
                return Err(scope_ambiguous_error(
                    &paths,
                    scope_path.unwrap_or_default(),
                    m,
                ))
            }
        },

        (Some(m), None) => match m.files.as_slice() {
            // Exactly one file and nothing under it — per-symbol detail.
            [i] if m.under.is_empty() => format::format_file_detail(&overlays[*i], budget),
            // A directory — overview of everything beneath it.
            [] => {
                let scoped: Vec<&FileOverlay> = m.under.iter().map(|&i| &overlays[i]).collect();
                let scoped_meta: Vec<(&Path, bool, bool)> =
                    m.under.iter().map(|&i| file_meta[i]).collect();
                // Warnings are global, not per-file, so they ride along
                // unfiltered — a file we could not read stays visible even when
                // it sits outside the requested directory.
                format::format_overview(&scoped, &scoped_meta, &warnings, &scoped_label(m), budget)
            }
            // More than one file, or a file shadowing a directory. Picking one
            // silently is how a whole changed directory used to disappear.
            _ => {
                return Err(scope_ambiguous_error(
                    &paths,
                    scope_path.unwrap_or_default(),
                    m,
                ))
            }
        },
    };

    // 9. Conflict detection for uncommitted diffs.
    if matches!(source, DiffSource::GitUncommitted) {
        let in_scope = |i: usize| selection.as_ref().is_none_or(|m| m.all().contains(&i));
        let mut all_conflicts = Vec::new();
        for (i, overlay) in overlays.iter().enumerate() {
            if !in_scope(i) {
                continue;
            }
            let conflicts = overlay::detect_conflicts(&overlay.path);
            if !conflicts.is_empty() {
                all_conflicts.push((&overlay.path, conflicts));
            }
        }
        if !all_conflicts.is_empty() {
            for (path, conflicts) in &all_conflicts {
                output.push('\n');
                output.push_str(&format::format_conflicts(conflicts, path));
            }
            if let Some(b) = budget {
                output = crate::budget::apply(&output, b);
            }
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Human-readable label for a diff source.
fn source_label(source: &DiffSource) -> String {
    match source {
        DiffSource::GitUncommitted => "uncommitted".to_string(),
        DiffSource::GitStaged => "staged".to_string(),
        DiffSource::GitRef(r) => r.clone(),
        DiffSource::Files(a, b) => format!("{} vs {}", a.display(), b.display()),
        DiffSource::Patch(p) => format!("patch: {}", p.display()),
        DiffSource::Log(r) => format!("log: {r}"),
    }
}

// ---------------------------------------------------------------------------
// Scope resolution
//
// A `scope` argument names either one changed file or a directory of them. It
// is matched against the paths *git* reported, which are always repo-relative
// with forward slashes — so everything here normalizes to that shape first.
// ---------------------------------------------------------------------------

/// Which changed paths a scope selected. Indices into the overlay list.
#[derive(Debug, Default, PartialEq, Eq)]
struct ScopeMatch {
    /// Paths that *are* the scope — it named a file.
    files: Vec<usize>,
    /// Paths *beneath* the scope — it named a directory.
    under: Vec<usize>,
}

impl ScopeMatch {
    fn is_empty(&self) -> bool {
        self.files.is_empty() && self.under.is_empty()
    }

    /// Every selected index. Log mode filters with this — it wants all the
    /// matches, not one of them.
    fn all(&self) -> Vec<usize> {
        let mut out = self.files.clone();
        out.extend(&self.under);
        out.sort_unstable();
        out
    }
}

/// How literally a scope has to match a changed path.
///
/// `Anchored` is git's pathspec rule — the scope is relative to the repo root.
/// `Suffix` is tilth's convenience, letting `parse.rs` stand for
/// `src/diff/parse.rs`, and is consulted ONLY when nothing anchored matched.
///
/// The ordering is the whole point. Without it a fully-qualified scope loses to
/// a partial one: an absolute `<root>/src/util.rs` reduces to `src/util.rs` and
/// then suffix-matched `a/src/util.rs` in a repo that had both — a confidently
/// wrong answer about a file the caller never named.
#[derive(Clone, Copy)]
enum Anchoring {
    Anchored,
    Suffix,
}

/// Fold a path into the shape git reports: forward slashes, no trailing slash.
fn normalize_path(p: &str) -> String {
    let slashed = p.trim().replace('\\', "/");
    // `std::fs::canonicalize` emits Windows' extended-length prefix, and an MCP
    // client that canonicalizes before calling sends it verbatim. Left in place
    // it can never match git's root, so every such scope missed.
    let stripped = if let Some(rest) = slashed.strip_prefix("//?/UNC/") {
        format!("//{rest}")
    } else if let Some(rest) = slashed.strip_prefix("//?/") {
        rest.to_string()
    } else {
        slashed
    };
    stripped.trim_end_matches('/').to_string()
}

/// Compare one path component. Windows paths are case-insensitive; git's are
/// not, so this is only used for matching against the *repository root*, never
/// against the paths inside the diff.
fn eq_root_component(a: &str, b: &str) -> bool {
    if cfg!(windows) {
        a.eq_ignore_ascii_case(b)
    } else {
        a == b
    }
}

/// Does this look like an absolute path? Used only to decide whether asking
/// git for the repository root could possibly help — a relative scope can never
/// be under it, so it must not pay for the subprocess.
fn looks_absolute(p: &str) -> bool {
    p.starts_with('/') || p.starts_with('\\') || p.as_bytes().get(1) == Some(&b':')
}

/// Make an absolute scope repo-relative by removing the repository root.
///
/// `prompts/mcp-base.md` tells agents never to pass a bare relative scope, so
/// absolute is the form `tilth_diff` actually receives — and every one of them
/// used to miss, because git's diff paths are repo-relative.
fn strip_repo_root(scope: &str) -> RootRelative {
    if !looks_absolute(scope) {
        return RootRelative::Outside;
    }
    let Ok(output) = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
    else {
        return RootRelative::Outside;
    };
    if !output.status.success() {
        return RootRelative::Outside;
    }
    let root = normalize_path(&String::from_utf8_lossy(&output.stdout));
    if root.is_empty() {
        return RootRelative::Outside;
    }

    // Walk components rather than slicing bytes: a case-insensitive compare on
    // a lowercased copy cannot be mapped back to the original by offset once a
    // non-ASCII character is involved.
    let scope = normalize_path(scope);
    let mut parts = scope.split('/');
    for want in root.split('/') {
        match parts.next() {
            Some(got) if eq_root_component(got, want) => {}
            _ => return RootRelative::Outside,
        }
    }
    let rest: Vec<&str> = parts.collect();
    if rest.is_empty() {
        RootRelative::WholeRepo
    } else {
        RootRelative::Under(rest.join("/"))
    }
}

/// Where an absolute scope sits relative to the repository root.
enum RootRelative {
    /// Not under the root — or not absolute, so it cannot be.
    Outside,
    /// The root itself. An agent resolving "the repo" to its path lands here,
    /// and it means the same thing as passing no scope at all.
    WholeRepo,
    /// A path beneath the root, repo-relative.
    Under(String),
}

/// Collect every changed path this scope selects, at one anchoring level.
fn match_tier(paths: &[String], want: &str, how: Anchoring) -> ScopeMatch {
    let mut m = ScopeMatch::default();
    for (i, path) in paths.iter().enumerate() {
        let (is_file, is_under) = match how {
            Anchoring::Anchored => (path == want, path.starts_with(&format!("{want}/"))),
            Anchoring::Suffix => (
                path.ends_with(&format!("/{want}")),
                path.contains(&format!("/{want}/")),
            ),
        };
        if is_file {
            m.files.push(i);
        } else if is_under {
            m.under.push(i);
        }
    }
    m
}

/// Resolve a `scope` argument against the changed paths.
///
/// Anchored matches are tried across every candidate form before any suffix
/// match is considered, so a path the caller fully qualified always beats a
/// partial one. Returns the form that matched alongside the selection, so
/// output can be labelled `src/diff` rather than echoing back the caller's
/// `C:/dev/tilth/src/diff`.
///
/// Every match is returned, not the first — log mode filters on all of them,
/// and `diff()` needs to see multiplicity to report it rather than silently
/// picking one.
fn select_scope(paths: &[String], candidates: &[String]) -> Option<(ScopeMatch, String)> {
    for how in [Anchoring::Anchored, Anchoring::Suffix] {
        for want in candidates {
            let m = match_tier(paths, want, how);
            if !m.is_empty() {
                return Some((m, want.clone()));
            }
        }
    }
    None
}

/// What a scope is asking for.
enum ScopeRequest {
    /// The whole repository — equivalent to no scope.
    Everything,
    /// Path forms worth matching, most literal first.
    Paths(Vec<String>),
}

/// Work out what a scope asks for, resolving the repository root at most once.
///
/// Log mode calls `select_scope` once per commit — building this in there would
/// have spawned a subprocess per commit, the same per-item-subprocess shape as
/// the merge-base call fixed in #112.
fn scope_request(scope: &str) -> ScopeRequest {
    let direct = normalize_path(scope);
    if direct == "." || direct.is_empty() {
        return ScopeRequest::Everything;
    }
    let mut out = vec![direct];
    match strip_repo_root(scope) {
        RootRelative::WholeRepo => return ScopeRequest::Everything,
        RootRelative::Under(relative) => {
            if !out.contains(&relative) {
                out.push(relative);
            }
        }
        RootRelative::Outside => {}
    }
    ScopeRequest::Paths(out)
}

/// Split a scope into `(path, symbol)` if it uses the `file:function` form.
///
/// Splits on the LAST colon, and only when the tail could be a symbol name.
/// Splitting on the first colon made `C:/dev/x.rs` a request for the file `C`,
/// which is why every absolute Windows scope reported `file 'C' not found`.
fn split_path_symbol(scope: &str) -> Option<(&str, &str)> {
    let (path, symbol) = scope.rsplit_once(':')?;
    if path.is_empty() || symbol.is_empty() {
        return None;
    }
    if symbol.contains('/') || symbol.contains('\\') {
        return None;
    }
    Some((path, symbol))
}

/// Error text for a scope that matched nothing, naming what *did* change.
///
/// `file 'x' not found in diff` left the caller with no way to tell a typo from
/// a genuinely untouched path.
fn scope_miss_error(paths: &[String], scope: &str) -> String {
    const SHOWN: usize = 10;
    if paths.is_empty() {
        return format!("scope '{scope}' matched no changed files — the diff is empty");
    }
    let shown: Vec<&str> = paths.iter().take(SHOWN).map(String::as_str).collect();
    let more = paths.len().saturating_sub(shown.len());
    let suffix = if more > 0 {
        format!(" (+{more} more)")
    } else {
        String::new()
    };
    format!(
        "scope '{scope}' matched no changed file or directory. Changed: {}{suffix}",
        shown.join(", ")
    )
}

/// Error text for a scope that names more than one thing.
///
/// The alternative is picking one and not saying so, which is how
/// `--scope hooks` came back as the single file `scripts/hooks` while quietly
/// dropping a whole changed `tools/hooks/` directory.
fn scope_ambiguous_error(paths: &[String], matched: &str, m: &ScopeMatch) -> String {
    let name = |i: &usize| paths[*i].as_str();
    let files: Vec<&str> = m.files.iter().map(name).collect();
    let under: Vec<&str> = m.under.iter().map(name).collect();

    let mut out = format!("scope '{matched}' is ambiguous — it matches ");
    if !files.is_empty() {
        let _ = write!(out, "the file{} {}", plural(files.len()), files.join(", "));
    }
    if !files.is_empty() && !under.is_empty() {
        out.push_str(" and ");
    }
    if !under.is_empty() {
        let _ = write!(
            out,
            "a directory containing {}",
            join_capped(&under, 5, "file")
        );
    }
    out.push_str(". Qualify it with more of the path.");
    out
}

fn plural(n: usize) -> &'static str {
    if n == 1 {
        ""
    } else {
        "s"
    }
}

/// `a, b, c (+N more)` — never a silent truncation.
fn join_capped(items: &[&str], cap: usize, noun: &str) -> String {
    let shown = items.len().min(cap);
    let more = items.len() - shown;
    let mut out = items[..shown].join(", ");
    if more > 0 {
        let _ = write!(out, " (+{more} more {noun}{})", plural(more));
    }
    out
}

/// Filter overlays to only symbols whose diff lines contain the search term
/// (case-insensitive substring match). Removes files with no matches.
fn filter_by_search(overlays: &mut Vec<FileOverlay>, term: &str) {
    let lower_term = term.to_lowercase();

    overlays.retain_mut(|overlay| {
        // Keep symbol changes that have matching diff lines.
        let matching_symbols: HashSet<String> = overlay
            .attributed_hunks
            .iter()
            .filter(|(_, lines)| {
                lines
                    .iter()
                    .any(|l| l.content.to_lowercase().contains(&lower_term))
            })
            .map(|(name, _)| name.clone())
            .collect();

        // Also match on symbol names themselves.
        let matching_names: HashSet<String> = overlay
            .symbol_changes
            .iter()
            .filter(|c| c.name.to_lowercase().contains(&lower_term))
            .map(|c| c.name.clone())
            .collect();

        let all_matching: HashSet<String> =
            matching_symbols.union(&matching_names).cloned().collect();

        if all_matching.is_empty() {
            return false;
        }

        overlay
            .symbol_changes
            .retain(|c| all_matching.contains(&c.name));
        overlay
            .attributed_hunks
            .retain(|(name, _)| all_matching.contains(name));

        true
    });
}

/// Find callers of signature-changed symbols and return warnings.
fn compute_blast(overlays: &[FileOverlay]) -> Vec<String> {
    let sig_changed: HashSet<String> = overlays
        .iter()
        .flat_map(|o| o.symbol_changes.iter())
        .filter(|c| matches!(c.change, ChangeType::SignatureChanged))
        .map(|c| c.name.clone())
        .collect();

    if sig_changed.is_empty() {
        return Vec::new();
    }

    let scope = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let bloom = crate::index::bloom::BloomFilterCache::new();

    match crate::search::callers::find_callers_batch(&sig_changed, &scope, &bloom, None) {
        Ok(matches) => {
            let mut counts: std::collections::HashMap<String, usize> =
                std::collections::HashMap::new();
            for (target, _) in &matches {
                *counts.entry(target.clone()).or_default() += 1;
            }
            counts
                .into_iter()
                .map(|(name, count)| {
                    format!(
                        "blast: `{name}` signature changed — {count} caller{} may need updating",
                        if count == 1 { "" } else { "s" }
                    )
                })
                .collect()
        }
        Err(_) => Vec::new(),
    }
}

/// Log mode pipeline: run per-commit diffs and format as commit summaries.
fn diff_log(range: &str, scope: Option<&str>, budget: Option<u64>) -> Result<String, String> {
    // Every field is NUL-separated: %P is a variable-length list of parent
    // hashes, so the old positional `%H %at %s` split cannot carry it.
    let output = Command::new("git")
        .args(["log", "--format=%H%x00%at%x00%P%x00%s%x00%an", range])
        .output()
        .map_err(|e| format!("failed to run git log: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("git log failed: {stderr}"));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut summaries: Vec<CommitSummary> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    // Resolved lazily and at most once: it costs a subprocess, and most log
    // ranges contain no parentless commit at all.
    let mut empty_tree_id: Option<String> = None;

    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Format: "<hash>\0<timestamp>\0<parents>\0<subject>\0<author>"
        let mut fields = line.splitn(5, '\0');
        let Some(hash) = fields.next() else {
            continue;
        };
        let timestamp: i64 = fields.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        let parents = fields.next().unwrap_or("").trim().to_string();
        let message = fields.next().unwrap_or("").to_string();
        let author = fields.next().unwrap_or("").to_string();

        // A parentless commit has no `^`, so `<hash>^..<hash>` is not a valid
        // range — diff it against the empty tree, which makes every file read
        // as added, exactly as `git log --patch` shows it. Without this the
        // whole log aborted the moment the range reached such a commit.
        //
        // Two kinds of commit land here, and %P reports both as parentless: the
        // true root, and a shallow clone's boundary commit (git grafts its
        // parents away). Shallow is the CI default, so this is not an edge.
        let commit_source = DiffSource::GitRef(if parents.is_empty() {
            let base = empty_tree_id.get_or_insert_with(empty_tree);
            format!("{base}..{hash}")
        } else {
            format!("{hash}^..{hash}")
        });

        // Defensive, with no trigger I have been able to construct: every
        // parentless case I found is handled above. It stays because the
        // alternative is what this commit is fixing — one unexpected commit
        // taking down the entire log through `?`. Losing one commit's body is
        // survivable; losing the log is not.
        let raw = match run_git_diff(&commit_source) {
            Ok(raw) => raw,
            Err(e) => {
                warnings.push(format!("no diff for {}: {e}", &hash[..hash.len().min(7)]));
                String::new()
            }
        };
        let file_diffs = parse::parse_unified_diff(&raw);

        let mut overlays: Vec<FileOverlay> = file_diffs
            .iter()
            .map(|fd| overlay::compute_overlay(fd, &commit_source))
            .collect();
        overlay::cross_file_matching(&mut overlays);

        for overlay in &overlays {
            if let Some(reason) = &overlay.analysis_failed {
                warnings.push(format!(
                    "could not analyze `{}` in {} — {reason}",
                    overlay.path.display(),
                    &hash[..hash.len().min(7)]
                ));
            }
        }

        summaries.push(CommitSummary {
            hash: hash.to_string(),
            timestamp,
            message,
            author,
            overlays,
        });
    }

    // Filter by scope if set.
    //
    // Log mode keeps its own filter because it applies per commit, but it must
    // use the same matcher as `diff()` — this copy accepted only files, and
    // matched them by loose suffix, so `--scope src/diff --log` came back empty
    // while `--scope src/diff` errored. Two different wrong answers to the same
    // question.
    // Built once, outside the loop: it can shell out to git.
    let candidates = match scope.map(scope_request) {
        None | Some(ScopeRequest::Everything) => None,
        Some(ScopeRequest::Paths(c)) => Some(c),
    };
    if let Some(candidates) = candidates {
        for summary in &mut summaries {
            let paths: Vec<String> = summary
                .overlays
                .iter()
                .map(|o| normalize_path(&o.path.to_string_lossy()))
                .collect();
            // Every match, not the first. Log mode is a filter, not a picker —
            // taking one match here dropped a real change to the very file the
            // caller scoped to, and relabelled the commit with a different one.
            let keep: HashSet<usize> = match select_scope(&paths, &candidates) {
                Some((m, _)) => m.all().into_iter().collect(),
                None => HashSet::new(),
            };
            let mut i = 0;
            summary.overlays.retain(|_| {
                let keep_this = keep.contains(&i);
                i += 1;
                keep_this
            });
        }
        summaries.retain(|s| !s.overlays.is_empty());
    }

    if summaries.is_empty() {
        return Ok("No commits found.".to_string());
    }

    Ok(format::format_log(&summaries, range, &warnings, budget))
}

/// Git's empty tree object — the implicit "before" of a root commit.
const EMPTY_TREE_SHA1: &str = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";

/// The empty tree's object id **in this repository**.
///
/// Not a constant: the well-known `4b825dc…` is the SHA-1 empty tree, and a
/// repository created with `--object-format=sha256` has a different one
/// (`6ef19b4…`). Passing the SHA-1 id there fails with "unknown revision", so
/// the parentless-commit and unborn-HEAD paths would both silently stop working
/// on such a repo. `git hash-object` computes whichever the repo uses.
///
/// Falls back to the SHA-1 id if git cannot be run, which is the same value the
/// hardcoded constant had, so nothing regresses relative to it.
fn empty_tree() -> String {
    Command::new("git")
        .args(["hash-object", "-t", "tree", "--stdin"])
        .stdin(std::process::Stdio::null())
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| EMPTY_TREE_SHA1.to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;
    // Serializing the cwd against *these* tests was never enough — `crate::testlock::cwd` is now
    // the one lock, and `mcp::bom_surfaces` takes it too. See #95 and the note on `run_diff_in`.
    //
    // `//` and not `///`: rustdoc concatenates doc attributes across a blank line, so the `///`
    // this replaced had silently become the first paragraph of `setup_test_repo`'s documentation.

    /// Create a test git repo with an initial commit containing a Rust file.
    fn setup_test_repo() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("failed to create tempdir");
        let p = dir.path();

        git(p, &["init"]);
        git(p, &["config", "user.email", "test@test.com"]);
        git(p, &["config", "user.name", "Test"]);

        let src = p.join("src");
        fs::create_dir_all(&src).unwrap();

        let main_rs = src.join("main.rs");
        fs::write(
            &main_rs,
            "fn hello() {\n    println!(\"hello\");\n}\n\nfn goodbye() {\n    println!(\"bye\");\n}\n\nfn main() {\n    hello();\n    goodbye();\n}\n",
        )
        .unwrap();

        git(p, &["add", "-A"]);
        git(p, &["commit", "-m", "initial"]);

        dir
    }

    /// Run a git command in the given directory.
    fn git(dir: &Path, args: &[&str]) -> String {
        let output = Command::new("git")
            .args(args)
            .current_dir(dir)
            .env("GIT_AUTHOR_NAME", "Test")
            .env("GIT_AUTHOR_EMAIL", "test@test.com")
            .env("GIT_COMMITTER_NAME", "Test")
            .env("GIT_COMMITTER_EMAIL", "test@test.com")
            .output()
            .expect("failed to run git");
        String::from_utf8_lossy(&output.stdout).into_owned()
    }

    /// Restores the process cwd on the way out, panic or not.
    ///
    /// Without this, a panic inside `diff()` unwinds past the restore and releases the lock with
    /// the cwd still inside a fixture that is about to be deleted — and the next holder is
    /// `bom_surfaces`, holding it across its whole table, which then fails *deterministically* with
    /// the #95 error. Test profiles unwind (`panic = "abort"` is set for `[profile.release]` only),
    /// so that path is live. It is also what makes `testlock::cwd`'s poison-ignoring honest: with
    /// the restore guaranteed, there really is nothing a panic can leave inconsistent here.
    struct RestoreCwd(std::path::PathBuf);

    impl Drop for RestoreCwd {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.0);
        }
    }

    /// Run diff() from within the test repo directory, serialized via the shared cwd lock.
    ///
    /// The lock has to be *shared* with every test that reads the cwd, not just with the ones that
    /// move it. Measured for #95: `mcp::bom_surfaces` would spawn `git` with the inherited cwd
    /// pointing at `dir`, and git's repository discovery would then race that fixture's deletion —
    /// `.git` present, `.git/HEAD` already gone. Hence the two symptoms, "not a git repository" and
    /// "your current branch appears to be broken", neither of which looks like a cwd race.
    ///
    /// What the lock buys is precisely that **no such child is ever spawned while the cwd is a
    /// fixture**. The `TempDir` drop happens in the test body after this returns, necessarily
    /// outside the lock, and deliberately so — it does not need to be inside it, and moving it
    /// there would not make anything safer.
    fn run_diff_in(
        dir: &Path,
        source: &DiffSource,
        scope: Option<&str>,
        search: Option<&str>,
        blast: bool,
        budget: Option<u64>,
    ) -> Result<String, String> {
        let _lock = crate::testlock::cwd();
        let _restore = RestoreCwd(std::env::current_dir().unwrap());
        std::env::set_current_dir(dir).unwrap();
        diff(source, scope, search, blast, 0, budget)
    }

    // 1. test_empty_diff
    #[test]
    fn test_empty_diff() {
        let dir = setup_test_repo();
        let result = run_diff_in(
            dir.path(),
            &DiffSource::GitUncommitted,
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert_eq!(result, "No changes.");
    }

    // 2. test_overview_modified
    #[test]
    fn test_overview_modified() {
        let dir = setup_test_repo();
        let main_rs = dir.path().join("src/main.rs");
        let content = fs::read_to_string(&main_rs).unwrap();
        fs::write(
            &main_rs,
            content.replace("println!(\"hello\")", "println!(\"hi there\")"),
        )
        .unwrap();

        let result = run_diff_in(
            dir.path(),
            &DiffSource::GitUncommitted,
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert!(result.contains("[~]"), "expected [~] marker in:\n{result}");
    }

    // 3. test_overview_added
    #[test]
    fn test_overview_added() {
        let dir = setup_test_repo();
        let main_rs = dir.path().join("src/main.rs");
        let mut content = fs::read_to_string(&main_rs).unwrap();
        content.push_str("\nfn new_function() {\n    println!(\"new\");\n}\n");
        fs::write(&main_rs, content).unwrap();

        let result = run_diff_in(
            dir.path(),
            &DiffSource::GitUncommitted,
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert!(result.contains("[+]"), "expected [+] marker in:\n{result}");
    }

    // 4. test_overview_deleted
    #[test]
    fn test_overview_deleted() {
        let dir = setup_test_repo();
        let main_rs = dir.path().join("src/main.rs");
        // Remove the goodbye function entirely.
        fs::write(
            &main_rs,
            "fn hello() {\n    println!(\"hello\");\n}\n\nfn main() {\n    hello();\n}\n",
        )
        .unwrap();

        let result = run_diff_in(
            dir.path(),
            &DiffSource::GitUncommitted,
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert!(result.contains("[-]"), "expected [-] marker in:\n{result}");
    }

    // 5. test_overview_signature_changed
    #[test]
    fn test_overview_signature_changed() {
        let dir = setup_test_repo();
        let main_rs = dir.path().join("src/main.rs");
        let content = fs::read_to_string(&main_rs).unwrap();
        // Change hello() to hello(name: &str)
        let new_content = content
            .replace("fn hello() {", "fn hello(name: &str) {")
            .replace("println!(\"hello\")", "println!(\"hello {}\", name)")
            .replace("hello();", "hello(\"world\");");
        fs::write(&main_rs, new_content).unwrap();

        let result = run_diff_in(
            dir.path(),
            &DiffSource::GitUncommitted,
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert!(
            result.contains("[~:sig]"),
            "expected [~:sig] marker in:\n{result}"
        );
    }

    // 6. test_file_detail_scope
    #[test]
    fn test_file_detail_scope() {
        let dir = setup_test_repo();
        let main_rs = dir.path().join("src/main.rs");
        let content = fs::read_to_string(&main_rs).unwrap();
        fs::write(
            &main_rs,
            content.replace("println!(\"hello\")", "println!(\"hi\")"),
        )
        .unwrap();

        let result = run_diff_in(
            dir.path(),
            &DiffSource::GitUncommitted,
            Some("src/main.rs"),
            None,
            false,
            None,
        )
        .unwrap();
        assert!(
            result.contains("# Diff: src/main.rs"),
            "expected file detail header in:\n{result}"
        );
        assert!(
            result.contains("symbols touched"),
            "expected symbols touched in:\n{result}"
        );
    }

    // 7. test_function_detail_scope
    #[test]
    fn test_function_detail_scope() {
        let dir = setup_test_repo();
        let main_rs = dir.path().join("src/main.rs");
        let content = fs::read_to_string(&main_rs).unwrap();
        fs::write(
            &main_rs,
            content.replace("println!(\"hello\")", "println!(\"hi\")"),
        )
        .unwrap();

        let result = run_diff_in(
            dir.path(),
            &DiffSource::GitUncommitted,
            Some("src/main.rs:hello"),
            None,
            false,
            None,
        )
        .unwrap();
        assert!(
            result.contains("hello"),
            "expected hello function in:\n{result}"
        );
    }

    // 8. test_staged_diff
    #[test]
    fn test_staged_diff() {
        let dir = setup_test_repo();
        let main_rs = dir.path().join("src/main.rs");
        let content = fs::read_to_string(&main_rs).unwrap();
        fs::write(
            &main_rs,
            content.replace("println!(\"hello\")", "println!(\"staged\")"),
        )
        .unwrap();
        git(dir.path(), &["add", "src/main.rs"]);

        let result =
            run_diff_in(dir.path(), &DiffSource::GitStaged, None, None, false, None).unwrap();
        assert!(
            result.contains("main.rs") || result.contains("[~]"),
            "expected staged changes in:\n{result}"
        );
    }

    // 9. test_ref_diff
    #[test]
    fn test_ref_diff() {
        let dir = setup_test_repo();
        let main_rs = dir.path().join("src/main.rs");
        let content = fs::read_to_string(&main_rs).unwrap();
        fs::write(
            &main_rs,
            content.replace("println!(\"hello\")", "println!(\"ref\")"),
        )
        .unwrap();
        git(dir.path(), &["add", "-A"]);
        git(dir.path(), &["commit", "-m", "change hello"]);

        let result = run_diff_in(
            dir.path(),
            &DiffSource::GitRef("HEAD~1..HEAD".to_string()),
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert!(
            result.contains("main.rs"),
            "expected main.rs in ref diff:\n{result}"
        );
    }

    // 10. test_generated_file
    #[test]
    fn test_generated_file() {
        let dir = setup_test_repo();
        let lock = dir.path().join("package-lock.json");
        fs::write(&lock, "{}").unwrap();
        git(dir.path(), &["add", "-A"]);
        git(dir.path(), &["commit", "-m", "add lock"]);

        fs::write(&lock, "{ \"version\": 2 }").unwrap();

        let result = run_diff_in(
            dir.path(),
            &DiffSource::GitUncommitted,
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert!(
            result.contains("generated"),
            "expected 'generated' in:\n{result}"
        );
    }

    // 11. test_multiple_files
    #[test]
    fn test_multiple_files() {
        let dir = setup_test_repo();
        let main_rs = dir.path().join("src/main.rs");
        let content = fs::read_to_string(&main_rs).unwrap();
        fs::write(
            &main_rs,
            content.replace("println!(\"hello\")", "println!(\"hi\")"),
        )
        .unwrap();

        let lib_rs = dir.path().join("src/lib.rs");
        fs::write(&lib_rs, "pub fn lib_fn() {\n    42\n}\n").unwrap();
        git(dir.path(), &["add", "src/lib.rs"]);
        git(dir.path(), &["commit", "-m", "add lib"]);
        fs::write(&lib_rs, "pub fn lib_fn() {\n    99\n}\n").unwrap();

        let result = run_diff_in(
            dir.path(),
            &DiffSource::GitUncommitted,
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert!(result.contains("main.rs"), "expected main.rs in:\n{result}");
        assert!(result.contains("lib.rs"), "expected lib.rs in:\n{result}");
        assert!(
            result.contains("2 files"),
            "expected '2 files' in:\n{result}"
        );
    }

    // 12. test_search_filter
    #[test]
    fn test_search_filter() {
        let dir = setup_test_repo();
        let main_rs = dir.path().join("src/main.rs");
        let content = fs::read_to_string(&main_rs).unwrap();
        // Modify both functions.
        let new_content = content
            .replace("println!(\"hello\")", "println!(\"UNIQUE_MARKER\")")
            .replace("println!(\"bye\")", "println!(\"other change\")");
        fs::write(&main_rs, new_content).unwrap();

        let result = run_diff_in(
            dir.path(),
            &DiffSource::GitUncommitted,
            None,
            Some("UNIQUE_MARKER"),
            false,
            None,
        )
        .unwrap();
        assert!(
            result.contains("hello"),
            "expected hello (matching) in:\n{result}"
        );
    }

    // 13. test_search_no_matches
    #[test]
    fn test_search_no_matches() {
        let dir = setup_test_repo();
        let main_rs = dir.path().join("src/main.rs");
        let content = fs::read_to_string(&main_rs).unwrap();
        fs::write(
            &main_rs,
            content.replace("println!(\"hello\")", "println!(\"hi\")"),
        )
        .unwrap();

        let result = run_diff_in(
            dir.path(),
            &DiffSource::GitUncommitted,
            None,
            Some("NONEXISTENT_TERM_XYZ"),
            false,
            None,
        )
        .unwrap();
        assert!(
            result.contains("No changes matching"),
            "expected no-match message in:\n{result}"
        );
    }

    /// A repo with changes in two sibling directories, so a directory scope has
    /// something to include *and* something to exclude.
    fn setup_two_dir_repo() -> tempfile::TempDir {
        let dir = setup_test_repo();
        let p = dir.path();
        let tools = p.join("tools/hooks");
        fs::create_dir_all(&tools).unwrap();
        fs::write(tools.join("prefer.py"), "def old():\n    pass\n").unwrap();
        fs::write(tools.join("other.py"), "def other_old():\n    pass\n").unwrap();
        git(p, &["add", "-A"]);
        git(p, &["commit", "-m", "add tools"]);

        fs::write(
            tools.join("prefer.py"),
            "def old():\n    pass\n\ndef prefer_added():\n    pass\n",
        )
        .unwrap();
        fs::write(
            tools.join("other.py"),
            "def other_old():\n    pass\n\ndef other_added():\n    pass\n",
        )
        .unwrap();
        let main_rs = p.join("src/main.rs");
        let content = fs::read_to_string(&main_rs).unwrap();
        fs::write(&main_rs, format!("{content}\nfn src_side_added() {{}}\n")).unwrap();
        dir
    }

    // 14. test_directory_scope
    #[test]
    fn test_directory_scope() {
        let dir = setup_two_dir_repo();

        // Issue #110: this errored with "file 'tools/hooks' not found in diff"
        // while an exact file scope in the same directory worked.
        let result = run_diff_in(
            dir.path(),
            &DiffSource::GitUncommitted,
            Some("tools/hooks"),
            None,
            false,
            None,
        )
        .expect("a directory scope must be accepted");

        assert!(
            result.contains("prefer_added") && result.contains("other_added"),
            "every changed file beneath the directory must appear:\n{result}"
        );
        assert!(
            !result.contains("src_side_added"),
            "a directory scope must exclude changes outside it:\n{result}"
        );
        assert!(
            result.contains("2 files"),
            "header must count only the scoped files:\n{result}"
        );
    }

    // 15. test_directory_scope_in_log_mode
    #[test]
    fn test_directory_scope_in_log_mode() {
        let dir = setup_two_dir_repo();
        git(dir.path(), &["add", "-A"]);
        git(dir.path(), &["commit", "-m", "second round"]);

        // Log mode filters scope through its own code path. It accepted only
        // files, so a directory scope came back "No commits found." — a
        // different wrong answer to the same question `diff()` errored on.
        let result = run_diff_in(
            dir.path(),
            &DiffSource::Log("HEAD~1..HEAD".to_string()),
            Some("tools/hooks"),
            None,
            false,
            None,
        )
        .expect("log mode must accept a directory scope");

        assert!(
            result.contains("prefer_added") && result.contains("other_added"),
            "log must report the scoped directory's changes:\n{result}"
        );
        assert!(
            !result.contains("src_side_added"),
            "log scope must exclude changes outside the directory:\n{result}"
        );
    }

    // 16. test_scope_naming_a_file_and_a_directory_errors
    #[test]
    fn test_scope_naming_a_file_and_a_directory_errors() {
        // Its own fixture: `setup_two_dir_repo` leaves its edits uncommitted,
        // so committing a new file on top of it would swallow them and leave
        // only one changed path — a fixture that cannot express the collision.
        let dir = setup_test_repo();
        let p = dir.path();
        fs::create_dir_all(p.join("scripts")).unwrap();
        fs::create_dir_all(p.join("tools/hooks")).unwrap();
        fs::write(p.join("scripts/hooks"), "#!/bin/sh\necho old\n").unwrap();
        fs::write(p.join("tools/hooks/a.py"), "def a():\n    pass\n").unwrap();
        fs::write(p.join("tools/hooks/b.py"), "def b():\n    pass\n").unwrap();
        git(p, &["add", "-A"]);
        git(p, &["commit", "-m", "baseline"]);

        // A file whose basename equals the directory everyone means by `hooks`,
        // and the directory itself — both changed, neither committed.
        fs::write(p.join("scripts/hooks"), "#!/bin/sh\necho new\n").unwrap();
        fs::write(
            p.join("tools/hooks/a.py"),
            "def a():\n    pass\n\ndef a_added():\n    pass\n",
        )
        .unwrap();
        fs::write(
            p.join("tools/hooks/b.py"),
            "def b():\n    pass\n\ndef b_added():\n    pass\n",
        )
        .unwrap();

        // Answering with just the file dropped two changed Python files and
        // reported `+0/−0` — reading as "nothing happened in hooks".
        let result = run_diff_in(
            p,
            &DiffSource::GitUncommitted,
            Some("hooks"),
            None,
            false,
            None,
        );
        let err = result.expect_err("a scope naming two different things must not pick one");
        assert!(err.contains("ambiguous"), "{err}");
        assert!(
            err.contains("scripts/hooks") && err.contains("tools/hooks"),
            "the error must name both interpretations: {err}"
        );
    }

    // 17. test_directory_scope_excludes_outside_conflicts
    #[test]
    fn test_directory_scope_excludes_outside_conflicts() {
        let dir = setup_two_dir_repo();
        // A conflict in a file the scope excludes.
        let main_rs = dir.path().join("src/main.rs");
        let content = fs::read_to_string(&main_rs).unwrap();
        fs::write(
            &main_rs,
            format!("{content}\n<<<<<<< HEAD\nfn ours() {{}}\n=======\nfn theirs() {{}}\n>>>>>>> other\n"),
        )
        .unwrap();

        let result = run_diff_in(
            dir.path(),
            &DiffSource::GitUncommitted,
            Some("tools/hooks"),
            None,
            false,
            None,
        )
        .unwrap();

        // The header says `(tools/hooks)`; a conflict block from `src/` under it
        // makes the body contradict the header.
        assert!(
            !result.contains("Conflicts"),
            "a scoped diff must not report conflicts outside the scope:\n{result}"
        );

        // But the conflict is still reported when nothing is scoped out.
        let unscoped = run_diff_in(
            dir.path(),
            &DiffSource::GitUncommitted,
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert!(
            unscoped.contains("Conflicts"),
            "the unscoped diff must still report it:\n{unscoped}"
        );
    }

    // 18. test_absolute_scope
    #[test]
    fn test_absolute_scope() {
        let dir = setup_two_dir_repo();

        // `prompts/mcp-base.md` tells agents: "DO NOT pass a relative path or
        // scope without also setting root (absolute)". So absolute is the form
        // `tilth_diff` actually receives — and every one of them missed, because
        // git's diff paths are repo-relative. On Windows the drive-letter colon
        // also split the scope, turning it into a request for the file `C`.
        // Canonicalize, don't just join. `git rev-parse --show-toplevel`
        // reports the physical path, and a TempDir is reached through a
        // symlink on macOS (`/var` → `/private/var`), so a raw `dir.path()`
        // would never match the root there. Canonicalizing is also what an MCP
        // client does before sending an absolute scope — hence the `\\?\`
        // prefix handling in `normalize_path`.
        let root = dir.path().canonicalize().unwrap();
        let abs_file = root.join("tools/hooks/prefer.py");
        let result = run_diff_in(
            dir.path(),
            &DiffSource::GitUncommitted,
            Some(&abs_file.to_string_lossy()),
            None,
            false,
            None,
        )
        .expect("an absolute file scope must be accepted");
        assert!(
            result.contains("prefer_added"),
            "absolute file scope must resolve:\n{result}"
        );

        let abs_dir = root.join("tools/hooks");
        let result = run_diff_in(
            dir.path(),
            &DiffSource::GitUncommitted,
            Some(&abs_dir.to_string_lossy()),
            None,
            false,
            None,
        )
        .expect("an absolute directory scope must be accepted");
        assert!(
            result.contains("prefer_added") && result.contains("other_added"),
            "absolute directory scope must resolve:\n{result}"
        );
        assert!(
            !result.contains("src_side_added"),
            "absolute directory scope must still exclude outside changes:\n{result}"
        );

        // The label reports the resolved relative path, not the caller's
        // absolute one — an agent's temp-dir prefix is noise in the output.
        assert!(
            result.contains("(tools/hooks)"),
            "expected the resolved relative scope in the header:\n{result}"
        );

        // An absolute scope that IS the repo root asks the unscoped question.
        // An agent resolving "the repo" to its path used to get an error.
        let result = run_diff_in(
            dir.path(),
            &DiffSource::GitUncommitted,
            Some(&root.to_string_lossy()),
            None,
            false,
            None,
        )
        .expect("the repo root is a valid scope");
        assert!(
            result.contains("prefer_added") && result.contains("src_side_added"),
            "a root scope must cover the whole repo:\n{result}"
        );
    }

    // 19. test_absolute_scope_with_symbol
    #[test]
    fn test_absolute_scope_with_symbol() {
        let dir = setup_two_dir_repo();
        // Canonicalized for the same reason as `test_absolute_scope`.
        let abs = dir
            .path()
            .canonicalize()
            .unwrap()
            .join("tools/hooks/prefer.py");

        // `<abs path>:<symbol>` must split at the symbol, not the drive letter.
        let result = run_diff_in(
            dir.path(),
            &DiffSource::GitUncommitted,
            Some(&format!("{}:prefer_added", abs.to_string_lossy())),
            None,
            false,
            None,
        )
        .expect("absolute file:function scope must be accepted");
        assert!(
            result.contains("prefer_added"),
            "expected the named symbol:\n{result}"
        );
    }

    fn paths(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// Resolve a relative scope. Relative never consults git, so this needs no
    /// repository around it.
    fn sel(p: &[String], scope: &str) -> Option<ScopeMatch> {
        match scope_request(scope) {
            ScopeRequest::Paths(c) => select_scope(p, &c).map(|(m, _)| m),
            ScopeRequest::Everything => None,
        }
    }

    fn files_of(m: &Option<ScopeMatch>) -> Vec<usize> {
        m.as_ref().map(|m| m.files.clone()).unwrap_or_default()
    }

    fn under_of(m: &Option<ScopeMatch>) -> Vec<usize> {
        m.as_ref().map(|m| m.under.clone()).unwrap_or_default()
    }

    #[test]
    fn scope_matches_on_component_boundaries() {
        let p = paths(&["src/diff/parse.rs", "tools/hooks/prefer_tools.py"]);

        // Exact and component-boundary suffix.
        assert_eq!(files_of(&sel(&p, "src/diff/parse.rs")), vec![0]);
        assert_eq!(files_of(&sel(&p, "parse.rs")), vec![0]);
        assert_eq!(files_of(&sel(&p, "diff/parse.rs")), vec![0]);

        // A bare substring is not a path. `arse.rs` used to select parse.rs and
        // answer confidently about a file nobody asked for.
        assert!(sel(&p, "arse.rs").is_none());
        assert!(sel(&p, "refer_tools.py").is_none());
        assert!(sel(&p, "rc").is_none());
    }

    #[test]
    fn scope_accepts_directories() {
        let p = paths(&[
            "tools/hooks/prefer_tools.py",
            "tools/hooks/other.py",
            "src/main.rs",
        ]);

        // The reported case: a directory selects everything beneath it.
        assert_eq!(under_of(&sel(&p, "tools/hooks")), vec![0, 1]);
        // Trailing slash is the same request.
        assert_eq!(under_of(&sel(&p, "tools/hooks/")), vec![0, 1]);
        // Directories get the same suffix latitude files do.
        assert_eq!(under_of(&sel(&p, "hooks")), vec![0, 1]);
        // Windows separators normalize to git's.
        assert_eq!(under_of(&sel(&p, r"tools\hooks")), vec![0, 1]);
        // A directory that changed nothing is a miss, not an empty success.
        assert!(sel(&p, "tools/other").is_none());
        // A file scope still resolves to its file.
        assert_eq!(files_of(&sel(&p, "src/main.rs")), vec![2]);
    }

    #[test]
    fn anchored_matches_beat_suffix_matches() {
        // A monorepo where the same relative path exists in several trees. This
        // is the shape that made a fully-qualified scope answer about the wrong
        // file: `src/util.rs` was suffix-matched, and `a/src/util.rs` sorts
        // first, so it won.
        let p = paths(&[
            "a/src/util.rs",
            "b/src/util.rs",
            "src/util.rs",
            "vendor/src/util.rs",
        ]);

        // Anchored at the repo root: exactly the one git's pathspec would give.
        assert_eq!(files_of(&sel(&p, "src/util.rs")), vec![2]);
        assert_eq!(under_of(&sel(&p, "src")), vec![2]);

        // A scope that is anchored nowhere still gets the suffix pass — but now
        // it reports every match rather than silently picking the first.
        assert_eq!(files_of(&sel(&p, "util.rs")), vec![0, 1, 2, 3]);

        // An inner tree is still reachable by qualifying it.
        assert_eq!(files_of(&sel(&p, "a/src/util.rs")), vec![0]);
        assert_eq!(under_of(&sel(&p, "vendor/src")), vec![3]);
    }

    #[test]
    fn a_file_never_silently_shadows_a_directory() {
        // `scripts/hooks` is a file; `tools/hooks/` is a directory. `hooks`
        // names both, and answering with just the file dropped two changed
        // Python files while reporting `+0/−0` — "nothing happened in hooks".
        let p = paths(&["scripts/hooks", "tools/hooks/a.py", "tools/hooks/b.py"]);
        let m = sel(&p, "hooks").expect("both interpretations match");
        assert_eq!(m.files, vec![0]);
        assert_eq!(m.under, vec![1, 2]);

        // The caller sees the collision rather than one arbitrary half of it.
        let err = scope_ambiguous_error(&p, "hooks", &m);
        assert!(err.contains("scripts/hooks"), "{err}");
        assert!(err.contains("tools/hooks/a.py"), "{err}");
    }

    #[test]
    fn a_scope_naming_the_repo_itself_means_everything() {
        // An agent resolving "the repo" to its root path, or passing `.`, is
        // asking the same question as passing no scope at all.
        assert!(matches!(scope_request("."), ScopeRequest::Everything));
        assert!(matches!(scope_request("./"), ScopeRequest::Everything));
        assert!(matches!(scope_request(""), ScopeRequest::Everything));
    }

    #[test]
    fn extended_length_prefixes_normalize_away() {
        // `std::fs::canonicalize` emits this on Windows, so an MCP client that
        // canonicalizes sends it verbatim. Left in place it can never match
        // git's root.
        assert_eq!(normalize_path(r"\\?\C:\dev\t\src"), "C:/dev/t/src");
        assert_eq!(
            normalize_path(r"\\?\UNC\server\share\r"),
            "//server/share/r"
        );
        assert_eq!(normalize_path("src/diff/"), "src/diff");
    }

    #[test]
    fn relative_scopes_never_consult_git() {
        // `strip_repo_root` shells out. A relative scope can never live under
        // the repo root, so it must bail before paying for that — log mode
        // resolves a scope per commit, and this is the same per-item-subprocess
        // shape as the merge-base call fixed in #112.
        assert!(!looks_absolute("src/main.rs"));
        assert!(!looks_absolute("tools/hooks"));
        assert!(looks_absolute("/home/u/repo/src/main.rs"));
        assert!(looks_absolute(r"C:\dev\t\src\main.rs"));
        assert!(looks_absolute("C:/dev/t/src/main.rs"));

        // Holds with no repository around it, which is the proof there was no
        // git call: a relative scope yields exactly one candidate, itself.
        assert!(matches!(
            strip_repo_root("src/main.rs"),
            RootRelative::Outside
        ));
        let one = |s: &str| match scope_request(s) {
            ScopeRequest::Paths(c) => c,
            ScopeRequest::Everything => panic!("{s} should not be whole-repo"),
        };
        assert_eq!(one("src/main.rs"), vec!["src/main.rs"]);
        assert_eq!(one(r"tools\hooks\"), vec!["tools/hooks"]);
    }

    #[test]
    fn path_symbol_split_survives_drive_letters() {
        // The `file:function` form.
        assert_eq!(
            split_path_symbol("src/main.rs:hello"),
            Some(("src/main.rs", "hello"))
        );
        assert_eq!(
            split_path_symbol("C:/dev/t/src/main.rs:hello"),
            Some(("C:/dev/t/src/main.rs", "hello"))
        );

        // A bare absolute path is NOT a file:function request. Splitting on the
        // first colon made every absolute Windows scope a request for the file
        // `C`, which is what `file 'C' not found in diff` was.
        assert_eq!(split_path_symbol("C:/dev/t/src/main.rs"), None);
        assert_eq!(split_path_symbol(r"C:\dev\t\src\main.rs"), None);
        assert_eq!(split_path_symbol("C:"), None);
        assert_eq!(split_path_symbol("src/main.rs"), None);
    }

    #[test]
    fn scope_miss_names_the_changed_files() {
        let p = paths(&["src/a.rs", "src/b.rs"]);
        let err = scope_miss_error(&p, "src/nope");
        assert!(
            err.contains("src/a.rs") && err.contains("src/b.rs"),
            "{err}"
        );

        // Long diffs are truncated with a count, not silently cut.
        let many: Vec<String> = (0..25).map(|i| format!("src/f{i}.rs")).collect();
        let err = scope_miss_error(&many, "nope");
        assert!(err.contains("(+15 more)"), "{err}");
    }

    // 20. test_file_scope_not_found
    #[test]
    fn test_file_scope_not_found() {
        let dir = setup_test_repo();
        let main_rs = dir.path().join("src/main.rs");
        let content = fs::read_to_string(&main_rs).unwrap();
        fs::write(
            &main_rs,
            content.replace("println!(\"hello\")", "println!(\"hi\")"),
        )
        .unwrap();

        let result = run_diff_in(
            dir.path(),
            &DiffSource::GitUncommitted,
            Some("nonexistent.rs"),
            None,
            false,
            None,
        );
        assert!(result.is_err(), "expected error for missing file scope");
        let err = result.unwrap_err();
        assert!(
            err.contains("matched no changed file"),
            "expected a scope-miss error, got {err:?}"
        );
        // The old message named only the thing that was missing, which left no
        // way to tell a typo from a genuinely untouched path. Name what did
        // change instead.
        assert!(
            err.contains("src/main.rs"),
            "a scope miss must name the changed files, got {err:?}"
        );
    }

    // 21. test_patch_file
    #[test]
    fn test_patch_file() {
        let dir = setup_test_repo();
        let patch = dir.path().join("test.patch");
        let patch_content = "\
diff --git a/src/main.rs b/src/main.rs
--- a/src/main.rs
+++ b/src/main.rs
@@ -1,3 +1,3 @@
 fn hello() {
-    println!(\"hello\");
+    println!(\"patched\");
 }
";
        fs::write(&patch, patch_content).unwrap();

        let result = run_diff_in(
            dir.path(),
            &DiffSource::Patch(patch.clone()),
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert!(
            result.contains("main.rs"),
            "expected main.rs in patch result:\n{result}"
        );
    }

    // 22. test_file_to_file
    #[test]
    fn test_file_to_file() {
        let dir = setup_test_repo();
        let file_a = dir.path().join("a.txt");
        let file_b = dir.path().join("b.txt");
        fs::write(&file_a, "line one\nline two\n").unwrap();
        fs::write(&file_b, "line one\nline three\n").unwrap();

        let result = run_diff_in(
            dir.path(),
            &DiffSource::Files(file_a, file_b),
            None,
            None,
            false,
            None,
        )
        .unwrap();
        // The diff should contain something — the files differ.
        assert!(
            !result.contains("No changes"),
            "expected changes between files:\n{result}"
        );
    }

    // 23. test_log_mode
    #[test]
    fn test_log_mode() {
        let dir = setup_test_repo();
        let main_rs = dir.path().join("src/main.rs");

        // Make a second commit.
        let content = fs::read_to_string(&main_rs).unwrap();
        fs::write(
            &main_rs,
            content.replace("println!(\"hello\")", "println!(\"log test\")"),
        )
        .unwrap();
        git(dir.path(), &["add", "-A"]);
        git(dir.path(), &["commit", "-m", "second commit"]);

        let result = run_diff_in(
            dir.path(),
            &DiffSource::Log("HEAD~1..HEAD".to_string()),
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert!(
            result.contains("# Log:"),
            "expected log header in:\n{result}"
        );
        assert!(
            result.contains("second commit"),
            "expected commit message in:\n{result}"
        );
    }

    // 24. test_resolve_source_variants
    #[test]
    fn test_resolve_source_variants() {
        // Default → uncommitted.
        assert!(matches!(
            resolve_source(None, None, None, None, None).unwrap(),
            DiffSource::GitUncommitted
        ));

        // Staged.
        assert!(matches!(
            resolve_source(Some("staged"), None, None, None, None).unwrap(),
            DiffSource::GitStaged
        ));

        // Working.
        assert!(matches!(
            resolve_source(Some("working"), None, None, None, None).unwrap(),
            DiffSource::GitUncommitted
        ));

        // Ref.
        match resolve_source(Some("HEAD~3..HEAD"), None, None, None, None).unwrap() {
            DiffSource::GitRef(r) => assert_eq!(r, "HEAD~3..HEAD"),
            other => panic!("expected GitRef, got {other:?}"),
        }

        // Files.
        match resolve_source(None, Some("a.rs"), Some("b.rs"), None, None).unwrap() {
            DiffSource::Files(a, b) => {
                assert_eq!(a, PathBuf::from("a.rs"));
                assert_eq!(b, PathBuf::from("b.rs"));
            }
            other => panic!("expected Files, got {other:?}"),
        }

        // Error: only one of a/b.
        assert!(resolve_source(None, Some("a.rs"), None, None, None).is_err());

        // Patch.
        match resolve_source(None, None, None, Some("test.patch"), None).unwrap() {
            DiffSource::Patch(p) => assert_eq!(p, PathBuf::from("test.patch")),
            other => panic!("expected Patch, got {other:?}"),
        }

        // Log.
        match resolve_source(None, None, None, None, Some("HEAD~5..HEAD")).unwrap() {
            DiffSource::Log(r) => assert_eq!(r, "HEAD~5..HEAD"),
            other => panic!("expected Log, got {other:?}"),
        }

        // Patch takes priority over source.
        assert!(matches!(
            resolve_source(Some("staged"), None, None, Some("x.patch"), None).unwrap(),
            DiffSource::Patch(_)
        ));
    }

    /// Diverge `base` and `feat` from a shared root, one new function on each
    /// side, so that merge-base(base, feat) is neither branch tip. That is the
    /// only topology where `base..feat` and `base...feat` disagree.
    fn setup_diverged_repo() -> tempfile::TempDir {
        let dir = setup_test_repo();
        let p = dir.path();
        // `git init` picks master or main depending on the host's config; pin it.
        git(p, &["branch", "-M", "base"]);
        let main_rs = p.join("src/main.rs");

        git(p, &["checkout", "-b", "feat"]);
        let content = fs::read_to_string(&main_rs).unwrap();
        fs::write(
            &main_rs,
            format!("{content}\nfn feat_only() {{\n    println!(\"feat\");\n}}\n"),
        )
        .unwrap();
        git(p, &["add", "-A"]);
        git(p, &["commit", "-m", "feat side"]);

        git(p, &["checkout", "base"]);
        let content = fs::read_to_string(&main_rs).unwrap();
        fs::write(
            &main_rs,
            format!("{content}\nfn base_only() {{\n    println!(\"base\");\n}}\n"),
        )
        .unwrap();
        git(p, &["add", "-A"]);
        git(p, &["commit", "-m", "base side"]);

        git(p, &["checkout", "feat"]);
        dir
    }

    /// A repo whose HEAD commit bumps a gitlink whose target object is absent.
    ///
    /// git happily diffs the pointer, but `git show <rev>:<gitlink>` fails, so
    /// the overlay is abandoned — the cheapest reachable way to reach that path
    /// without wiring up a real submodule. Ordinary for anyone diffing a repo
    /// with submodules they have not initialized.
    fn setup_dangling_gitlink_repo() -> tempfile::TempDir {
        let dir = setup_test_repo();
        let p = dir.path();
        git(
            p,
            &[
                "update-index",
                "--add",
                "--cacheinfo",
                "160000,0000000000000000000000000000000000000001,sub",
            ],
        );
        git(p, &["commit", "-m", "add submodule pointer"]);
        git(
            p,
            &[
                "update-index",
                "--cacheinfo",
                "160000,0000000000000000000000000000000000000002,sub",
            ],
        );
        git(p, &["commit", "-m", "bump submodule"]);
        dir
    }

    // 25. test_search_keeps_unreadable_file_warning
    #[test]
    fn test_search_keeps_unreadable_file_warning() {
        let dir = setup_dangling_gitlink_repo();

        // `filter_by_search` always drops an abandoned overlay — no symbols and
        // no hunks means nothing can match — and the empty-result early return
        // is the one path that never renders `warnings`. Collecting them before
        // the filter is necessary but was not sufficient.
        let result = run_diff_in(
            dir.path(),
            &DiffSource::GitRef("HEAD~1..HEAD".to_string()),
            None,
            Some("Subproject"),
            false,
            None,
        )
        .unwrap();

        assert!(
            result.contains("could not analyze"),
            "a file we could not read must not be reported as simply absent:\n{result}"
        );
    }

    // 26. test_log_spans_the_root_commit
    #[test]
    fn test_log_spans_the_root_commit() {
        let dir = setup_test_repo();
        let main_rs = dir.path().join("src/main.rs");
        let content = fs::read_to_string(&main_rs).unwrap();
        fs::write(
            &main_rs,
            content.replace("println!(\"hello\")", "println!(\"second\")"),
        )
        .unwrap();
        git(dir.path(), &["add", "-A"]);
        git(dir.path(), &["commit", "-m", "second commit"]);

        // `HEAD` reaches the root commit, whose `^` does not resolve. Making a
        // failed `git diff` fatal turned that into an aborted log for every
        // full-history request — `test_log_mode` only ever asked for
        // `HEAD~1..HEAD`, which stops one commit short of the problem.
        let result = run_diff_in(
            dir.path(),
            &DiffSource::Log("HEAD".to_string()),
            None,
            None,
            false,
            None,
        )
        .expect("a log spanning the root commit must not fail");

        assert!(
            result.contains("second commit") && result.contains("initial"),
            "both commits must appear:\n{result}"
        );
        // The root commit's files are added against the empty tree, which is
        // how `git log --patch` presents them.
        //
        // Assert on `goodbye`, not `hello`: the second commit touches `hello`,
        // so that name appears either way and the assertion would pass with the
        // root commit rendered empty. `goodbye` and `main` exist only in the
        // root commit's add.
        assert!(
            result.contains("goodbye"),
            "root commit must report its symbols as added:\n{result}"
        );
    }

    // 27. test_bad_ref_is_an_error_not_no_changes
    #[test]
    fn test_bad_ref_is_an_error_not_no_changes() {
        let dir = setup_test_repo();

        // `git diff` exits 128 here and writes nothing to stdout. Reporting
        // that as "No changes." with a zero exit is the #111 danger in its
        // purest form: a failure that reads as evidence of a clean tree.
        let result = run_diff_in(
            dir.path(),
            &DiffSource::GitRef("HEAD..no-such-ref-xyz".to_string()),
            None,
            None,
            false,
            None,
        );
        let err = result.expect_err("a nonexistent ref must not succeed");
        assert!(
            err.contains("git diff failed"),
            "error must say git failed, got {err:?}"
        );
        assert!(
            !err.contains("No changes"),
            "a bad ref must never be reported as no changes, got {err:?}"
        );
    }

    /// A repository with no commit yet, so `HEAD` does not resolve.
    fn setup_unborn_repo() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("failed to create tempdir");
        let p = dir.path();
        git(p, &["init"]);
        git(p, &["config", "user.email", "test@test.com"]);
        git(p, &["config", "user.name", "Test"]);
        fs::create_dir_all(p.join("src")).unwrap();
        fs::write(
            p.join("src/main.rs"),
            "fn hello() {\n    println!(\"hi\");\n}\n\nfn main() {\n    hello();\n}\n",
        )
        .unwrap();
        dir
    }

    // 28. test_unborn_head_reports_staged_files
    #[test]
    fn test_unborn_head_reports_staged_files() {
        let dir = setup_unborn_repo();
        git(dir.path(), &["add", "-A"]);

        // `git diff HEAD` cannot work before the first commit. That used to be
        // reported as "No changes." — a lie, there are staged files — and #112
        // turned it into a raw git fatal. The empty tree is the honest base.
        let result = run_diff_in(
            dir.path(),
            &DiffSource::GitUncommitted,
            None,
            None,
            false,
            None,
        )
        .expect("a repo with no commit yet must still diff");

        // Assert on the symbol lines, not bare names: the fixture file is
        // `src/main.rs`, so `contains("main")` is satisfied by the `## src/main.rs`
        // header alone — it would hold even if no symbol were extracted, which
        // is exactly what an abandoned overlay looks like.
        assert!(
            result.contains("[+]      hello") && result.contains("[+]      main"),
            "staged files must be reported as added symbols:\n{result}"
        );
        assert!(
            !result.contains("analysis unavailable"),
            "the unborn path must not leave an overlay abandoned:\n{result}"
        );
        assert!(
            !result.contains("No changes"),
            "there are staged files, so this is not 'no changes':\n{result}"
        );
    }

    // 29. test_unborn_head_with_nothing_tracked
    #[test]
    fn test_unborn_head_with_nothing_tracked() {
        // Nothing added to the index: `git init` then `tilth diff`, which is
        // plausible first contact with the tool. Untracked files are not part
        // of a diff, so "No changes." is the correct answer — but it has to be
        // reached deliberately, not by a git fatal.
        let dir = setup_unborn_repo();
        let result = run_diff_in(
            dir.path(),
            &DiffSource::GitUncommitted,
            None,
            None,
            false,
            None,
        )
        .expect("a freshly initialized repo must not error");
        assert_eq!(result, "No changes.");
    }

    // 30. test_unborn_head_staged_mode_unaffected
    #[test]
    fn test_unborn_head_staged_mode_unaffected() {
        // `git diff --staged` already works before the first commit. Locking it
        // so the retry added for `GitUncommitted` cannot regress this path.
        let dir = setup_unborn_repo();
        git(dir.path(), &["add", "-A"]);

        let result =
            run_diff_in(dir.path(), &DiffSource::GitStaged, None, None, false, None).unwrap();
        assert!(
            result.contains("hello"),
            "staged mode must still report the staged file:\n{result}"
        );
    }

    // 31. test_corrupt_head_ref_still_errors
    #[test]
    fn test_corrupt_head_ref_still_errors() {
        let dir = setup_test_repo();
        let p = dir.path();

        // Point the checked-out branch at garbage. The repository still HAS
        // history, but HEAD cannot be resolved — and `git rev-parse --verify
        // HEAD` fails exactly as it does for an unborn HEAD, with git emitting
        // the same fatal for both. Falling back to the empty tree here would
        // report the whole working tree as newly added: a damaged repo
        // rendered as a brand new one, which is the silent-success shape #112
        // exists to prevent.
        let head_ref = String::from_utf8_lossy(
            &Command::new("git")
                .args(["symbolic-ref", "HEAD"])
                .current_dir(p)
                .output()
                .expect("git")
                .stdout,
        )
        .trim()
        .to_string();
        fs::write(p.join(".git").join(&head_ref), "garbagegarbage\n").unwrap();

        let result = run_diff_in(p, &DiffSource::GitUncommitted, None, None, false, None);
        let err = result.expect_err("a repo with an unreadable HEAD must not report a fresh tree");
        assert!(
            err.contains("git diff failed"),
            "expected git's error, got {err:?}"
        );
    }

    // 32. test_unborn_head_reports_empty_files
    #[test]
    fn test_unborn_head_reports_empty_files() {
        let dir = setup_unborn_repo();
        // A zero-byte file — `.gitkeep`, `__init__.py`, `py.typed`. git emits
        // no `--- /dev/null` for these, only `new file mode`, so they parsed as
        // Modified and the overlay went looking for an old side that cannot
        // exist before the first commit.
        fs::write(dir.path().join("src/empty.rs"), "").unwrap();
        git(dir.path(), &["add", "-A"]);

        let result = run_diff_in(
            dir.path(),
            &DiffSource::GitUncommitted,
            None,
            None,
            false,
            None,
        )
        .expect("an empty staged file must not break the diff");

        assert!(
            !result.contains("analysis unavailable"),
            "a zero-byte addition has no old side to fail on:\n{result}"
        );
        assert!(
            !result.contains("could not analyze"),
            "and must not raise a warning about one:\n{result}"
        );
        assert!(
            result.contains("[+]      hello"),
            "the real file must still be reported:\n{result}"
        );
    }

    // 33. test_deleting_an_empty_file
    #[test]
    fn test_deleting_an_empty_file() {
        // The mirror of the addition case: git omits `+++ /dev/null` when the
        // file being deleted had no content, so only `deleted file mode` marks
        // it. Parsed as Modified, the overlay tried to read a new side that is
        // gone from disk and was abandoned.
        let dir = setup_test_repo();
        let p = dir.path();
        fs::write(p.join("src/empty.rs"), "").unwrap();
        git(p, &["add", "-A"]);
        git(p, &["commit", "-m", "add empty"]);
        fs::remove_file(p.join("src/empty.rs")).unwrap();

        let result = run_diff_in(p, &DiffSource::GitUncommitted, None, None, false, None)
            .expect("deleting an empty file must not break the diff");
        assert!(
            !result.contains("analysis unavailable") && !result.contains("could not analyze"),
            "an empty deletion has no new side to fail on:\n{result}"
        );
    }

    // 34. test_unborn_head_in_a_sha256_repo
    #[test]
    fn test_unborn_head_in_a_sha256_repo() {
        // The well-known `4b825dc…` empty tree is SHA-1 only. A repo created
        // with `--object-format=sha256` has a different one, and passing the
        // SHA-1 id there fails with "unknown revision" — so a hardcoded
        // constant made both the unborn-HEAD and parentless-commit paths
        // silently stop working on such a repo.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        git(p, &["init", "--object-format=sha256"]);

        // Old git builds have no SHA-256 support; skip rather than fail there.
        let fmt = {
            let out = Command::new("git")
                .args(["rev-parse", "--show-object-format"])
                .current_dir(p)
                .output()
                .expect("git");
            String::from_utf8_lossy(&out.stdout).trim().to_string()
        };
        if fmt != "sha256" {
            eprintln!("skipping: git here does not support sha256 (got {fmt:?})");
            return;
        }

        git(p, &["config", "user.email", "test@test.com"]);
        git(p, &["config", "user.name", "Test"]);
        fs::create_dir_all(p.join("src")).unwrap();
        fs::write(p.join("src/main.rs"), "fn sha_only() {}\n").unwrap();
        git(p, &["add", "-A"]);

        let result = run_diff_in(p, &DiffSource::GitUncommitted, None, None, false, None)
            .expect("a sha256 repo with no commit must still diff");
        assert!(
            result.contains("sha_only"),
            "staged file must be reported in a sha256 repo:\n{result}"
        );
    }

    // 35. test_symmetric_ref_range_uses_merge_base
    #[test]
    fn test_symmetric_ref_range_uses_merge_base() {
        let dir = setup_diverged_repo();

        // `base...feat` is merge-base(base, feat) vs feat. `feat_only` is the
        // only symbol added on that path; `base_only` never exists on either
        // side of it. Before #111 the ref split at the first `..`, leaving a
        // new-side rev of `.feat` that `git show` rejects — so every file's
        // overlay was abandoned and this rendered as an authoritative zero.
        let symmetric = run_diff_in(
            dir.path(),
            &DiffSource::GitRef("base...feat".to_string()),
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert!(
            symmetric.contains("feat_only"),
            "`base...feat` must report the symbol added on feat:\n{symmetric}"
        );
        assert!(
            !symmetric.contains("base_only"),
            "`base...feat` must not report base-side symbols — merge-base, not base:\n{symmetric}"
        );
        assert!(
            !symmetric.contains("analysis unavailable"),
            "`base...feat` must resolve both sides:\n{symmetric}"
        );

        // The two-dot spelling asks a different question and must keep its own
        // answer: relative to `base`, `base_only` is gone.
        let two_dot = run_diff_in(
            dir.path(),
            &DiffSource::GitRef("base..feat".to_string()),
            None,
            None,
            false,
            None,
        )
        .unwrap();
        assert!(
            two_dot.contains("base_only"),
            "`base..feat` must still diff against the base tip:\n{two_dot}"
        );
    }

    // 36. test_symmetric_ref_range_file_scope
    #[test]
    fn test_symmetric_ref_range_file_scope() {
        let dir = setup_diverged_repo();

        // The reported symptom, verbatim: a scoped symmetric diff answering
        // "0 symbols touched, +0/−0 lines" for a file that really changed.
        let result = run_diff_in(
            dir.path(),
            &DiffSource::GitRef("base...feat".to_string()),
            Some("src/main.rs"),
            None,
            false,
            None,
        )
        .unwrap();
        assert!(
            !result.contains("0 symbols touched"),
            "a real change must not report zero symbols:\n{result}"
        );
        assert!(
            result.contains("feat_only"),
            "scoped symmetric diff must name the added symbol:\n{result}"
        );
    }
}
// test
