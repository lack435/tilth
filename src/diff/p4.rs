//! Perforce backend for `tilth_diff` — sources the "uncommitted" diff (opened
//! files) from a Perforce workspace instead of git, then feeds the same
//! structural overlay/matcher/formatters that the git path uses.
//!
//! # Why this exists
//!
//! `tilth_diff`'s value over a raw textual `p4 diff` is entirely downstream of
//! two inputs: the unified-diff hunks and the old/new content of each file. Git
//! is only the *front-end* that supplies those. Perforce can supply them too —
//! `p4 diff -du` already emits unified diff, `p4 print` yields the depot
//! revision, and the working file on disk is the new side — so the overlay,
//! three-phase symbol matcher, blast radius, and progressive-disclosure
//! formatters are reused unchanged.
//!
//! # Safety patterns (mirrored from the workspace's `mcp-perforce` server)
//!
//! Perforce environment resolution is *not* safe to trust blindly:
//!
//!  * **Client derivation.** A tree may set no `P4CLIENT`, so p4 defaults it to
//!    the machine hostname and every op fails with "Client unknown". We instead
//!    find the client whose `Host` matches this machine and whose `Root` is the
//!    longest prefix of the target directory — never trusting the ambient
//!    `P4CLIENT`.
//!  * **Charset.** Unicode-mode servers corrupt content ops under a shell's
//!    `auto` charset, so every call passes `-C utf8`.
//!  * **stdin.** p4 is spawned with stdin from null: a p4 that prompts (expired
//!    ticket, trust dialog) can never steal our stdio or hang. The per-request
//!    wall-clock timeout (`crate::timeout`) is the second backstop.
//!
//! Every command here is read-only (`info`, `clients`, `fstat`, `diff`,
//! `print`). This module never opens, reverts, or submits anything.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use super::parse::parse_unified_diff;
use super::{DiffSource, FileDiff, FileStatus};

/// How a file's old (before) side is sourced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OldSide {
    /// No prior revision — a fresh `add`/`branch`. Empty old content is correct.
    None,
    /// Fetch `depot#rev` for the old side.
    Rev(u32),
    /// The file has an old side but its have-revision was absent/unparseable.
    /// Fetching must *fail* rather than return empty — an empty old side makes a
    /// modified file read as entirely added, the confidently-wrong shape #111
    /// exists to prevent.
    Missing,
}

/// Per-file Perforce metadata, carried on the `DiffSource` so the overlay's
/// content-fetch can resolve each side without re-enumerating.
#[derive(Debug, Clone)]
pub struct P4Entry {
    /// Depot path (`//depot/...`) of the file's OLD side, for `p4 print`. Equal
    /// to the working file's depot for an edit; the move-from path for a rename.
    pub depot: String,
    /// Local filesystem path of the working (new-side) file, read from disk.
    pub local: PathBuf,
    /// How to source the old side.
    pub old: OldSide,
}

/// Run one read-only `p4` command. `-C utf8` always; `-c <client>` when known;
/// stdin from null so a prompting p4 can never hang us. Returns
/// `(exit_code, stdout_bytes, stderr_text)`.
fn run_p4(
    args: &[&str],
    client: Option<&str>,
    cwd: Option<&Path>,
) -> Result<(i32, Vec<u8>, String), String> {
    let mut cmd = Command::new("p4");
    cmd.args(["-C", "utf8"]);
    if let Some(c) = client {
        cmd.args(["-c", c]);
    }
    cmd.args(args);
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    cmd.stdin(Stdio::null());
    let output = cmd
        .output()
        .map_err(|e| format!("p4 not found on PATH (or failed to launch): {e}"))?;
    let code = output.status.code().unwrap_or(-1);
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    Ok((code, output.stdout, stderr))
}

/// Parse `p4 -ztag` output: `... key value` lines, blank lines separate records.
fn parse_ztag(text: &str) -> Vec<HashMap<String, String>> {
    let mut records = Vec::new();
    let mut cur: HashMap<String, String> = HashMap::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            if !cur.is_empty() {
                records.push(std::mem::take(&mut cur));
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("... ") {
            let mut parts = rest.splitn(2, ' ');
            if let Some(key) = parts.next() {
                cur.insert(key.to_string(), parts.next().unwrap_or("").to_string());
            }
        }
    }
    if !cur.is_empty() {
        records.push(cur);
    }
    records
}

/// Normalize a path for prefix comparison: forward slashes, no trailing slash,
/// lowercased. Perforce/Windows paths are case-insensitive, so a case-only
/// mismatch between the client `Root` and the scope must not defeat the match.
fn norm(path: &str) -> String {
    path.trim()
        .replace('\\', "/")
        .trim_end_matches('/')
        .to_lowercase()
}

/// Resolve the Perforce client for `dir`, mirroring `mcp-perforce`'s algorithm.
///
/// The client whose `Host` matches this machine and whose `Root` is a parent of
/// `dir` (longest root wins, so a nested workspace beats an outer one). A tie
/// (two clients sharing a root) or no match is an error naming the reason —
/// which doubles as the "this is not a resolvable Perforce workspace" signal
/// that `diff()` uses to fall through to a clean message.
pub fn resolve_client(dir: &Path) -> Result<String, String> {
    let key = norm(&dir.to_string_lossy());

    let (_, info_out, info_err) = run_p4(&["-ztag", "info"], None, Some(dir))?;
    let info = parse_ztag(&String::from_utf8_lossy(&info_out));
    let info = info.first().ok_or_else(|| {
        let e = info_err.trim();
        if e.is_empty() {
            "p4 info returned nothing".to_string()
        } else {
            format!("p4 info: {e}")
        }
    })?;
    let host = info.get("clientHost").cloned().unwrap_or_default();
    let user = info
        .get("userName")
        .filter(|u| !u.is_empty())
        .ok_or("could not determine P4USER from p4 info")?;

    let (_, clients_out, clients_err) = run_p4(&["-ztag", "clients", "-u", user], None, Some(dir))?;
    let clients = parse_ztag(&String::from_utf8_lossy(&clients_out));

    // (root_len, client) for every client on this host whose root is a prefix of `dir`.
    let mut matches: Vec<(usize, String)> = Vec::new();
    for rec in &clients {
        let chost = rec.get("Host").map_or("", String::as_str);
        // An empty Host field means the client is usable from any host.
        if !chost.is_empty() && !chost.eq_ignore_ascii_case(&host) {
            continue;
        }
        let nroot = norm(rec.get("Root").map_or("", String::as_str));
        if nroot.is_empty() {
            continue;
        }
        if key == nroot || key.starts_with(&format!("{nroot}/")) {
            if let Some(name) = rec.get("client") {
                matches.push((nroot.len(), name.clone()));
            }
        }
    }

    if matches.is_empty() {
        let e = clients_err.trim();
        let hint = if e.is_empty() { "" } else { " " };
        return Err(format!(
            "no Perforce client for {key} on host '{}' (user {user}){hint}{e}",
            if host.is_empty() { "?" } else { &host }
        ));
    }
    let best = matches.iter().map(|m| m.0).max().unwrap();
    let tied: Vec<&String> = matches
        .iter()
        .filter(|m| m.0 == best)
        .map(|m| &m.1)
        .collect();
    if tied.len() > 1 {
        let mut names: Vec<String> = tied.iter().map(|s| (*s).clone()).collect();
        names.sort();
        return Err(format!(
            "ambiguous Perforce workspace: clients share a root ({})",
            names.join(", ")
        ));
    }
    Ok(tied[0].clone())
}

/// Make a local filesystem path relative to the working dir, forward-slashed,
/// preserving the original case of the tail. Falls back to the whole path when
/// it is not under `work_dir` (should not happen for opened files under scope).
///
/// Walks path components rather than byte-slicing off a lowercased prefix
/// length: `char::to_lowercase` can change a string's byte length (e.g. `İ`
/// U+0130), so slicing the original-case string at the lowercased prefix's byte
/// length can land mid-codepoint (panic) or shifted (wrong path). `work_dir_norm`
/// is already lowercased by [`norm`]; each component is compared case-folded.
fn make_rel(local: &str, work_dir_norm: &str) -> String {
    let fwd = local.replace('\\', "/");
    let mut tail = fwd.split('/');
    for want in work_dir_norm.split('/') {
        match tail.next() {
            Some(got) if got.to_lowercase() == want => {}
            // Not under work_dir — return the whole (forward-slashed) path.
            _ => return fwd,
        }
    }
    tail.collect::<Vec<_>>().join("/")
}

/// Re-base an absolute scope path onto the work-dir-relative paths the diff
/// enumeration produces, for downstream selection. Returns `None` when the
/// scope *is* the working directory (whole-tree — selection treats that as
/// "everything", the same as no scope).
#[must_use]
pub fn relativize(scope_path: &str, work_dir: &Path) -> Option<String> {
    let wd = norm(&work_dir.to_string_lossy());
    if norm(scope_path) == wd {
        return None;
    }
    let rel = make_rel(scope_path, &wd);
    let trimmed = rel.trim_end_matches('/');
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Map a Perforce `action` to a `FileStatus`.
///
/// `move/add` and `move/delete` are deliberately reported as plain Added and
/// Deleted rather than a Renamed pair: the overlay's `cross_file_matching`
/// already collapses a name-matched delete+add across files into a `Moved`
/// change, so re-deriving the rename here would duplicate that work.
fn action_to_status(action: &str) -> Option<FileStatus> {
    match action {
        "edit" | "integrate" => Some(FileStatus::Modified),
        "add" | "branch" | "move/add" | "import" => Some(FileStatus::Added),
        "delete" | "move/delete" | "purge" => Some(FileStatus::Deleted),
        _ => None,
    }
}

/// Normalize `p4 diff -du` output into git-style unified diff so the existing,
/// well-tested `parse_unified_diff` consumes it unchanged.
///
/// p4 emits `--- //depot/path\t<ts>` / `+++ <local>\t<ts>` headers with no
/// `diff --git` line and no `a/`/`b/` prefixes. For each header pair we look the
/// local path up in `local_to_rel` (built from `fstat`, so the rel is identical
/// to the one the file's `FileDiff` carries) and synthesize the three lines the
/// parser keys on. Hunk and content lines pass through verbatim.
///
/// A `---`/`+++` pair is only a header when the two lines are *adjacent* — that
/// is the shape unified diff guarantees, and it is what distinguishes a header
/// from a hunk body line that happens to start with `--- `/`+++ ` (a removed
/// `-- x` line becomes `--- x`; an added `++ x` becomes `+++ x`). Lines for a
/// file whose local is not in the map (outside the enumerated scope) are dropped
/// entirely — header *and* body — so an orphan hunk can never attach to the
/// previous file (which `parse_unified_diff` would do, since it only starts a
/// new file on `diff --git`).
fn normalize_p4_diff(raw: &str, local_to_rel: &HashMap<String, String>) -> String {
    let mut out = String::with_capacity(raw.len() + raw.len() / 8);
    let lines: Vec<&str> = raw.lines().collect();
    // Whether the file currently being streamed is one we mapped (emit its body)
    // or an unknown/out-of-scope file (drop its body until the next header pair).
    let mut emitting = false;
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if line.starts_with("--- ") && lines.get(i + 1).is_some_and(|n| n.starts_with("+++ ")) {
            let plus = lines[i + 1].strip_prefix("+++ ").unwrap_or(lines[i + 1]);
            let local = plus.split('\t').next().unwrap_or(plus);
            let lower = local.replace('\\', "/").to_lowercase();
            if let Some(rel) = local_to_rel.get(&lower) {
                out.push_str("diff --git a/");
                out.push_str(rel);
                out.push_str(" b/");
                out.push_str(rel);
                out.push_str("\n--- a/");
                out.push_str(rel);
                out.push_str("\n+++ b/");
                out.push_str(rel);
                out.push('\n');
                emitting = true;
            } else {
                emitting = false;
            }
            i += 2;
            continue;
        }
        if emitting {
            out.push_str(line);
            out.push('\n');
        }
        i += 1;
    }
    out
}

/// Build the uncommitted (opened-files) diff for a Perforce workspace.
///
/// `work_dir` is the directory the diff is scoped to (already a directory —
/// a file scope passes its parent and `file_filter` names the file). Returns
/// the `FileDiff` list and a `DiffSource::P4Uncommitted` carrying the per-file
/// depot/local/have-rev map the overlay needs, or `Ok(None)` when nothing is
/// open under the scope (the caller renders "No changes.").
pub fn build_uncommitted(
    work_dir: &Path,
    file_filter: Option<&Path>,
    client: &str,
) -> Result<Option<(Vec<FileDiff>, DiffSource)>, String> {
    let work_dir_norm = norm(&work_dir.to_string_lossy());

    // p4 file argument: the scoped file, or everything under the working dir.
    // An *explicit absolute* path (`<dir>/...`) rather than a bare `...`: p4's
    // `...` resolves against its own process cwd, and a child cwd set via the
    // OS does not reliably steer that the way a shell `cd` does — so the bare
    // form leaked the whole workspace. The absolute filter is cwd-independent.
    let arg: String = match file_filter {
        Some(f) => f.to_string_lossy().into_owned(),
        None => work_dir.join("...").to_string_lossy().into_owned(),
    };

    // 1. Enumerate opened files: one call, restricted to the fields we use.
    let fields = "depotFile,clientFile,action,haveRev,movedFile,type,headType";
    let (_, fstat_out, fstat_err) = run_p4(
        &["-ztag", "fstat", "-Ro", "-T", fields, &arg],
        Some(client),
        Some(work_dir),
    )?;
    let records = parse_ztag(&String::from_utf8_lossy(&fstat_out));
    if records.is_empty() {
        // "not opened" / "no such file(s)" is the empty case, not an error.
        let e = fstat_err.trim();
        if e.is_empty() || e.contains("not opened") || e.contains("no such file") {
            return Ok(None);
        }
        return Err(format!("p4 fstat failed: {e}"));
    }

    // 2. Hunks for edited files: normalize `p4 diff -du` and parse once.
    let mut local_to_rel: HashMap<String, String> = HashMap::new();
    for rec in &records {
        if let Some(local) = rec.get("clientFile") {
            let rel = make_rel(local, &work_dir_norm);
            local_to_rel.insert(local.replace('\\', "/").to_lowercase(), rel);
        }
    }
    let (_, diff_out, _) = run_p4(&["diff", "-du", &arg], Some(client), Some(work_dir))?;
    let normalized = normalize_p4_diff(&String::from_utf8_lossy(&diff_out), &local_to_rel);
    let mut hunks_by_rel: HashMap<String, Vec<super::Hunk>> = HashMap::new();
    for fd in parse_unified_diff(&normalized) {
        hunks_by_rel.insert(fd.path.to_string_lossy().into_owned(), fd.hunks);
    }

    // 3. Index the move-from (delete) side of every rename by depot path, so a
    //    `move/add` can pair with it into a single Renamed FileDiff. Without this
    //    a move that also edited the file would surface as a pure move and drop
    //    the body/signature change (the new and old sides never get diffed).
    let move_dels: HashMap<&str, &HashMap<String, String>> = records
        .iter()
        .filter(|r| r.get("action").map(String::as_str) == Some("move/delete"))
        .filter_map(|r| r.get("depotFile").map(|d| (d.as_str(), r)))
        .collect();
    let mut consumed_dels: std::collections::HashSet<&str> = std::collections::HashSet::new();

    let mut file_diffs: Vec<FileDiff> = Vec::new();
    let mut entries: HashMap<PathBuf, P4Entry> = HashMap::new();

    // Pass 1 — every record except `move/delete` (handled in pass 2, so a delete
    // consumed by its paired add is never emitted twice).
    for rec in &records {
        let (Some(depot), Some(local), Some(action)) = (
            rec.get("depotFile"),
            rec.get("clientFile"),
            rec.get("action"),
        ) else {
            continue;
        };
        if action == "move/delete" {
            continue;
        }
        let have_rev = rec.get("haveRev").and_then(|r| r.parse::<u32>().ok());

        // A `move/add` paired with a known move-from becomes one Renamed diff.
        if action == "move/add" {
            if let Some(del) = rec
                .get("movedFile")
                .and_then(|mf| move_dels.get(mf.as_str()))
            {
                let moved_from = rec.get("movedFile").unwrap().as_str();
                let old_depot = del.get("depotFile").map_or(moved_from, String::as_str);
                let old_local = del.get("clientFile").map_or(old_depot, String::as_str);
                let old_have = del.get("haveRev").and_then(|r| r.parse::<u32>().ok());
                push_entry(
                    &mut file_diffs,
                    &mut entries,
                    &mut hunks_by_rel,
                    &work_dir_norm,
                    local,
                    Some(old_local),
                    old_depot,
                    FileStatus::Renamed,
                    classify_old(FileStatus::Renamed, old_have),
                    is_binary(rec),
                );
                consumed_dels.insert(moved_from);
                continue;
            }
        }

        let Some(status) = action_to_status(action) else {
            continue;
        };
        push_entry(
            &mut file_diffs,
            &mut entries,
            &mut hunks_by_rel,
            &work_dir_norm,
            local,
            None,
            depot,
            status,
            classify_old(status, have_rev),
            is_binary(rec),
        );
    }

    // Pass 2 — move/delete records not consumed by a paired add (the add is out
    // of scope, or p4 gave no `movedFile`) fall back to a plain Deleted.
    for rec in &records {
        if rec.get("action").map(String::as_str) != Some("move/delete") {
            continue;
        }
        let (Some(depot), Some(local)) = (rec.get("depotFile"), rec.get("clientFile")) else {
            continue;
        };
        if consumed_dels.contains(depot.as_str()) {
            continue;
        }
        let have_rev = rec.get("haveRev").and_then(|r| r.parse::<u32>().ok());
        push_entry(
            &mut file_diffs,
            &mut entries,
            &mut hunks_by_rel,
            &work_dir_norm,
            local,
            None,
            depot,
            FileStatus::Deleted,
            classify_old(FileStatus::Deleted, have_rev),
            is_binary(rec),
        );
    }

    if file_diffs.is_empty() {
        return Ok(None);
    }
    Ok(Some((
        file_diffs,
        DiffSource::P4Uncommitted {
            client: client.to_string(),
            work_dir: work_dir.to_path_buf(),
            entries,
        },
    )))
}

/// Classify a file's old side from its status and have-revision.
fn classify_old(status: FileStatus, have: Option<u32>) -> OldSide {
    match status {
        FileStatus::Added => OldSide::None,
        _ => match have {
            Some(r) => OldSide::Rev(r),
            None => OldSide::Missing,
        },
    }
}

/// Does this fstat record describe a binary file? p4 marks it via the open
/// `type` (`binary`, `binary+l`), or `headType` for an already-tracked file.
fn is_binary(rec: &HashMap<String, String>) -> bool {
    rec.get("type")
        .or_else(|| rec.get("headType"))
        .is_some_and(|t| t.starts_with("binary"))
}

/// Push one file's `FileDiff` + `P4Entry`. `local` is the new-side working file;
/// `depot` is the OLD-side depot path (the move-from for a rename). Hunks are
/// claimed from the parsed `p4 diff` only for a modified file.
#[allow(clippy::too_many_arguments)]
fn push_entry(
    file_diffs: &mut Vec<FileDiff>,
    entries: &mut HashMap<PathBuf, P4Entry>,
    hunks_by_rel: &mut HashMap<String, Vec<super::Hunk>>,
    work_dir_norm: &str,
    local: &str,
    old_local: Option<&str>,
    depot: &str,
    status: FileStatus,
    old: OldSide,
    is_binary: bool,
) {
    let rel = make_rel(local, work_dir_norm);
    let is_generated = Path::new(&rel)
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(crate::lang::detection::is_generated_by_name);
    let hunks = if status == FileStatus::Modified {
        hunks_by_rel.remove(&rel).unwrap_or_default()
    } else {
        Vec::new()
    };
    let rel_path = PathBuf::from(&rel);
    entries.insert(
        rel_path.clone(),
        P4Entry {
            depot: depot.to_string(),
            local: PathBuf::from(local),
            old,
        },
    );
    file_diffs.push(FileDiff {
        path: rel_path,
        old_path: old_local.map(|l| PathBuf::from(make_rel(l, work_dir_norm))),
        status,
        hunks,
        is_generated,
        is_binary,
    });
}

/// Fetch a file's old-side content from Perforce: `p4 print -q <depot>#<rev>`.
///
/// Runs in `work_dir` so p4's `P4CONFIG` discovery (which walks up from the
/// process cwd) resolves the same server/port/client the enumeration used — a
/// bare cwd would break every per-workspace `.p4config` setup.
pub fn print_old(entry: &P4Entry, client: &str, work_dir: &Path) -> Result<String, String> {
    let rev = match entry.old {
        OldSide::None => return Ok(String::new()),
        OldSide::Missing => {
            return Err(format!(
                "no Perforce have-revision for {} — cannot fetch its old side",
                entry.depot
            ));
        }
        OldSide::Rev(r) => r,
    };
    let spec = format!("{}#{rev}", entry.depot);
    let (code, out, err) = run_p4(&["print", "-q", &spec], Some(client), Some(work_dir))?;
    if code != 0 {
        return Err(format!("p4 print {spec}: {}", err.trim()));
    }
    Ok(String::from_utf8_lossy(&out).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ztag_splits_records_and_keys() {
        let text = "... depotFile //d/a.cpp\n... action edit\n... haveRev 45\n\n\
                    ... depotFile //d/b.cpp\n... action add\n";
        let recs = parse_ztag(text);
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0]["depotFile"], "//d/a.cpp");
        assert_eq!(recs[0]["action"], "edit");
        assert_eq!(recs[0]["haveRev"], "45");
        assert_eq!(recs[1]["action"], "add");
        assert!(!recs[1].contains_key("haveRev"));
    }

    #[test]
    fn parse_ztag_value_may_contain_spaces() {
        let recs = parse_ztag("... desc a b c\n");
        assert_eq!(recs[0]["desc"], "a b c");
    }

    #[test]
    fn action_mapping_covers_the_open_actions() {
        assert_eq!(action_to_status("edit"), Some(FileStatus::Modified));
        assert_eq!(action_to_status("integrate"), Some(FileStatus::Modified));
        assert_eq!(action_to_status("add"), Some(FileStatus::Added));
        assert_eq!(action_to_status("branch"), Some(FileStatus::Added));
        assert_eq!(action_to_status("move/add"), Some(FileStatus::Added));
        assert_eq!(action_to_status("delete"), Some(FileStatus::Deleted));
        assert_eq!(action_to_status("move/delete"), Some(FileStatus::Deleted));
        assert_eq!(action_to_status("unknown-verb"), None);
    }

    #[test]
    fn make_rel_strips_work_dir_case_insensitively() {
        let wd = norm("C:/dev/main/UE");
        // p4 hands back backslashes and possibly different case on the drive.
        assert_eq!(
            make_rel(r"C:\dev\main\UE\Bobcat\Source\Foo.cpp", &wd),
            "Bobcat/Source/Foo.cpp"
        );
        assert_eq!(
            make_rel(r"c:\DEV\Main\ue\Bar.h", &wd),
            "Bar.h",
            "a case-only difference must still strip the prefix"
        );
    }

    #[test]
    fn make_rel_keeps_outside_paths_whole() {
        let wd = norm("C:/dev/main/UE");
        assert_eq!(make_rel(r"D:\other\x.cpp", &wd), "D:/other/x.cpp");
    }

    #[test]
    fn normalize_produces_parseable_git_headers() {
        // A realistic two-line p4 header with tab-separated timestamps, then a hunk.
        let raw = "--- //bobcat/main/UE/Bobcat/Foo.cpp\t2026-08-25 22:29:19.000000000 0000\n\
                   +++ C:\\dev\\main\\UE\\Bobcat\\Foo.cpp\t2026-08-25 22:29:19.000000000 0000\n\
                   @@ -1,2 +1,3 @@\n \
                   ctx\n-old\n+new\n+extra\n";
        let mut map = HashMap::new();
        map.insert(
            "c:/dev/main/ue/bobcat/foo.cpp".to_string(),
            "Bobcat/Foo.cpp".to_string(),
        );
        let norm = normalize_p4_diff(raw, &map);
        assert!(
            norm.contains("diff --git a/Bobcat/Foo.cpp b/Bobcat/Foo.cpp"),
            "normalized:\n{norm}"
        );
        let diffs = parse_unified_diff(&norm);
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].path, PathBuf::from("Bobcat/Foo.cpp"));
        assert_eq!(diffs[0].status, FileStatus::Modified);
        assert_eq!(diffs[0].hunks.len(), 1);
        assert_eq!(diffs[0].hunks[0].new_count, 3);
    }

    #[test]
    fn normalize_keeps_body_lines_that_look_like_headers() {
        // A removed source line `-- x` becomes the diff line `--- x`, and an
        // added `++ x` becomes `+++ x`. Neither is a header (its neighbour is not
        // the matching half), so both must survive as hunk content.
        let raw = "--- //d/UE/a.cpp\t2026\n+++ C:\\dev\\main\\UE\\a.cpp\t2026\n\
                   @@ -1,2 +1,2 @@\n-- old comment\n++ new comment\n";
        let mut map = HashMap::new();
        map.insert("c:/dev/main/ue/a.cpp".to_string(), "a.cpp".to_string());
        let norm = normalize_p4_diff(raw, &map);
        let diffs = parse_unified_diff(&norm);
        assert_eq!(diffs.len(), 1, "normalized:\n{norm}");
        let lines = &diffs[0].hunks[0].lines;
        assert_eq!(lines.len(), 2, "both body lines must be kept: {lines:?}");
        assert_eq!(lines[0].content, "- old comment");
        assert_eq!(lines[1].content, "+ new comment");
    }

    #[test]
    fn normalize_does_not_leak_unknown_hunks_into_previous_file() {
        // Known file A, then unknown file B. B's hunk must be dropped, not
        // appended to A (which `parse_unified_diff` would do without a boundary).
        let raw = "--- //d/UE/a.cpp\t1\n+++ C:\\dev\\main\\UE\\a.cpp\t1\n\
                   @@ -1 +1 @@\n-a\n+A\n\
                   --- //d/UE/b.cpp\t1\n+++ C:\\dev\\main\\UE\\b.cpp\t1\n\
                   @@ -9 +9 @@\n-b\n+B\n";
        let mut map = HashMap::new();
        map.insert("c:/dev/main/ue/a.cpp".to_string(), "a.cpp".to_string());
        // b.cpp deliberately absent from the map.
        let norm = normalize_p4_diff(raw, &map);
        let diffs = parse_unified_diff(&norm);
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].path, PathBuf::from("a.cpp"));
        assert_eq!(diffs[0].hunks.len(), 1, "only A's hunk, not B's");
        assert_eq!(diffs[0].hunks[0].old_start, 1);
    }

    #[test]
    fn classify_old_side_by_status() {
        assert_eq!(classify_old(FileStatus::Added, None), OldSide::None);
        assert_eq!(classify_old(FileStatus::Added, Some(3)), OldSide::None);
        assert_eq!(classify_old(FileStatus::Modified, Some(7)), OldSide::Rev(7));
        // A modified/deleted file with no have-rev must be Missing (fetch fails)
        // rather than silently reading as an all-added file.
        assert_eq!(classify_old(FileStatus::Modified, None), OldSide::Missing);
        assert_eq!(classify_old(FileStatus::Deleted, None), OldSide::Missing);
    }

    #[test]
    fn make_rel_handles_non_ascii_without_panicking() {
        // `İ` (U+0130) lowercases to a longer byte sequence, so a byte-length
        // slice off the lowercased prefix would panic or mis-slice. Component
        // walking must strip it cleanly.
        let wd = norm(r"C:\dev\İ\proj");
        let got = make_rel(r"C:\dev\İ\proj\src\Foo.cpp", &wd);
        assert_eq!(got, "src/Foo.cpp");
    }

    #[test]
    fn normalize_drops_sections_for_unknown_locals() {
        // A file not in the enumerated map (e.g. outside scope) must not produce
        // a headerless hunk that the parser would mis-attach.
        let raw = "--- //d/UE/Unknown.cpp\t2026\n+++ C:\\dev\\main\\UE\\Unknown.cpp\t2026\n\
                   @@ -1 +1 @@\n-a\n+b\n";
        let norm = normalize_p4_diff(raw, &HashMap::new());
        assert!(!norm.contains("diff --git"), "normalized:\n{norm}");
        assert!(parse_unified_diff(&norm).is_empty());
    }
}
