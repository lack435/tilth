mod definitions;
mod deps;
mod diff;
mod files;
mod grok;
mod read;
mod savings;
mod search;
mod write;

pub(super) use definitions::tool_definitions;
pub(super) use deps::tool_deps;
pub(super) use diff::tool_diff;
pub(super) use files::tool_files;
pub(super) use grok::tool_grok;
pub(super) use read::tool_read;
pub(super) use savings::tool_savings;
pub(super) use search::tool_search;
pub(super) use write::tool_write;

use std::path::PathBuf;

use serde_json::Value;

/// Anchor a caller-supplied path/scope under the absolute-path discipline:
/// the server's process cwd is frozen at spawn and cannot track the caller's
/// live directory, so a relative path is only resolvable when an absolute
/// `root` is supplied to anchor it.
///
/// - **Absolute** path → used as-is (`root` ignored).
/// - **Relative** path + **absolute** `root` → joined under `root`.
/// - **Relative** path + **relative** `root` → `Err` (a relative root
///   reintroduces the cwd hazard it was meant to remove).
/// - **Relative** path + **no** `root` → `Err`.
///
/// `label` names the offending input in the error (e.g. `path` / `scope`).
fn anchor_path(
    raw: &std::path::Path,
    root: Option<&std::path::Path>,
    label: &str,
) -> Result<PathBuf, String> {
    if raw.is_absolute() {
        return Ok(raw.to_path_buf());
    }
    match root {
        Some(r) if r.is_absolute() => Ok(r.join(raw)),
        Some(r) => Err(format!(
            "relative {label} \"{}\" cannot be resolved: \"root\" is itself relative (\"{}\"). \
             Set \"root\" to an absolute checkout directory (the server cannot see your shell's cwd).",
            raw.display(),
            r.display(),
        )),
        None => Err(format!(
            "relative {label} \"{}\" cannot be resolved: pass an absolute {label}, or set \"root\" \
             to an absolute checkout directory (the server cannot see your shell's cwd).",
            raw.display(),
        )),
    }
}

/// Resolve the `scope` arg under the absolute-path discipline (`anchor_path`).
///
/// The require-root discipline fires ONLY when a caller EXPLICITLY passes a
/// relative `scope` without an absolute `root` — that is a deliberate but
/// unresolvable request, since the server cannot see the caller's shell cwd.
///
/// When `scope` is **absent entirely**, this falls back to today's default
/// behavior (server launch cwd, exactly as on `main`): no refusal, no `root`
/// requirement. A bare repo-wide search (no `scope`, no `root`) must keep
/// working — that is the default flow of every session.
///
/// When the anchored path does not resolve to an existing directory, this returns
/// `Err`. It does **not** fall back to `root`, and it does not fall back to `"."`.
///
/// Returning a bare `PathBuf` rather than a `(PathBuf, warning)` pair is deliberate:
/// under this contract there is nothing left to warn about. Every outcome is either a
/// directory the caller asked for or a refusal.
///
/// A scope naming a **file** is accepted and restricts the search to that one file (#141). The
/// three roles `scope` overloads are split rather than conflated: the walk root is the file (so the
/// walk yields exactly that file and enumerates no siblings), while the *base* for path rendering
/// and C/C++ include-containment is the file's parent directory (`search::scope_base`). Feeding the
/// file to both roles is what used to break — `rel(file, file)` collapses to the empty string and
/// `is_within(dir, file)` flips every local header to external — so this returns the file as-is and
/// the divergence is handled downstream where each role is consumed.
pub(super) fn resolve_scope(
    args: &Value,
    root: Option<&std::path::Path>,
) -> Result<PathBuf, String> {
    let scope_arg = args.get("scope").and_then(|v| v.as_str());
    let raw_str = scope_arg.unwrap_or(".");
    let raw: PathBuf = raw_str.into();

    // No explicit scope: behave exactly like main's default-cwd flow. Do not
    // apply the require-root discipline to a value the caller never passed.
    if scope_arg.is_none() {
        let resolved = raw.canonicalize().unwrap_or(raw);
        let cwd = std::env::current_dir().unwrap_or_else(|_| ".".into());
        if resolved == cwd {
            return Ok(".".into());
        }
        return Ok(resolved);
    }

    let anchored = anchor_path(&raw, root, "scope")?;
    let resolved = anchored.canonicalize().unwrap_or(anchored);
    // A directory *or* a file are both valid scopes. A directory recurses; a file scope (#141)
    // restricts the search to that one file, with its parent directory as the render/containment
    // base — see `search::scope_base` and the callers that consume it.
    if resolved.is_dir() || resolved.is_file() {
        return Ok(resolved);
    }

    // Neither a directory nor a file. Refused rather than widened to `root`.
    //
    // This deliberately reverses the earlier root-fallback behaviour. That fallback was
    // chosen over falling back to `"."` (the server cwd, frozen at spawn — the worktree
    // wrong-checkout hazard), and `"."` is indeed the worse of those two. What it missed
    // is that the third option — refusing — beats both, because the fallback silently
    // substitutes a search the caller did not ask for and cannot see they got:
    //
    //   * the soft warning announcing it rides on the *output* string, so when the
    //     widened walk then exceeds the request timeout the warning is discarded with
    //     it. The caller is told "reduce scope" while holding a scope of one file that
    //     was thrown away.
    //   * the widening is unbounded. Measured on a real Unreal checkout: a scope naming
    //     one .cpp file fell back to a 2,043,544-file tree (835 of them tracked) and ran
    //     2m51s against 0.10s for the intended scope, against a 90s request timeout.
    //
    // Three cases, distinguished because they need different fixes from the caller.
    // `is_dir()`/`exists()` both answer "no" to a permission error, so a bare "does not
    // exist" would be a false statement about a directory that demonstrably does.
    // A file scope returned above, so `Ok(true)` here is neither a file nor a directory — a
    // special file or a symlink whose target does not resolve. `is_dir()`/`is_file()` both answer
    // "no" to a permission error too, so the three cases stay distinct.
    let detail: String = match resolved.try_exists() {
        Ok(true) => "exists but is neither a file nor a directory (e.g. a special file, or a \
                     symlink whose target is missing)"
            .to_string(),
        Ok(false) => "does not exist".to_string(),
        Err(e) => format!("cannot be inspected: {e}"),
    };
    // Name the resolved path too, but only when anchoring actually moved it. With a
    // relative scope the two differ and which one is wrong is exactly what the caller
    // cannot tell; with an absolute scope they agree up to path normalization and
    // echoing both just doubles the length of the message for no information.
    // Separator- and verbatim-prefix-insensitive, so a caller who spelled an absolute
    // Windows path with forward slashes is not shown a near-identical twin of it.
    let resolved_str = resolved.display().to_string();
    let same_place = |a: &str| a.trim_start_matches(r"\\?\").replace('\\', "/");
    let where_it_looked = if same_place(&resolved_str) == same_place(raw_str) {
        String::new()
    } else {
        format!(" (resolved to \"{resolved_str}\")")
    };
    Err(format!(
        "scope \"{raw_str}\"{where_it_looked} {detail}. \
         Refusing rather than falling back to a broader scope, which would silently \
         search far more than you asked for."
    ))
}

/// Resolve a relative read path under the absolute-path discipline
/// (`anchor_path`). Absolute paths are used as-is; a relative path requires an
/// absolute `root`, otherwise it is unresolvable (the server cannot see the
/// caller's shell cwd).
pub(super) fn resolve_read_path(
    path: &std::path::Path,
    root: Option<&std::path::Path>,
) -> Result<PathBuf, String> {
    anchor_path(path, root, "path")
}

pub(super) fn apply_budget(output: &str, budget: Option<u64>) -> String {
    match budget {
        Some(b) => crate::budget::apply(output, b),
        None => crate::budget::apply(output, crate::budget::DEFAULT_BUDGET),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_scope_explicit_absolute_arg() {
        let tmp = tempfile::tempdir().unwrap();
        let args = serde_json::json!({ "scope": tmp.path().to_str().unwrap() });
        let scope = resolve_scope(&args, None).unwrap();
        assert_eq!(scope, tmp.path().canonicalize().unwrap());
    }

    #[test]
    fn resolve_scope_no_arg_no_root_defaults_to_cwd() {
        // WHY: the require-root discipline fires ONLY when a caller EXPLICITLY
        // passes a relative scope/path without an absolute root. An omitted
        // `scope` must keep today's default behavior (server launch cwd, exactly
        // as on main) — refusing here would break the default flow of every
        // session (e.g. a bare tilth_search/tilth_files call with no scope).
        let args = serde_json::json!({});
        let scope = resolve_scope(&args, None).unwrap();
        assert_eq!(scope, PathBuf::from("."));
    }

    #[test]
    fn resolve_scope_no_arg_ignores_root() {
        // An omitted scope must default to cwd even when `root` IS supplied —
        // `root` only matters when the caller explicitly passes something
        // relative to anchor.
        let args = serde_json::json!({});
        let tmp = tempfile::tempdir().unwrap();
        let scope = resolve_scope(&args, Some(tmp.path())).unwrap();
        assert_eq!(scope, PathBuf::from("."));
    }

    #[test]
    fn resolve_scope_relative_arg_no_root_errors() {
        // A relative scope with no root is unresolvable (same hazard as omitted).
        let args = serde_json::json!({ "scope": "src" });
        let err = resolve_scope(&args, None).unwrap_err();
        assert!(err.contains("relative scope"), "got: {err}");
    }

    #[test]
    fn resolve_scope_relative_arg_relative_root_errors() {
        // A relative root reintroduces the cwd hazard, so it must be refused.
        let args = serde_json::json!({ "scope": "src" });
        let err = resolve_scope(&args, Some(std::path::Path::new("relative/root"))).unwrap_err();
        assert!(
            err.contains("root") && err.contains("relative"),
            "relative root must be refused: {err}"
        );
    }

    /// A scope naming a file is *accepted* (#141) and resolves to that file itself — not
    /// widened to `root`, not to its parent, and not refused.
    ///
    /// Accepting a file root is what removes the round-trip the issue is about: an agent that
    /// scopes to one file no longer gets a refusal it has to recover from. The three roles
    /// `scope` overloads are split downstream — the walk root is the file, the render and
    /// include-containment base is its parent (`search::scope_base`) — so accepting the file
    /// here does not reintroduce the empty-path / external-header breakage that made the earlier
    /// refusal correct. This pins only the resolution half: the file comes back verbatim.
    #[test]
    fn resolve_scope_file_is_accepted_and_returned_verbatim() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("only.rs");
        std::fs::write(&file, "fn f() {}\n").unwrap();

        let args = serde_json::json!({ "scope": file.to_str().unwrap() });
        let scope = resolve_scope(&args, Some(tmp.path())).unwrap();

        assert_eq!(
            scope,
            file.canonicalize().unwrap(),
            "a file scope must resolve to the file itself, not its parent or root"
        );
    }

    /// A missing scope still refuses, and a file scope now resolves rather than erroring.
    ///
    /// `is_dir()`/`is_file()` both answer "no" to a permission error, so the three diagnoses
    /// (accepted / missing / uninspectable) stay distinct. Windows has no cheap way to build an
    /// unreadable directory in a unit test, so this pins the reachable pair: a file is accepted,
    /// and a genuinely absent scope says "does not exist".
    #[test]
    fn resolve_scope_accepts_file_and_still_refuses_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("f.rs");
        std::fs::write(&file, "fn f() {}\n").unwrap();

        let file_scope = resolve_scope(
            &serde_json::json!({ "scope": file.to_str().unwrap() }),
            None,
        )
        .expect("a file scope must be accepted");
        let missing_err = resolve_scope(
            &serde_json::json!({ "scope": tmp.path().join("nope").to_str().unwrap() }),
            None,
        )
        .unwrap_err();

        assert_eq!(file_scope, file.canonicalize().unwrap());
        assert!(missing_err.contains("does not exist") && !missing_err.contains("neither"));
    }

    #[test]
    fn resolve_scope_missing_scope_errors_even_with_root() {
        // WHY (reverses the earlier root-fallback behaviour, deliberately):
        // falling back to `root` silently substitutes a search the caller did not
        // ask for. The soft warning that announced it rides on the output string,
        // so a widened walk that then exceeds the request timeout discards the
        // warning too — the caller is told "reduce scope" while holding a scope of
        // one file that was thrown away. Refusing is immediate and actionable.
        //
        // The older `"."`-vs-`root` reasoning still holds on its own terms (`"."` is
        // the frozen server cwd and the worse of those two); what it missed is that
        // refusing beats both.
        let tmp = tempfile::tempdir().unwrap();
        let args = serde_json::json!({ "scope": "/nonexistent/directory/zzz" });
        let err = resolve_scope(&args, Some(tmp.path())).unwrap_err();
        assert!(
            err.contains("/nonexistent/directory/zzz") && err.contains("does not exist"),
            "error must name the missing scope: {err}"
        );
        assert!(
            !err.is_empty() && err.contains("Refusing"),
            "error should say why it is not falling back: {err}"
        );
    }

    #[test]
    fn resolve_scope_missing_scope_no_root_errors() {
        // Same refusal with no root available — there was never a safe fallback here.
        let args = serde_json::json!({ "scope": "/nonexistent/directory/zzz" });
        let err = resolve_scope(&args, None).unwrap_err();
        assert!(
            err.contains("/nonexistent/directory/zzz") && err.contains("does not exist"),
            "error must name the missing scope: {err}"
        );
    }

    #[test]
    fn apply_budget_none_caps_at_default() {
        // An output far larger than DEFAULT_BUDGET must be truncated even with
        // no explicit budget — otherwise a broad read/regex/diff blows the host
        // ~25K tool-response limit.
        let oversized = format!(
            "# header line\n{}",
            "filler content that repeats and repeats\n".repeat(20_000)
        );
        let capped = apply_budget(&oversized, None);
        assert!(
            capped.len() < oversized.len(),
            "output should be truncated below the default budget"
        );
        assert!(
            capped.contains("truncated"),
            "truncation notice should be present: {}",
            &capped[capped.len().saturating_sub(120)..]
        );
    }

    #[test]
    fn resolve_read_path_relative_anchors_under_absolute_root() {
        // Guards the spec contract: a relative path + absolute root resolves under
        // root, not under the server's cwd. Prevents worktree agents from silently
        // reading the parent checkout.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let result = resolve_read_path(std::path::Path::new("src/foo.rs"), Some(root)).unwrap();
        assert_eq!(result, root.join("src/foo.rs"));
    }

    #[test]
    fn resolve_read_path_absolute_unaffected_by_root() {
        // Absolute paths must be used as-is regardless of root.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let abs = std::path::Path::new("/tmp/other/file.rs");
        let result = resolve_read_path(abs, Some(root)).unwrap();
        assert_eq!(result, abs);
    }

    #[test]
    fn resolve_read_path_relative_no_root_errors() {
        // WHY (inverted from the old "no root → cwd-relative" guard): the old
        // behavior WAS the worktree bug — a relative path silently resolved
        // against the frozen server cwd. It must now refuse with an actionable
        // message naming the path and the absolute-root escape hatch.
        let err = resolve_read_path(std::path::Path::new("src/foo.rs"), None).unwrap_err();
        assert!(
            err.contains("src/foo.rs") && err.contains("root"),
            "refusal must name the path and the root option: {err}"
        );
    }

    #[test]
    fn resolve_read_path_relative_relative_root_errors() {
        // A relative root reintroduces the cwd hazard, so it must be refused too.
        let err = resolve_read_path(
            std::path::Path::new("src/foo.rs"),
            Some(std::path::Path::new("relative/root")),
        )
        .unwrap_err();
        assert!(
            err.contains("root") && err.contains("relative"),
            "relative root must be refused: {err}"
        );
    }

    #[test]
    fn resolve_scope_with_absolute_root_anchors_relative_scope() {
        // resolve_scope(Some(abs_root)) must resolve a relative scope under root,
        // not under cwd — same contract as resolve_read_path.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let sub = root.join("sub");
        std::fs::create_dir(&sub).unwrap();
        let args = serde_json::json!({ "scope": "sub" });
        let scope = resolve_scope(&args, Some(root)).unwrap();
        assert_eq!(scope, sub.canonicalize().unwrap());
    }

    #[test]
    fn anchor_path_dotdot_not_normalized() {
        // WHY: anchor_path uses root.join(raw) without normalizing `..` components.
        // A path like "../../y" with root "/x" produces "/x/../../y", not "/y".
        // This pins the current behavior so any future traversal normalization is
        // a deliberate, reviewed change — not an accidental side-effect.
        let root = std::path::Path::new("/x");
        let raw = std::path::Path::new("../../y");
        let result = anchor_path(raw, Some(root), "path").unwrap();
        assert_eq!(result, std::path::PathBuf::from("/x/../../y"));
    }
}
