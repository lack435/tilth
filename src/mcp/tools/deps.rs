use std::path::PathBuf;
use std::sync::Arc;

use serde_json::Value;

use crate::index::bloom::BloomFilterCache;

use super::resolve_scope;

pub(in crate::mcp) fn tool_deps(
    args: &Value,
    bloom: &Arc<BloomFilterCache>,
) -> Result<String, String> {
    let path_str = args
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or("missing required parameter: path")?;
    let root = args
        .get("root")
        .and_then(|v| v.as_str())
        .map(std::path::Path::new);
    let path = super::resolve_read_path(&PathBuf::from(path_str), root)?;
    let (scope, scope_warning) = resolve_scope(args, root)?;
    let budget = args
        .get("budget")
        .and_then(serde_json::Value::as_u64)
        .map(|b| b as usize);

    let deps_result =
        crate::search::deps::analyze_deps(&path, &scope, bloom).map_err(|e| e.to_string())?;
    let mut output = scope_warning.unwrap_or_default();
    output.push_str(&crate::search::deps::format_deps(&deps_result, budget));
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bloom() -> Arc<BloomFilterCache> {
        Arc::new(BloomFilterCache::new())
    }

    #[test]
    fn relative_path_no_root_errors() {
        // WHY: tilth_deps resolves its `path` arg through resolve_read_path. A
        // relative path with no absolute root silently resolved against the
        // frozen server cwd before this spec. The `?` on the path resolution must
        // propagate the refusal, naming the path and the root escape hatch.
        let args = serde_json::json!({ "path": "src/foo.rs" });
        let err = tool_deps(&args, &bloom()).unwrap_err();
        assert!(
            err.contains("src/foo.rs") && err.contains("root"),
            "relative deps path without root must refuse: {err}"
        );
    }

    #[test]
    fn absolute_path_omitted_scope_no_root_defaults_to_cwd() {
        // WHY: the require-root discipline fires ONLY when a caller EXPLICITLY
        // passes a relative path/scope without an absolute root. `scope` is
        // never required by tilth_deps — an absolute `path` with an omitted
        // `scope` must resolve scope to the server's default cwd (exactly as on
        // main), not refuse. This inverts the PR's original (too strict)
        // assertion, which broke the default flow (path-only tilth_deps calls).
        let tmp = tempfile::tempdir().unwrap();
        let abs = tmp.path().join("foo.rs");
        std::fs::write(&abs, "fn foo() {}\n").unwrap();
        let args = serde_json::json!({ "path": abs.to_str().unwrap() });
        let out = tool_deps(&args, &bloom())
            .expect("absolute path + omitted scope must default to cwd, not refuse");
        assert!(
            !out.contains("cannot be resolved"),
            "unexpected refusal: {out}"
        );
    }

    /// #97 through the layer that shipped it, and on the platform that shipped it.
    ///
    /// `resolve_scope` canonicalizes every scope it resolves, so the one spelling that reaches
    /// `analyze_deps` uncanonicalized is its missing-directory fallback: a `scope` that is not a
    /// directory plus an absolute `root` returns that **`root` as the caller spelled it**
    /// (`resolve_scope`'s `Some(r) if r.is_absolute()` arm). #97 called this the cheapest
    /// reproduction, and it was measured by hand and guarded by nothing.
    ///
    /// The `root` here is deliberately spelled `<tmp>/src/..`, which is non-canonical on every
    /// platform. That matters: a plain tempdir path is already canonical on the Linux runner, so
    /// the same test written the obvious way would exercise nothing there — which is the platform
    /// the defect actually shipped on, since `resolve_scope`'s other uncanonical spelling (a
    /// literal `"."`) can only arise on Linux.
    ///
    /// Asserts the *quiet* half. The rendering was the visible symptom; a file reported as its own
    /// dependent is the wrong answer, and the one worth a layer test.
    #[test]
    fn a_non_canonical_root_fallback_does_not_make_a_file_its_own_dependent() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(
            src.join("target.rs"),
            "pub fn shared() -> u32 { 1 }\npub fn inner() -> u32 { shared() + 1 }\n",
        )
        .unwrap();
        std::fs::write(
            src.join("a.rs"),
            "use crate::target::shared;\n\npub fn go() -> u32 { shared() }\n",
        )
        .unwrap();

        let args = serde_json::json!({
            "path": src.join("target.rs").to_str().unwrap(),
            "scope": "no_such_dir",
            "root": tmp.path().join("src/..").to_str().unwrap(),
        });
        let out = tool_deps(&args, &bloom()).expect("missing scope + absolute root must not error");

        assert!(
            out.contains("1 dependent"),
            "control failed: the fixture must produce exactly the one real dependent, or the \
             assertion below proves nothing:\n{out}"
        );
        assert!(
            !out.contains("target.rs:"),
            "the target is listed among its own dependents — the self-reference filter compared \
             a canonical target against paths spelled like the raw root (#97):\n{out}"
        );
    }

    #[test]
    fn absolute_path_explicit_relative_scope_no_root_errors() {
        // An EXPLICITLY passed relative `scope` with no absolute root is
        // unresolvable (the server cannot see the caller's shell cwd) — this
        // must still refuse, even though `path` is absolute.
        let tmp = tempfile::tempdir().unwrap();
        let abs = tmp.path().join("foo.rs");
        std::fs::write(&abs, "fn foo() {}\n").unwrap();
        let args = serde_json::json!({
            "path": abs.to_str().unwrap(),
            "scope": "some/relative/dir",
        });
        let err = tool_deps(&args, &bloom()).unwrap_err();
        assert!(
            err.contains("relative scope") && err.contains("root"),
            "explicit relative scope without root must refuse: {err}"
        );
    }
}
