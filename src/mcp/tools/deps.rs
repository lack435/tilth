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
    let scope = resolve_scope(args, root)?;
    let budget = args
        .get("budget")
        .and_then(serde_json::Value::as_u64)
        .map(|b| b as usize);

    let deps_result =
        crate::search::deps::analyze_deps(&path, &scope, bloom).map_err(|e| e.to_string())?;
    Ok(crate::search::deps::format_deps(&deps_result, budget))
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

    /// A `scope` that does not exist is refused, not widened to `root`.
    ///
    /// This replaces `a_non_canonical_root_fallback_does_not_make_a_file_its_own_dependent`,
    /// which drove #97 through this layer using the missing-directory root fallback — the one
    /// spelling that reached `analyze_deps` uncanonicalized. That fallback is gone: it silently
    /// substituted a search the caller never asked for, and on a real Unreal checkout turned a
    /// one-file scope into a 2M-file walk.
    ///
    /// Deleting it costs no coverage, which was checked rather than assumed: on `main`, that test
    /// **passed with #97's `scope.canonicalize()` reverted**. Its assertion could not fail against
    /// that revert, because #98 had already moved the self-reference filter onto
    /// `CallerMatch::identity` (a canonicalized path) and so made it independent of how `scope` is
    /// spelled. What genuinely holds #97 down is the rendering pair in
    /// `search::deps::scope_spelling_tests` — see the table on that module.
    #[test]
    fn a_missing_scope_is_refused_rather_than_widened_to_root() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("target.rs"), "pub fn shared() -> u32 { 1 }\n").unwrap();

        let args = serde_json::json!({
            "path": src.join("target.rs").to_str().unwrap(),
            "scope": "no_such_dir",
            "root": tmp.path().to_str().unwrap(),
        });
        let err = tool_deps(&args, &bloom())
            .expect_err("a scope that does not exist must be refused, not silently widened");

        assert!(
            err.contains("no_such_dir") && err.contains("does not exist"),
            "refusal must name the bad scope: {err}"
        );
    }

    /// A file `scope` is refused here too, and — the point of the test — refused *before*
    /// `analyze_deps` runs, so it cannot answer with the include-boundary defect.
    ///
    /// `scope` is the containment boundary `resolve_c_include` uses. Handing it a file makes
    /// `is_within(dir, boundary)` false for every directory, so local headers resolve as
    /// external. Measured on a build that accepted file scopes: a UE-style
    /// `#include "Module/Public/Thing.h"` went from `1 local, 0 external` to
    /// `0 local, 1 external`. Refusing keeps that unreachable until the walk-root and
    /// boundary roles are actually separated.
    #[test]
    fn a_file_scope_is_refused_before_deps_can_misclassify_includes() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(
            src.join("target.rs"),
            "pub fn shared() -> u32 { 1 }\npub fn inner() -> u32 { shared() + 1 }\n",
        )
        .unwrap();

        let args = serde_json::json!({
            "path": src.join("target.rs").to_str().unwrap(),
            "scope": src.join("target.rs").to_str().unwrap(),
            "root": tmp.path().to_str().unwrap(),
        });
        let err = tool_deps(&args, &bloom())
            .expect_err("a file scope must be refused until the boundary role is split out");

        assert!(
            err.contains("not a directory") && err.contains("parent directory"),
            "refusal must diagnose the file and name the fix: {err}"
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
