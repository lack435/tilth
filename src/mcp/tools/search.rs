use std::path::PathBuf;
use std::sync::Arc;

use serde_json::Value;

use crate::cache::OutlineCache;
use crate::index::bloom::BloomFilterCache;
use crate::session::Session;

use super::{apply_budget, resolve_scope};

/// Split a multi-symbol query into its individual targets.
///
/// `,` is the documented separator. We also accept `|` because agents reach
/// for regex-style `A|B|C` alternation out of grep muscle memory — without
/// this, that whole string is searched as one literal symbol name and
/// silently returns zero matches (the failure that made an agent conclude
/// tilth's index was "stale" and fall back to grep).
///
/// The one identifier family that legitimately contains a `|` is a C++
/// operator overload: tilth extracts operator names verbatim (see
/// `lang/treesitter.rs`), so `operator|`, `operator|=`, and `operator||` are
/// real symbol names. A part whose `operator` keyword is followed by
/// punctuation is kept intact rather than split on its pipe.
fn split_symbol_list(query: &str) -> Vec<&str> {
    // Comma is the true separator — split on it first, then split each part on
    // `|` unless the part is a pipe-bearing C++ operator overload.
    query
        .split(',')
        .flat_map(|part| {
            let part = part.trim();
            if is_operator_overload(part) {
                vec![part]
            } else {
                part.split('|').map(str::trim).collect::<Vec<_>>()
            }
        })
        .filter(|s| !s.is_empty())
        .collect()
}

/// A C++ operator overload like `operator|` or `operator|=` — the `operator`
/// keyword followed immediately by punctuation. Distinguished from ordinary
/// identifiers such as `operatorId`, where a letter/digit/`_` follows and the
/// pipe (if any) is real alternation.
fn is_operator_overload(part: &str) -> bool {
    part.strip_prefix("operator")
        .is_some_and(|rest| rest.starts_with(|c: char| !c.is_alphanumeric() && c != '_'))
}

pub(in crate::mcp) fn tool_search(
    args: &Value,
    cache: &OutlineCache,
    session: &Session,
    bloom: &Arc<BloomFilterCache>,
) -> Result<String, String> {
    let query = args
        .get("query")
        .and_then(|v| v.as_str())
        .ok_or("missing required parameter: query")?;
    let root = args
        .get("root")
        .and_then(|v| v.as_str())
        .map(std::path::Path::new);
    let scope = resolve_scope(args, root)?;
    let kind = args
        .get("kind")
        .and_then(|v| v.as_str())
        .unwrap_or("symbol");
    let expand = args
        .get("expand")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(2) as usize;
    let context_path = args
        .get("context")
        .and_then(|v| v.as_str())
        .map(PathBuf::from);
    let context = context_path.as_deref();
    let glob = args.get("glob").and_then(|v| v.as_str());
    let budget = args.get("budget").and_then(serde_json::Value::as_u64);
    // Tri-state: absent → smart-case (the CLI default), `true` → force
    // insensitive, `false` → force sensitive. Consulted only by the content and
    // regex arms; symbol/callers ignore it.
    let case = match args.get("ignore_case").and_then(serde_json::Value::as_bool) {
        None => crate::types::CaseMode::Smart,
        Some(true) => crate::types::CaseMode::Insensitive,
        Some(false) => crate::types::CaseMode::Sensitive,
    };

    let output = match kind {
        "symbol" => {
            let queries = split_symbol_list(query);
            match queries.len() {
                0 => return Err("missing required parameter: query".into()),
                1 => crate::search::search_symbol_expanded(
                    queries[0], &scope, cache, session, bloom, expand, context, glob, false, budget,
                ),
                2..=5 => crate::search::search_multi_symbol_expanded(
                    &queries, &scope, cache, session, bloom, expand, context, glob, false, budget,
                ),
                _ => {
                    return Err(format!(
                        "multi-symbol search limited to 5 queries (got {})",
                        queries.len()
                    ))
                }
            }
        }
        "content" => crate::search::search_content_expanded(
            query, &scope, cache, session, expand, context, glob, false, budget, case,
        ),
        "regex" => {
            let result =
                crate::search::content::search(query, &scope, true, context, glob, false, case)
                    .map_err(|e| e.to_string())?;
            crate::search::format_raw_result(&result, cache)
        }
        "callers" => {
            let targets = split_symbol_list(query);
            match targets.len() {
                0 => return Err("missing required parameter: query".into()),
                1 => crate::search::callers::search_callers_expanded(
                    targets[0], &scope, bloom, expand, context, glob, false,
                ),
                2..=5 => crate::search::callers::search_callers_multi_expanded(
                    &targets, &scope, bloom, expand, context, glob, false,
                ),
                _ => {
                    return Err(format!(
                        "multi-target callers search limited to 5 queries (got {})",
                        targets.len()
                    ))
                }
            }
        }
        _ => {
            return Err(format!(
                "unknown search kind: {kind}. Use: symbol, content, regex, callers"
            ))
        }
    }
    .map_err(|e| e.to_string())?;

    Ok(apply_budget(&output, budget))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::OutlineCache;
    use crate::index::bloom::BloomFilterCache;
    use crate::session::Session;

    /// Regression: `kind=callers` with a comma query must search each target
    /// separately, not for a literal symbol named "alpha,beta". Before the
    /// comma-split arm this returned an empty no-callers message.
    #[test]
    fn callers_comma_query_finds_both_targets() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("lib.rs"),
            "fn alpha() {}\n\
             fn beta() {}\n\
             fn uses_alpha() { alpha(); }\n\
             fn uses_beta() { beta(); }\n",
        )
        .unwrap();

        let cache = OutlineCache::new();
        let session = Session::new();
        let bloom = std::sync::Arc::new(BloomFilterCache::new());
        let args = serde_json::json!({
            "query": "alpha,beta",
            "kind": "callers",
            "scope": tmp.path().to_str().unwrap(),
        });

        let out = tool_search(&args, &cache, &session, &bloom).unwrap();

        // Both targets must be reported with a real call site, not a single
        // literal "alpha,beta" lookup that finds nothing. Header uses the
        // unified single-target shape: `# Callers of "<target>" in <scope>`.
        assert!(
            out.contains("Callers of \"alpha\""),
            "missing alpha section: {out}"
        );
        assert!(
            out.contains("Callers of \"beta\""),
            "missing beta section: {out}"
        );
        assert!(
            out.contains("uses_alpha"),
            "alpha call site not found: {out}"
        );
        assert!(out.contains("uses_beta"), "beta call site not found: {out}");
        // The literal combined string must never be searched as one symbol.
        assert!(
            !out.contains("\"alpha,beta\""),
            "comma query was treated as a literal symbol: {out}"
        );
    }

    /// Regression for the duplicate-target render bug: query "alpha,alpha"
    /// must still report alpha's call site once, not render an empty
    /// no-callers section on the second occurrence after the first consumed
    /// the matched bucket.
    #[test]
    fn callers_duplicate_target_does_not_render_empty_section() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("lib.rs"),
            "fn alpha() {}\n\
             fn uses_alpha() { alpha(); }\n",
        )
        .unwrap();

        let cache = OutlineCache::new();
        let session = Session::new();
        let bloom = std::sync::Arc::new(BloomFilterCache::new());
        let args = serde_json::json!({
            "query": "alpha,alpha",
            "kind": "callers",
            "scope": tmp.path().to_str().unwrap(),
        });

        let out = tool_search(&args, &cache, &session, &bloom).unwrap();

        assert!(
            out.contains("uses_alpha"),
            "alpha call site not found: {out}"
        );
        // The duplicate must collapse to a single section: no no-callers
        // message should appear — that is what the second occurrence rendered
        // before the dedupe consumed the bucket on the first pass.
        assert!(
            !out.contains("no call sites") && !out.contains("no direct call sites"),
            "duplicate target rendered a false no-callers section: {out}"
        );
    }

    /// HIGH finding from PR review: the multi-target path must not silently
    /// drop the single-target path's "Adaptive 2nd-hop impact analysis".
    /// `alpha` is called by exactly `IMPACT_FANOUT_THRESHOLD`-or-fewer unique
    /// functions (one: `uses_alpha`), which are themselves called by
    /// `hop2_alpha` — so the 2-target search "alpha,beta" must show a 2nd-hop
    /// section for the alpha bucket, same as a lone `callers("alpha")` would.
    #[test]
    fn callers_multi_target_includes_second_hop_impact_per_bucket() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("lib.rs"),
            "fn alpha() {}\n\
             fn beta() {}\n\
             fn uses_alpha() { alpha(); }\n\
             fn hop2_alpha() { uses_alpha(); }\n\
             fn uses_beta() { beta(); }\n",
        )
        .unwrap();

        let cache = OutlineCache::new();
        let session = Session::new();
        let bloom = std::sync::Arc::new(BloomFilterCache::new());

        // Single-target baseline: what callers("alpha") alone produces.
        let single_args = serde_json::json!({
            "query": "alpha",
            "kind": "callers",
            "scope": tmp.path().to_str().unwrap(),
        });
        let single_out = tool_search(&single_args, &cache, &session, &bloom).unwrap();
        assert!(
            single_out.contains("impact (2nd hop)"),
            "single-target baseline should show 2nd-hop impact: {single_out}"
        );
        assert!(single_out.contains("hop2_alpha"));

        // Multi-target: "alpha,beta" must not omit what a lone "alpha" search
        // would show for the alpha bucket.
        let multi_args = serde_json::json!({
            "query": "alpha,beta",
            "kind": "callers",
            "scope": tmp.path().to_str().unwrap(),
        });
        let multi_out = tool_search(&multi_args, &cache, &session, &bloom).unwrap();
        assert!(
            multi_out.contains("impact (2nd hop)"),
            "multi-target alpha bucket dropped the 2nd-hop impact section: {multi_out}"
        );
        assert!(
            multi_out.contains("hop2_alpha"),
            "multi-target alpha bucket missing the hop-2 caller: {multi_out}"
        );
    }

    /// MED finding from PR review: single- and multi-target output must use
    /// the same header shape for the same target — the review found multi
    /// diverging into a `## callers of "foo"` / `### path:line` style while
    /// single used `# Callers of "foo" in <scope> — N call site(s)` /
    /// `## path:line`. A caller diffing single vs. one bucket of multi should
    /// see the identical shape (same target, same scope, same one hit).
    #[test]
    fn callers_multi_target_header_matches_single_target_shape() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("lib.rs"),
            "fn alpha() {}\n\
             fn beta() {}\n\
             fn uses_alpha() { alpha(); }\n\
             fn uses_beta() { beta(); }\n",
        )
        .unwrap();

        let cache = OutlineCache::new();
        let session = Session::new();
        let bloom = std::sync::Arc::new(BloomFilterCache::new());

        let single_args = serde_json::json!({
            "query": "alpha",
            "kind": "callers",
            "scope": tmp.path().to_str().unwrap(),
        });
        let single_out = tool_search(&single_args, &cache, &session, &bloom).unwrap();

        let multi_args = serde_json::json!({
            "query": "alpha,beta",
            "kind": "callers",
            "scope": tmp.path().to_str().unwrap(),
        });
        let multi_out = tool_search(&multi_args, &cache, &session, &bloom).unwrap();

        // Top-level bucket header: same "# Callers of ... — N call site(s)" shape.
        assert!(
            single_out.contains("# Callers of \"alpha\""),
            "single-target header shape missing: {single_out}"
        );
        assert!(
            multi_out.contains("# Callers of \"alpha\""),
            "multi-target alpha bucket must render the single-target header shape, \
             not a divergent '## callers of' shape: {multi_out}"
        );
        assert!(
            single_out.contains("1 call site"),
            "single-target count phrase missing: {single_out}"
        );
        assert!(
            multi_out.contains("1 call site"),
            "multi-target alpha bucket must render the same count phrase: {multi_out}"
        );

        // Call-site sub-header: same "## path:line [caller: name]" shape,
        // not multi's divergent "### path:line [caller: name]".
        assert!(
            single_out.contains("[caller: uses_alpha]"),
            "single-target caller label missing: {single_out}"
        );
        assert!(
            multi_out.contains("[caller: uses_alpha]"),
            "multi-target alpha bucket must render the same caller label: {multi_out}"
        );
        assert!(
            !multi_out.contains("### lib.rs"),
            "multi-target must use single-target's '##' sub-header level, not '###': {multi_out}"
        );
    }

    /// A rare target must not be starved by a hit-rich one sharing the walk.
    ///
    /// This originally guarded `BATCH_EARLY_QUIT`, a walk-wide raw-match budget shared by
    /// every target and checked once per file visited: 60 `alpha` sites across 60 files
    /// could exhaust it before `z_beta.rs` — which sorts after all of them — was ever
    /// read, so `beta` reported nothing. Scaling the budget by target count was the fix
    /// then; removing the budget is the fix now, because a count-based cutoff over a
    /// parallel walk cannot be deterministic (see `src/search/callers.rs`).
    ///
    /// Kept as a scenario-level guard rather than deleted with the mechanism: starvation
    /// of a later file by an earlier one is exactly what any future reintroduction of a
    /// walk-wide cutoff would cause, and this fixture reproduces it.
    #[test]
    fn callers_multi_target_later_target_not_starved_by_hit_rich_earlier_target() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("defs.rs"), "fn alpha() {}\nfn beta() {}\n").unwrap();
        for i in 0..60 {
            std::fs::write(
                tmp.path().join(format!("a_{i:02}.rs")),
                format!("fn uses_alpha_{i}() {{ alpha(); }}\n"),
            )
            .unwrap();
        }
        // Sorts after every "a_*.rs" file — only reached by a walk that visits every
        // candidate rather than stopping on a match count.
        std::fs::write(tmp.path().join("z_beta.rs"), "fn uses_beta() { beta(); }\n").unwrap();

        let cache = OutlineCache::new();
        let session = Session::new();
        let bloom = std::sync::Arc::new(BloomFilterCache::new());
        let args = serde_json::json!({
            "query": "alpha,beta",
            "kind": "callers",
            "scope": tmp.path().to_str().unwrap(),
        });

        let out = tool_search(&args, &cache, &session, &bloom).unwrap();

        assert!(
            out.contains("uses_beta"),
            "beta's call site was starved by alpha's — the walk stopped before reaching \
             z_beta.rs, which means a match-count cutoff is back: {out}"
        );
    }

    /// WHY: the require-root discipline fires ONLY when a caller EXPLICITLY
    /// passes a relative scope/path without an absolute root. A bare
    /// `tilth_search(query)` call with no scope is the default flow of every
    /// session and must keep working exactly as it does on main — refusing
    /// here would break every session's default search. This inverts the PR's
    /// original (too strict) assertion.
    ///
    /// Asserts only `is_ok()`, not the response body: the body is real search
    /// output over whatever tree the test runs in (including this very source
    /// file), so substring-matching it is not a reliable way to detect a
    /// require-root refusal. `resolve_scope`'s own unit tests in
    /// `mcp::tools::tests` already pin the exact refusal-vs-default-cwd
    /// behavior directly; this test only pins that `tool_search` propagates
    /// success through to its caller instead of swallowing it into an error.
    #[test]
    fn no_scope_no_root_defaults_to_cwd() {
        let cache = OutlineCache::new();
        let session = Session::new();
        let bloom = Arc::new(BloomFilterCache::new());
        let args = serde_json::json!({ "query": "anything_unlikely_to_match_zzz" });
        let result = tool_search(&args, &cache, &session, &bloom);
        assert!(
            result.is_ok(),
            "bare search must default to cwd, not refuse: {result:?}"
        );
    }

    #[test]
    fn split_symbol_list_accepts_both_separators() {
        // Comma — the documented separator.
        assert_eq!(split_symbol_list("a,b,c"), vec!["a", "b", "c"]);
        // Pipe — grep/regex-alternation muscle memory.
        assert_eq!(split_symbol_list("a|b|c"), vec!["a", "b", "c"]);
        // Mixed, with surrounding whitespace to trim and an empty run to drop.
        assert_eq!(split_symbol_list("a, b | c ||"), vec!["a", "b", "c"]);
        // A lone identifier is a single target, untouched.
        assert_eq!(split_symbol_list("handleRequest"), vec!["handleRequest"]);
    }

    /// A C++ operator overload's name legitimately contains a pipe. tilth
    /// extracts these verbatim, so an agent copies `operator|` straight out of
    /// an outline into a search — splitting it into a bare `operator` would
    /// silently lose the definition. Common in Unreal C++ (`ENUM_CLASS_FLAGS`
    /// generates `operator|`, `operator|=`).
    #[test]
    fn split_symbol_list_preserves_cpp_operator_overloads() {
        assert_eq!(split_symbol_list("operator|"), vec!["operator|"]);
        assert_eq!(split_symbol_list("operator|="), vec!["operator|="]);
        assert_eq!(split_symbol_list("operator||"), vec!["operator||"]);
        // Comma still separates an overload from other targets.
        assert_eq!(
            split_symbol_list("operator|,operator&"),
            vec!["operator|", "operator&"]
        );
        // An ordinary identifier that merely starts with "operator" is NOT an
        // overload — the pipe is real alternation and must still split.
        assert_eq!(
            split_symbol_list("operatorName|foo"),
            vec!["operatorName", "foo"]
        );
    }

    /// Regression: a symbol query written with regex-style `|` alternation —
    /// `alpha|beta` out of grep habit — must resolve each symbol, not search
    /// for one literal symbol named "alpha|beta" and return zero matches. This
    /// is the exact failure mode that made an agent conclude tilth's index was
    /// "stale" and fall back to grep.
    #[test]
    fn symbol_pipe_query_finds_each_symbol() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("lib.rs"),
            "fn alpha() {}\n\
             fn beta() {}\n\
             fn gamma() {}\n",
        )
        .unwrap();

        let cache = OutlineCache::new();
        let session = Session::new();
        let bloom = std::sync::Arc::new(BloomFilterCache::new());
        let args = serde_json::json!({
            "query": "alpha|beta",
            "scope": tmp.path().to_str().unwrap(),
        });

        let out = tool_search(&args, &cache, &session, &bloom).unwrap();

        assert!(out.contains("alpha"), "alpha not found: {out}");
        assert!(out.contains("beta"), "beta not found: {out}");
        // The combined literal must never be searched as one symbol name.
        assert!(
            !out.contains("\"alpha|beta\""),
            "pipe query was treated as a literal symbol: {out}"
        );
    }

    /// The pipe separator must reach the callers path too, matching the comma
    /// behavior guarded by `callers_comma_query_finds_both_targets`.
    #[test]
    fn callers_pipe_query_finds_both_targets() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("lib.rs"),
            "fn alpha() {}\n\
             fn beta() {}\n\
             fn uses_alpha() { alpha(); }\n\
             fn uses_beta() { beta(); }\n",
        )
        .unwrap();

        let cache = OutlineCache::new();
        let session = Session::new();
        let bloom = std::sync::Arc::new(BloomFilterCache::new());
        let args = serde_json::json!({
            "query": "alpha|beta",
            "kind": "callers",
            "scope": tmp.path().to_str().unwrap(),
        });

        let out = tool_search(&args, &cache, &session, &bloom).unwrap();

        assert!(
            out.contains("Callers of \"alpha\""),
            "missing alpha section: {out}"
        );
        assert!(
            out.contains("Callers of \"beta\""),
            "missing beta section: {out}"
        );
        assert!(out.contains("uses_alpha"), "alpha call site missing: {out}");
        assert!(out.contains("uses_beta"), "beta call site missing: {out}");
    }

    /// The `ignore_case` tri-state maps through `tool_search` end to end:
    /// absent → smart-case (all-lowercase query folds case), `false` → strictly
    /// sensitive (no match on the uppercase-only token), `true` → insensitive.
    /// The matched line's text is echoed into the output, so its presence is a
    /// reliable proxy for a hit.
    #[test]
    fn ignore_case_tristate_controls_content_matching() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("data.txt"), "ZEBRA_TOKEN lives here\n").unwrap();

        let cache = OutlineCache::new();
        let session = Session::new();
        let bloom = std::sync::Arc::new(BloomFilterCache::new());

        let run = |ignore_case: Option<bool>| {
            let mut args = serde_json::json!({
                "query": "zebra_token",
                "kind": "content",
                "scope": tmp.path().to_str().unwrap(),
            });
            if let Some(b) = ignore_case {
                args["ignore_case"] = serde_json::Value::Bool(b);
            }
            tool_search(&args, &cache, &session, &bloom).unwrap()
        };

        // Absent → smart-case: an all-lowercase query folds case and finds the
        // uppercase token.
        assert!(
            run(None).contains("ZEBRA_TOKEN"),
            "smart-case default should match the uppercase token"
        );
        // Explicit false → sensitive: the lowercase query must NOT match.
        assert!(
            !run(Some(false)).contains("ZEBRA_TOKEN"),
            "ignore_case=false must be strictly case-sensitive"
        );
        // Explicit true → insensitive.
        assert!(
            run(Some(true)).contains("ZEBRA_TOKEN"),
            "ignore_case=true must fold case"
        );
    }

    /// A file path as `scope` (#141) restricts a symbol search to that one file and never
    /// enumerates its siblings — the whole point of the feature.
    ///
    /// `shared_symbol` is used in both `target.rs` and `sibling.rs`. Scoped to `target.rs`, only
    /// the target's usage may appear. Mutating the walk root back to the parent (or dropping the
    /// file-aware `WalkBuilder`) resurfaces `sibling.rs` and fails this.
    #[test]
    fn symbol_file_scope_searches_only_that_file() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("target.rs"),
            "pub fn a() { shared_symbol(); }\n",
        )
        .unwrap();
        std::fs::write(
            tmp.path().join("sibling.rs"),
            "pub fn b() { shared_symbol(); }\n",
        )
        .unwrap();

        let cache = OutlineCache::new();
        let session = Session::new();
        let bloom = Arc::new(BloomFilterCache::new());
        let args = serde_json::json!({
            "query": "shared_symbol",
            "scope": tmp.path().join("target.rs").to_str().unwrap(),
        });

        let out = tool_search(&args, &cache, &session, &bloom).unwrap();

        // The match path renders as the basename (relative to the file's parent), not the empty
        // string a file render-base would produce and not an absolute path.
        assert!(
            out.contains("target.rs:"),
            "the target file's match must render as a basename: {out}"
        );
        assert!(
            !out.contains("sibling"),
            "a file scope must not walk siblings: {out}"
        );
    }

    /// The same restriction for content search.
    #[test]
    fn content_file_scope_searches_only_that_file() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("target.rs"), "// MARKER_TOKEN here\n").unwrap();
        std::fs::write(tmp.path().join("sibling.rs"), "// MARKER_TOKEN there\n").unwrap();

        let cache = OutlineCache::new();
        let session = Session::new();
        let bloom = Arc::new(BloomFilterCache::new());
        let args = serde_json::json!({
            "query": "MARKER_TOKEN",
            "kind": "content",
            "scope": tmp.path().join("target.rs").to_str().unwrap(),
        });

        let out = tool_search(&args, &cache, &session, &bloom).unwrap();

        assert!(
            out.contains("target.rs"),
            "target file's match missing: {out}"
        );
        assert!(
            !out.contains("sibling"),
            "a file content scope must not walk siblings: {out}"
        );
    }

    /// And for callers: a call site of `foo` in a sibling must not appear when scoped to one file.
    #[test]
    fn callers_file_scope_searches_only_that_file() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("target.rs"),
            "fn foo() {}\nfn in_target() { foo(); }\n",
        )
        .unwrap();
        std::fs::write(
            tmp.path().join("sibling.rs"),
            "fn in_sibling() { foo(); }\n",
        )
        .unwrap();

        let cache = OutlineCache::new();
        let session = Session::new();
        let bloom = Arc::new(BloomFilterCache::new());
        let args = serde_json::json!({
            "query": "foo",
            "kind": "callers",
            "scope": tmp.path().join("target.rs").to_str().unwrap(),
        });

        let out = tool_search(&args, &cache, &session, &bloom).unwrap();

        assert!(out.contains("in_target"), "target call site missing: {out}");
        assert!(
            !out.contains("in_sibling") && !out.contains("sibling.rs"),
            "a file callers scope must not walk siblings: {out}"
        );
    }

    /// The basename-outline fallback must respect the file scope too (#141): a single-word query
    /// that happens to match a *sibling's* filename must not drag that sibling's outline into a
    /// one-file search. This pins the `walk_root` field — the fallback walks the walk root (the
    /// file), not the render base (its parent). Routing the fallback back through `result.scope`
    /// resurfaces `thing.rs` and fails this.
    #[test]
    fn file_scope_basename_fallback_does_not_surface_sibling() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("target.rs"), "pub fn run() { thing(); }\n").unwrap();
        // A sibling whose *stem* equals the query. The fallback would surface its outline if it
        // walked the parent directory.
        std::fs::write(tmp.path().join("thing.rs"), "pub fn thing() {}\n").unwrap();

        let cache = OutlineCache::new();
        let session = Session::new();
        let bloom = Arc::new(BloomFilterCache::new());
        let args = serde_json::json!({
            "query": "thing",
            "scope": tmp.path().join("target.rs").to_str().unwrap(),
        });

        let out = tool_search(&args, &cache, &session, &bloom).unwrap();

        assert!(
            !out.contains("thing.rs") && !out.contains("File overview"),
            "a file scope must not surface a sibling via the basename-outline fallback: {out}"
        );
    }

    /// An EXPLICITLY passed relative scope with no absolute root to anchor it
    /// is unresolvable (the server cannot see the caller's shell cwd) — this
    /// must still refuse.
    #[test]
    fn explicit_relative_scope_no_root_errors() {
        let cache = OutlineCache::new();
        let session = Session::new();
        let bloom = Arc::new(BloomFilterCache::new());
        let args = serde_json::json!({ "query": "anything", "scope": "some/relative/dir" });
        let err = tool_search(&args, &cache, &session, &bloom).unwrap_err();
        assert!(
            err.contains("relative scope") && err.contains("root"),
            "explicit relative scope without root must refuse: {err}"
        );
    }
}
