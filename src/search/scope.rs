//! Enclosing-scope annotator: given a `(file, line)`, return the nearest
//! enclosing definition, qualified with its containing type or module.
//! Used by the search formatter to annotate usages with their containing
//! function/class/module, and internally by `callers::find_enclosing_function`
//! when reporting the calling-function context of a call site.

use std::path::Path;

use crate::cache::{OutlineCache, ScopeLabel};
use crate::lang::treesitter::{
    cpp_misparsed_class_name, extract_definition_name, extract_elixir_definition_name,
    is_definition_node, is_elixir_definition, node_text_simple,
};

/// Type-like node kinds that can enclose a function definition.
const TYPE_KINDS: &[&str] = &[
    "class_declaration",
    "class_definition",
    "struct_item",
    "impl_item",
    "interface_declaration",
    "trait_item",
    "trait_declaration",
    "type_declaration",
    "enum_item",
    "enum_declaration",
    "module",
    "mod_item",
    "namespace_definition",
    // C/C++ — qualifies a member declared inside a class body as `Holder.Work`.
    // Kept in sync with `treesitter::SPECIFIER_KINDS`; `enum_specifier` is included so
    // an enumerator resolves as `Mode.On` rather than bare.
    "class_specifier",
    "struct_specifier",
    "union_specifier",
    "enum_specifier",
];

/// Walk up the AST from `node` to the nearest definition, qualified with its
/// enclosing type/module if one wraps it. Returns the AST node so the caller
/// can read its kind, plus the rendered name and line range.
pub(super) fn walk_to_enclosing_definition<'a>(
    node: tree_sitter::Node<'a>,
    lines: &[&str],
    lang: crate::types::Lang,
) -> Option<(tree_sitter::Node<'a>, String, (u32, u32))> {
    let mut current = Some(node);
    while let Some(n) = current {
        let def_name = if is_definition_node(n, Some(lang)) {
            extract_definition_name(n, lines)
        } else if lang == crate::types::Lang::Elixir && is_elixir_definition(n, lines) {
            extract_elixir_definition_name(n, lines)
        } else {
            None
        };

        if let Some(name) = def_name {
            let range = (
                n.start_position().row as u32 + 1,
                n.end_position().row as u32 + 1,
            );

            // Walk further up to find an enclosing type/module and qualify the name.
            // `defmodule` is a `call` node, not in TYPE_KINDS, so Elixir needs a
            // separate check to produce `Module.func`.
            let mut parent = n.parent();
            while let Some(p) = parent {
                if TYPE_KINDS.contains(&p.kind()) {
                    if let Some(type_name) = extract_definition_name(p, lines) {
                        return Some((n, format!("{type_name}.{name}"), range));
                    }
                }
                if lang == crate::types::Lang::Elixir && is_elixir_definition(p, lines) {
                    if let Some(type_name) = extract_elixir_definition_name(p, lines) {
                        return Some((n, format!("{type_name}.{name}"), range));
                    }
                }
                parent = p.parent();
            }

            return Some((n, name, range));
        }
        current = n.parent();
    }
    None
}

/// Find the nearest enclosing definition for `(path, line)`.
///
/// Served from the session's label cache when this file+mtime has already been asked about
/// that line; otherwise the file is parsed, the answer taken, and **the tree dropped before
/// returning**. Nothing here retains a tree — see the `cache` module header for the 1.2 GB
/// that used to buy.
///
/// Prefer `warm_labels` when several lines of the same page need answering: this entry point
/// parses once per *call*, that one parses once per *file*.
///
/// Returns `None` if the file isn't a code file, the parse fails, or `line` sits at the top
/// level outside any definition.
pub fn enclosing_definition_at(path: &Path, line: u32, cache: &OutlineCache) -> Option<ScopeLabel> {
    if line == 0 {
        return None;
    }
    let mtime = file_mtime(path)?;
    if let Some(hit) = cache.cached_label(path, mtime, line) {
        return hit;
    }
    let resolved = resolve_lines(path, &[line])?;
    let answer = resolved.get(&line).cloned().flatten();
    cache.store_labels(path, mtime, resolved);
    answer
}

/// Resolve the enclosing scope of every line in `targets`, grouped by file, and record the
/// answers on `cache`.
///
/// This is the shape that keeps peak memory flat. A rendered page asks about up to ten lines
/// (a hundred under `--full`), and answering them one at a time either re-parses a file per
/// line or — as it did before #67 — keeps every file's tree alive at once. Grouping by path
/// gives one parse per *distinct file* with exactly **one tree live at any instant**, which
/// is the best of both and needs no ceiling to tune.
///
/// Best-effort by design: an unreadable, oversized or unparseable file simply contributes no
/// answers, and the formatter renders those matches without an `in …` suffix, exactly as it
/// did when `get_or_parse` returned `None`.
pub(crate) fn warm_labels<'a>(
    targets: impl IntoIterator<Item = (&'a Path, u32)>,
    cache: &OutlineCache,
) {
    let mut by_file: std::collections::HashMap<&Path, Vec<u32>> = std::collections::HashMap::new();
    for (path, line) in targets {
        if line == 0 {
            continue;
        }
        by_file.entry(path).or_default().push(line);
    }
    for (path, mut lines) in by_file {
        lines.sort_unstable();
        lines.dedup();
        let Some(mtime) = file_mtime(path) else {
            continue;
        };
        // Only the lines this page still needs — a file already answered for these lines
        // costs nothing.
        let missing: Vec<u32> = lines
            .into_iter()
            .filter(|l| cache.cached_label(path, mtime, *l).is_none())
            .collect();
        if missing.is_empty() {
            continue;
        }
        if let Some(resolved) = resolve_lines(path, &missing) {
            cache.store_labels(path, mtime, resolved);
        }
    }
}

/// Parse `path` once and answer every line in `lines`. The tree is dropped when this
/// returns; nothing it allocates outlives the call.
///
/// `None` means the file could not be parsed at all — distinct from a parsed file where a
/// line has no enclosing definition, which is `Some(map)` with a `None` value for that line.
/// The difference matters: the second is an answer worth caching, the first is not.
fn resolve_lines(
    path: &Path,
    lines: &[u32],
) -> Option<std::collections::HashMap<u32, Option<ScopeLabel>>> {
    let meta = std::fs::metadata(path).ok()?;
    if meta.len() > OutlineCache::max_parse_bytes() {
        return None;
    }
    let crate::types::FileType::Code(lang) = crate::lang::detect_file_type(path) else {
        return None;
    };
    let ts_lang = crate::lang::outline::outline_language(lang)?;
    let content = std::fs::read_to_string(path).ok()?;
    let tree = crate::lang::parse_masked(&content, Some(lang), &ts_lang)?;
    let src: Vec<&str> = content.lines().collect();
    let root = tree.root_node();

    let mut out = std::collections::HashMap::with_capacity(lines.len());
    for &line in lines {
        let row = (line - 1) as usize;
        if row >= src.len() {
            continue;
        }
        let point = tree_sitter::Point { row, column: 0 };
        let answer = root
            .descendant_for_point_range(point, point)
            .and_then(|target| walk_to_enclosing_definition(target, &src, lang))
            .map(|(def_node, name, _range)| ScopeLabel {
                kind: kind_label(def_node, &src, lang),
                name,
            });
        out.insert(line, answer);
    }
    Some(out)
}

/// A file's mtime, or `None` when it cannot be read — which also means it cannot be cached
/// against, since mtime is the whole staleness guard.
fn file_mtime(path: &Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(path).ok()?.modified().ok()
}

/// Map a tree-sitter definition node to a short user-facing label. Every kind
/// we handle is enumerated here, so adding a new language grammar is "add the
/// node kind to this match" with no string heuristics elsewhere.
fn kind_label(node: tree_sitter::Node, lines: &[&str], lang: crate::types::Lang) -> &'static str {
    // A C++ export macro in a class head misparses into a `function_definition` that
    // actually declares a class — label it for what it is, not for the node kind.
    if cpp_misparsed_class_name(node, lines).is_some() {
        return "class";
    }
    match node.kind() {
        "function_declaration"
        | "function_definition"
        | "function_item"
        | "method_definition"
        | "method_declaration"
        | "decorated_definition" => "function",
        // A C/C++ `field_declaration` is a class member; a `function_definition`
        // that actually declares a class is handled by the `class` arm below.
        "field_declaration" => "member",
        "class_declaration" | "class_definition" | "class_specifier" => "class",
        "struct_item" | "struct_specifier" => "struct",
        "union_specifier" => "union",
        "interface_declaration" => "interface",
        "trait_declaration" | "trait_item" => "trait",
        "type_alias_declaration"
        | "type_item"
        | "type_declaration"
        | "type_definition"
        | "alias_declaration" => "type",
        "enum_item" | "enum_declaration" | "enum_specifier" => "enum",
        "template_declaration" => "template",
        // A C/C++ `declaration` reaches here only as an out-of-line static member
        // definition (`int Widget::sCount = 0;`) or a member template — a
        // macro-misparsed class head is caught by the `cpp_misparsed_class_name` check
        // above. Without it the node fell through to `_ => "definition"`, rendering
        // `in definition sCount`.
        "declaration" | "lexical_declaration" | "variable_declaration" => "variable",
        "const_item" | "const_declaration" => "const",
        "static_item" => "static",
        "property_declaration" => "property",
        "mod_item" | "namespace_definition" => "module",
        "object_declaration" => "object",
        "impl_item" => "impl",
        "export_statement" => "export",
        "call" if lang == crate::types::Lang::Elixir => elixir_kind_label(node, lines),
        _ => "definition",
    }
}

/// Elixir definitions are all `call` nodes; the keyword (`def`, `defmodule`,
/// …) lives in the call's `target` field. Map it to the same vocabulary
/// `kind_label` produces for other languages.
fn elixir_kind_label(node: tree_sitter::Node, lines: &[&str]) -> &'static str {
    let Some(target) = node.child_by_field_name("target") else {
        return "definition";
    };
    match node_text_simple(target, lines).as_str() {
        "defmodule" => "module",
        "defprotocol" => "protocol",
        "defimpl" => "impl",
        "def" | "defp" | "defmacro" | "defmacrop" | "defguard" | "defguardp" | "defdelegate" => {
            "function"
        }
        "defstruct" | "defexception" => "struct",
        _ => "definition",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn write(dir: &Path, name: &str, content: &str) -> PathBuf {
        let p = dir.join(name);
        fs::write(&p, content).unwrap();
        p
    }

    #[test]
    fn enclosing_at_rust_top_level_function() {
        let tmp = tempfile::tempdir().unwrap();
        let p = write(tmp.path(), "a.rs", "fn foo() {\n    let x = 1;\n}\n");
        let cache = OutlineCache::new();
        let scope = enclosing_definition_at(&p, 2, &cache).unwrap();
        assert_eq!(scope.kind, "function");
        assert_eq!(scope.name, "foo");
    }

    #[test]
    fn enclosing_at_rust_method_inside_mod() {
        // mod_item has a name field; impl_item does not, so the qualifier path
        // exercised here is mod-name → method-name.
        let tmp = tempfile::tempdir().unwrap();
        let p = write(
            tmp.path(),
            "a.rs",
            "mod outer {\n    fn helper() {\n        let x = 1;\n    }\n}\n",
        );
        let cache = OutlineCache::new();
        let scope = enclosing_definition_at(&p, 3, &cache).unwrap();
        assert_eq!(scope.kind, "function");
        assert_eq!(scope.name, "outer.helper");
    }

    #[test]
    fn enclosing_at_typescript_method_qualifies_with_class() {
        let tmp = tempfile::tempdir().unwrap();
        let p = write(
            tmp.path(),
            "a.ts",
            "class Foo {\n  bar() {\n    const x = 1;\n  }\n}\n",
        );
        let cache = OutlineCache::new();
        let scope = enclosing_definition_at(&p, 3, &cache).unwrap();
        assert_eq!(scope.kind, "function");
        assert_eq!(scope.name, "Foo.bar");
    }

    #[test]
    fn enclosing_at_python_method_qualifies_with_class() {
        let tmp = tempfile::tempdir().unwrap();
        let p = write(
            tmp.path(),
            "a.py",
            "class Foo:\n    def bar(self):\n        x = 1\n",
        );
        let cache = OutlineCache::new();
        let scope = enclosing_definition_at(&p, 3, &cache).unwrap();
        assert_eq!(scope.kind, "function");
        assert_eq!(scope.name, "Foo.bar");
    }

    #[test]
    fn enclosing_at_elixir_def_qualifies_with_module() {
        let tmp = tempfile::tempdir().unwrap();
        let p = write(
            tmp.path(),
            "a.ex",
            "defmodule Foo do\n  def bar do\n    :ok\n  end\nend\n",
        );
        let cache = OutlineCache::new();
        let scope = enclosing_definition_at(&p, 3, &cache).unwrap();
        assert_eq!(scope.kind, "function");
        assert_eq!(scope.name, "Foo.bar");
    }

    #[test]
    fn enclosing_at_elixir_defmodule_kind_is_module() {
        let tmp = tempfile::tempdir().unwrap();
        let p = write(
            tmp.path(),
            "a.ex",
            "defmodule Foo do\n  @moduledoc \"hi\"\nend\n",
        );
        let cache = OutlineCache::new();
        let scope = enclosing_definition_at(&p, 2, &cache).unwrap();
        assert_eq!(scope.kind, "module");
        assert_eq!(scope.name, "Foo");
    }

    #[test]
    fn enclosing_at_top_level_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let p = write(tmp.path(), "a.rs", "// just a comment\nfn foo() {}\n");
        let cache = OutlineCache::new();
        assert!(enclosing_definition_at(&p, 1, &cache).is_none());
    }

    #[test]
    fn enclosing_at_zero_line_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let p = write(tmp.path(), "a.rs", "fn foo() {}\n");
        let cache = OutlineCache::new();
        assert!(enclosing_definition_at(&p, 0, &cache).is_none());
    }

    #[test]
    fn enclosing_at_non_code_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let p = write(tmp.path(), "a.md", "# heading\n\nsome text\n");
        let cache = OutlineCache::new();
        assert!(enclosing_definition_at(&p, 3, &cache).is_none());
    }

    #[test]
    fn enclosing_at_caches_parse_across_calls() {
        // Two calls into the same file should reuse the cached parse —
        // observable indirectly by mutating the file between calls without
        // touching mtime: the first parse wins, the second sees stale data
        // because the mtime didn't change. (Test only asserts the cache hit
        // path returns the first-parse result.)
        let tmp = tempfile::tempdir().unwrap();
        let p = write(tmp.path(), "a.rs", "fn foo() { let x = 1; }\n");
        let cache = OutlineCache::new();
        let a = enclosing_definition_at(&p, 1, &cache).unwrap();
        let b = enclosing_definition_at(&p, 1, &cache).unwrap();
        assert_eq!(a.name, b.name);
        assert_eq!(a.kind, b.kind);
    }

    #[test]
    fn enclosing_at_kind_labels_for_common_definition_kinds() {
        // One case per kind_label match arm beyond `function`/`module`,
        // so a regression that miscategorizes (e.g.) a Rust `struct` as
        // `definition` would surface here.
        let cases: &[(&str, &str, u32, &str, &str)] = &[
            ("a.rs", "struct Foo { x: u32 }\n", 1, "struct", "Foo"),
            ("b.rs", "enum Color { Red, Blue }\n", 1, "enum", "Color"),
            (
                "c.rs",
                "trait Greeter { fn hi(&self); }\n",
                1,
                "trait",
                "Greeter",
            ),
            (
                "d.ts",
                "interface Shape { area(): number; }\n",
                1,
                "interface",
                "Shape",
            ),
            ("e.ts", "class Widget { x = 1; }\n", 1, "class", "Widget"),
        ];
        let cache = OutlineCache::new();
        for (filename, content, line, kind, name) in cases {
            let tmp = tempfile::tempdir().unwrap();
            let p = write(tmp.path(), filename, content);
            let scope = enclosing_definition_at(&p, *line, &cache)
                .unwrap_or_else(|| panic!("no scope returned for {filename}"));
            assert_eq!(scope.kind, *kind, "kind mismatch for {filename}");
            assert_eq!(scope.name, *name, "name mismatch for {filename}");
        }
    }

    #[test]
    fn enclosing_at_rust_impl_block_does_not_qualify_with_type() {
        // tree-sitter-rust's `impl_item` exposes its type via a `type` field,
        // not via the `name`/`identifier`/`declarator` fields that
        // extract_definition_name probes. So methods inside `impl Foo {...}`
        // produce the bare function name, not `"Foo.bar"`. Pre-existing
        // behavior of find_enclosing_function — pinned here so a future
        // qualifier improvement is an intentional, visible change.
        let tmp = tempfile::tempdir().unwrap();
        let p = write(
            tmp.path(),
            "a.rs",
            "struct Foo;\nimpl Foo {\n    fn bar(&self) {\n        let x = 1;\n    }\n}\n",
        );
        let cache = OutlineCache::new();
        let scope = enclosing_definition_at(&p, 4, &cache).unwrap();
        assert_eq!(scope.kind, "function");
        assert_eq!(scope.name, "bar");
    }

    // ── #67: the grouped warm pass ──────────────────────────────────────────

    const THREE_FNS: &str =
        "fn one() {\n    let a = 1;\n}\nfn two() {\n    let b = 2;\n}\nfn three() {\n    let c = 3;\n}\n";

    /// The memory bound rests on one parse answering every line of a file, so pin the
    /// observable form of that: after a single `warm_labels`, every requested line is a
    /// cache hit. If the grouping regressed to one parse per line, the answers would still
    /// be right — this is the only assertion that would notice.
    #[test]
    fn warm_labels_answers_every_line_of_a_file_in_one_pass() {
        let tmp = tempfile::tempdir().unwrap();
        let p = write(tmp.path(), "a.rs", THREE_FNS);
        let cache = OutlineCache::new();
        let mtime = file_mtime(&p).unwrap();

        warm_labels(
            [(p.as_path(), 2u32), (p.as_path(), 5), (p.as_path(), 8)],
            &cache,
        );

        for (line, want) in [(2u32, "one"), (5, "two"), (8, "three")] {
            let hit = cache
                .cached_label(&p, mtime, line)
                .unwrap_or_else(|| panic!("line {line} was not answered by the warm pass"));
            assert_eq!(hit.expect("inside a function").name, want);
        }
    }

    /// Two entry points now resolve the same question — the batched `warm_labels` and the
    /// lazy `enclosing_definition_at`. Two paths to one answer is exactly the shape that
    /// drifts, so pin that they agree, including on the "top level, no enclosing definition"
    /// answer where one path caches `None` and the other must not read that as a miss.
    #[test]
    fn warmed_and_lazy_resolution_agree() {
        let tmp = tempfile::tempdir().unwrap();
        let p = write(tmp.path(), "a.rs", THREE_FNS);
        // Line 4 is `fn two()`'s own signature line; line 1 is `fn one()`'s. Both sit inside
        // a definition. A line at true top level needs a file with one.
        let q = write(
            tmp.path(),
            "b.rs",
            "const X: u32 = 1;\nfn only() {\n    let a = 1;\n}\n",
        );

        for (path, lines) in [(&p, vec![1u32, 2, 5, 8]), (&q, vec![1, 3])] {
            let lazy = OutlineCache::new();
            let warmed = OutlineCache::new();
            warm_labels(lines.iter().map(|l| (path.as_path(), *l)), &warmed);
            for &line in &lines {
                assert_eq!(
                    enclosing_definition_at(path, line, &lazy),
                    enclosing_definition_at(path, line, &warmed),
                    "paths disagree for {}:{line}",
                    path.display()
                );
            }
        }
    }

    /// A file rewritten between two calls must not serve answers from its old contents. The
    /// warm pass stores against the mtime it read, so this pins the guard across both the
    /// batched store and the lazy read.
    #[test]
    fn a_rewritten_file_is_not_answered_from_stale_labels() {
        let tmp = tempfile::tempdir().unwrap();
        let p = write(
            tmp.path(),
            "a.rs",
            "fn before_edit() {\n    let a = 1;\n}\n",
        );
        let cache = OutlineCache::new();
        warm_labels([(p.as_path(), 2u32)], &cache);
        assert_eq!(
            enclosing_definition_at(&p, 2, &cache).unwrap().name,
            "before_edit"
        );

        // Rewrite with a different function name. The sleep is what makes the mtime differ
        // on filesystems with coarse timestamps; the assertion below makes the test fail
        // loudly rather than pass vacuously if it ever does not.
        let before = file_mtime(&p).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        fs::write(&p, "fn after_edit() {\n    let a = 1;\n}\n").unwrap();
        assert_ne!(
            file_mtime(&p).unwrap(),
            before,
            "the rewrite did not change the mtime, so this test would prove nothing"
        );

        assert_eq!(
            enclosing_definition_at(&p, 2, &cache).unwrap().name,
            "after_edit",
            "the stale answer survived the rewrite"
        );
    }

    /// Files the resolver cannot handle must stay silent rather than poisoning the cache
    /// with a wrong answer — the formatter renders those matches with no `in …` suffix, the
    /// same as when the old `get_or_parse` returned `None`.
    #[test]
    fn unparseable_and_missing_files_contribute_no_answers() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = OutlineCache::new();
        let missing = tmp.path().join("nope.rs");
        let not_code = write(
            tmp.path(),
            "a.txt",
            "fn looks_like_code() {\n    let a = 1;\n}\n",
        );

        warm_labels([(missing.as_path(), 2u32), (not_code.as_path(), 2)], &cache);
        assert_eq!(enclosing_definition_at(&missing, 2, &cache), None);
        assert_eq!(enclosing_definition_at(&not_code, 2, &cache), None);
    }
}
