use crate::lang::treesitter::{
    c_declarator_name, c_declarators, cpp_misparsed_class_name, declarator_chain_has_function,
    declarator_declares_function, enclosing_misparsed_class_name, is_bodied_specifier,
    is_cpp_macro_invocation, is_named_bodied_specifier, misparsed_member_name,
    multi_declarator_names, node_text_simple, SPECIFIER_KINDS,
};
use crate::types::{Lang, OutlineEntry, OutlineKind};

/// Get the tree-sitter Language for a given Lang variant.
pub fn outline_language(lang: Lang) -> Option<tree_sitter::Language> {
    let lang = match lang {
        Lang::Rust => tree_sitter_rust::LANGUAGE,
        Lang::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT,
        Lang::Tsx => tree_sitter_typescript::LANGUAGE_TSX,
        Lang::JavaScript => tree_sitter_javascript::LANGUAGE,
        Lang::Python => tree_sitter_python::LANGUAGE,
        Lang::Scala => tree_sitter_scala::LANGUAGE,
        Lang::Go => tree_sitter_go::LANGUAGE,
        Lang::Java => tree_sitter_java::LANGUAGE,
        Lang::C => tree_sitter_c::LANGUAGE,
        Lang::Cpp => tree_sitter_cpp::LANGUAGE,
        Lang::Ruby => tree_sitter_ruby::LANGUAGE,
        Lang::Php => tree_sitter_php::LANGUAGE_PHP,
        // Languages without shipped grammars — fall back
        Lang::CSharp => tree_sitter_c_sharp::LANGUAGE,
        Lang::Swift => tree_sitter_swift::LANGUAGE,
        Lang::Kotlin => tree_sitter_kotlin_ng::LANGUAGE,
        Lang::Elixir => tree_sitter_elixir::LANGUAGE,
        Lang::Bash => tree_sitter_bash::LANGUAGE,
        Lang::Dockerfile | Lang::Make => {
            return None;
        }
    };
    Some(lang.into())
}

/// Parse markdown content into a tree-sitter block tree.
///
/// Returns `None` if the parser fails to set the language (should not happen
/// in practice). The block grammar is what tilth's outline / definition
/// scanners need: it emits `atx_heading`, `setext_heading`, `section`, and
/// `fenced_code_block` nodes. Inline structure (emphasis, links inside the
/// heading text) is parsed by a separate inline grammar tilth doesn't use —
/// heading text is read as the raw inline node's text.
///
/// Centralised so both `read::outline::markdown` and
/// `search::symbol::stream_defs_markdown` configure the parser the same
/// way.
pub fn parse_markdown(content: &str) -> Option<tree_sitter::Tree> {
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&tree_sitter_md::LANGUAGE.into()).ok()?;
    parser.parse(content, None)
}

/// Map an `atx_heading` or `setext_heading` node to its 1-6 level by
/// inspecting the marker child. Returns `None` for malformed nodes.
pub fn heading_level(node: tree_sitter::Node) -> Option<u8> {
    let kind = node.kind();
    if kind == "atx_heading" {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "atx_h1_marker" => return Some(1),
                "atx_h2_marker" => return Some(2),
                "atx_h3_marker" => return Some(3),
                "atx_h4_marker" => return Some(4),
                "atx_h5_marker" => return Some(5),
                "atx_h6_marker" => return Some(6),
                _ => {}
            }
        }
        None
    } else if kind == "setext_heading" {
        // setext H1: `=====`; H2: `-----`. Marker is a child node.
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "setext_h1_underline" => return Some(1),
                "setext_h2_underline" => return Some(2),
                _ => {}
            }
        }
        None
    } else {
        None
    }
}

/// Read the heading text of an `atx_heading` / `setext_heading` node from
/// pre-split source lines. Returns the inline content with surrounding
/// whitespace + trailing `#`s (for ATX-closed headings like `## Foo ##`)
/// trimmed, matching the previous hand-rolled scanner's output.
pub fn heading_text(node: tree_sitter::Node, lines: &[&str]) -> String {
    // Both heading kinds expose their inline content as an `inline` child.
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "inline" {
            let text = node_text_simple(child, lines);
            return text.trim().trim_end_matches('#').trim().to_string();
        }
    }
    String::new()
}

/// Walk top-level children of the root node, extracting outline entries.
pub(crate) fn walk_top_level(
    root: tree_sitter::Node,
    lines: &[&str],
    lang: Lang,
) -> Vec<OutlineEntry> {
    let mut entries = Vec::new();
    let mut cursor = root.walk();

    for child in root.children(&mut cursor) {
        // C/C++ conditional-compilation blocks are transparent. Platform guards are
        // the dominant use of `#ifdef` in headers, so anything inside one — includes
        // especially, but declarations too — was invisible in the outline while
        // `tilth_deps` (which reads lines, not the AST) reported it, an outline/deps
        // disagreement. Both branches of an `#if`/`#else` are surfaced: both exist in
        // the source, and tilth does not evaluate the preprocessor.
        if matches!(lang, Lang::C | Lang::Cpp) && is_preproc_conditional(child.kind()) {
            entries.extend(walk_top_level(child, lines, lang));
            continue;
        }
        entries.extend(node_to_entries(child, lines, lang, 0));
    }

    entries
}

/// True for the C/C++ conditional-compilation wrappers that hold ordinary declarations.
fn is_preproc_conditional(kind: &str) -> bool {
    matches!(
        kind,
        "preproc_if" | "preproc_ifdef" | "preproc_else" | "preproc_elif" | "preproc_elifdef"
    )
}

/// `node_to_entry`, plus one entry per *additional* declarator the node introduces.
///
/// One C/C++ declaration can name several things — `int mWidth, mHeight;`,
/// `typedef int A, B, C;` — and `node_to_entry` returns a single `OutlineEntry` by
/// construction, so every name after the first was missing from the outline.
///
/// A wrapper rather than a new return type for `node_to_entry` itself: that function has
/// eight call sites, and the two that *collect* entries — `walk_top_level` and
/// `collect_member` — are wired here. The other four return a single inner entry from a
/// wrapper (`template_declaration`, `export_statement`, a `field_declaration` or
/// `declaration` whose `type` is a bodied specifier), and by definition can only pass one
/// entry upward. `template <class T> int a, b;` therefore still outlines `a` alone; it is
/// ill-formed C++ and left as-is rather than reshaping the wrapper contract.
///
/// The name list comes from `multi_declarator_names`, which is also what the search side
/// uses — one implementation of "which names does this declaration introduce", so the two
/// surfaces cannot answer it differently. It returns `None`, and this returns the single
/// entry unchanged, whenever the primary name did not come from the first declarator.
/// `struct Config { int x; } gConfigA, gConfigB;` is why: `node_to_entry` renders the
/// *type* there, so appending "the declarators after the first" produced `gConfigB` and
/// silently not `gConfigA` — worse than the previous behaviour, which emitted neither.
fn node_to_entries(
    node: tree_sitter::Node,
    lines: &[&str],
    lang: Lang,
    depth: usize,
) -> Vec<OutlineEntry> {
    let Some(first) = node_to_entry(node, lines, lang, depth) else {
        return Vec::new();
    };
    if !matches!(lang, Lang::C | Lang::Cpp) {
        return vec![first];
    }
    let Some(names) = multi_declarator_names(node, lines, &first.name) else {
        return vec![first];
    };
    let declarators = c_declarators(node);
    let mut out = Vec::with_capacity(names.len());
    // Skip the first name: `node_to_entry` already rendered it, with all the special-casing
    // its arm carries (macro invocations, nested types, misparsed class bodies).
    for name in names.into_iter().skip(1) {
        // Pair the name back to its declarator so the kind is decided per declarator. The
        // lookup is by resolved name because `multi_declarator_names` may have deduplicated.
        let kind = declarators
            .iter()
            .find(|d| c_declarator_name(**d, lines).as_deref() == Some(name.as_str()))
            .map_or(OutlineKind::Variable, |d| declarator_kind(node, *d, lines));
        out.push(OutlineEntry {
            kind,
            name,
            start_line: node.start_position().row as u32 + 1,
            end_line: node.end_position().row as u32 + 1,
            // Extras carry no signature or doc. `int x, f();` therefore renders `fn f`
            // without the signature line that `int f();` alone would get — the signature is
            // extracted from the whole node and would repeat the entire declaration under
            // each name.
            signature: None,
            doc: None,
            children: Vec::new(),
        });
    }
    out.insert(0, first);
    out
}

/// The outline kind for one declarator of a multi-declarator C/C++ declaration.
///
/// Mirrors the classification its node's own arm applies, but asks the *declarator* rather
/// than the declaration — the two differ for `int f(), x;`, where one node declares both a
/// function and a variable, and asking the node would label both the same.
fn declarator_kind(
    node: tree_sitter::Node,
    declarator: tree_sitter::Node,
    lines: &[&str],
) -> OutlineKind {
    // `type_definition` is decided by the node, not the declarator, and is checked first for
    // that reason: `typedef void (*CbA)(int), (*CbB)(int);` has a `function_declarator` for
    // each alias, but both are type aliases. Testing function-ness first rendered the first
    // as `type CbA` (from `node_to_entry`'s arm) and the second as `fn CbB` — the two halves
    // of one declaration disagreeing, which is the whole defect class this fix is in.
    if node.kind() == "type_definition" {
        return OutlineKind::TypeAlias;
    }
    if declarator_declares_function(declarator) {
        return OutlineKind::Function;
    }
    match node.kind() {
        "field_declaration" => OutlineKind::Property,
        // A data member of a macro-misparsed class body arrives as a `declaration`, and the
        // `declaration` arm renders it `prop` rather than `let` for parity with the same
        // class spelled without its export macro. The extras have to follow.
        _ if enclosing_misparsed_class_name(node, lines).is_some() => OutlineKind::Property,
        _ => OutlineKind::Variable,
    }
}

/// Convert a tree-sitter node to an `OutlineEntry` based on its kind.
fn node_to_entry(
    node: tree_sitter::Node,
    lines: &[&str],
    lang: Lang,
    depth: usize,
) -> Option<OutlineEntry> {
    let kind_str = node.kind();
    let start_line = node.start_position().row as u32 + 1;
    let end_line = node.end_position().row as u32 + 1;

    // Computed once: the macro-misparse check walks children, and it gates both the
    // arm below and the name that arm uses.
    let misparsed_class = cpp_misparsed_class_name(node, lines);

    let (kind, name, signature) = match kind_str {
        // C++ export-macro misparse: `class LIBFOO_API Widget : public Base { … };`
        // reaches us as a `function_definition` that actually declares a class. Must
        // precede the function arm below, which would render it `<anonymous>`.
        // See `cpp_misparsed_class_name`.
        "function_definition" | "declaration" if misparsed_class.is_some() => {
            let name = misparsed_class.unwrap_or_else(|| "<anonymous>".into());
            // The specifier still says which keyword was written, even though it is
            // named for the macro — without this a `struct` behind an export macro
            // outlined as a `class`, a parity failure of its own.
            let kind = match node.child_by_field_name("type").map(|t| t.kind()) {
                Some("struct_specifier" | "union_specifier") => OutlineKind::Struct,
                _ => OutlineKind::Class,
            };
            (kind, name, None)
        }

        // A C/C++ type specifier with no `body` is an elaborated type specifier — a
        // forward declaration (`class Fwd;`) or a type reference (`class Fwd* p;`),
        // not a definition. Reject before the class/struct/enum arms below so it
        // never reaches an outline. This also keeps the macro-misparse arm above from
        // emitting its (macro-named) bodyless specifier child as a nested class.
        k if SPECIFIER_KINDS.contains(&k) && !is_bodied_specifier(node) => {
            return None;
        }

        // Functions
        "function_declaration"
        | "function_definition"
        | "function_item"
        | "method_definition"
        | "method_declaration"
        | "constructor_declaration"
        | "init_declaration"
        | "deinit_declaration"
        | "protocol_function_declaration" => {
            let name = find_child_text(node, "name", lines)
                .or_else(|| find_child_text(node, "identifier", lines))
                // C/C++ put the name inside a declarator chain rather than a `name`
                // field, so every C and C++ function outlined as `<anonymous>`.
                .or_else(|| c_declarator_child_name(node, lines))
                .unwrap_or_else(|| {
                    // Swift deinit has no name field — use the node kind as name
                    if kind_str == "deinit_declaration" {
                        "deinit".into()
                    } else {
                        "<anonymous>".into()
                    }
                });
            let sig = extract_signature(node, lines);
            (OutlineKind::Function, name, Some(sig))
        }

        // Classes & structs
        "class_declaration" | "class_definition" | "class_specifier" => {
            let name = find_child_text(node, "name", lines)
                .or_else(|| find_child_text(node, "identifier", lines))
                .unwrap_or_else(|| "<anonymous>".into());
            (OutlineKind::Class, name, None)
        }
        "struct_item" | "struct_declaration" | "struct_specifier" | "union_specifier" => {
            let name = find_child_text(node, "name", lines).unwrap_or_else(|| "<anonymous>".into());
            (OutlineKind::Struct, name, None)
        }

        // C++ `template <typename T> class Foo` / `… void bar()`. Like
        // `export_statement`, the wrapper carries no name — recurse into the
        // declaration it encloses and render that with its real kind, widening the
        // range to cover the `template` clause.
        "template_declaration" => {
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if let Some(mut inner) = node_to_entry(child, lines, lang, depth) {
                    inner.start_line = start_line;
                    return Some(inner);
                }
            }
            return None;
        }

        // C/C++ class member: `void DoThing();` or `int Count;` inside a class body.
        // Language-gated because `field_declaration` also exists in the Rust, Go,
        // Java and C# grammars, where surfacing every struct field would change
        // those outlines. A member wrapping a nested type (`class Inner { … };`)
        // recurses so the inner type renders as itself.
        "field_declaration" if matches!(lang, Lang::C | Lang::Cpp) => {
            if let Some(type_node) = node.child_by_field_name("type") {
                // Requires a *named* specifier: an anonymous one has a body but nothing
                // to call it, so preferring it would replace a searchable identifier
                // (`anonInst`) with `<anonymous>`.
                if is_named_bodied_specifier(type_node) {
                    let mut inner = node_to_entry(type_node, lines, lang, depth)?;
                    inner.start_line = start_line;
                    return Some(inner);
                }
            }
            let name = c_declarator_child_name(node, lines)?;
            if declarator_chain_has_function(node) {
                let sig = extract_signature(node, lines);
                (OutlineKind::Function, name, Some(sig))
            } else {
                (OutlineKind::Property, name, None)
            }
        }

        // C/C++ `declaration`: a function prototype (`void Utility();`) or a global.
        // Also how a member of a macro-misparsed class body reaches us, since that
        // body parses as statements rather than as a `field_declaration_list`.
        // Language-gated — `declaration` exists in the TS, JS, Java, C# and Kotlin
        // grammars too, where surfacing it would change those outlines.
        // A macro invocation in a class body (`GENERATED_BODY()`) is shaped exactly like
        // a constructor declaration. Outlining it as a member also makes it an "exported
        // symbol" for `tilth_deps`, which then reports every other file invoking the same
        // macro as a dependent. See `is_cpp_macro_invocation`.
        "declaration"
            if matches!(lang, Lang::C | Lang::Cpp) && is_cpp_macro_invocation(node, lines) =>
        {
            return None;
        }

        "declaration" if matches!(lang, Lang::C | Lang::Cpp) => {
            // `struct S { int a; } sInstance;` declares a type *and* a variable in one
            // node. The type is the more useful entry — and it is otherwise invisible,
            // since this arm returns before the walk would reach the specifier — so
            // surface it rather than only the variable. Matches the `field_declaration`
            // arm above and keeps the outline consistent with symbol search, which
            // already finds the type here.
            if let Some(type_node) = node.child_by_field_name("type") {
                // Requires a *named* specifier: an anonymous one has a body but nothing
                // to call it, so preferring it would replace a searchable identifier
                // (`anonInst`) with `<anonymous>`.
                if is_named_bodied_specifier(type_node) {
                    let mut inner = node_to_entry(type_node, lines, lang, depth)?;
                    inner.start_line = start_line;
                    return Some(inner);
                }
            }
            let name = c_declarator_child_name(node, lines)?;
            if declarator_chain_has_function(node) {
                let sig = extract_signature(node, lines);
                (OutlineKind::Function, name, Some(sig))
            } else if enclosing_misparsed_class_name(node, lines).is_some() {
                // A data member of a misparsed class body is shaped exactly like a
                // local variable — the body is a `compound_statement`, so `int Value;`
                // is an ordinary `declaration` rather than a `field_declaration`.
                // Without this it outlines as `let Value` where the same class without
                // its export macro gives `prop Value`.
                (OutlineKind::Property, name, None)
            } else {
                (OutlineKind::Variable, name, None)
            }
        }

        // Interfaces & traits
        "interface_declaration"
        | "type_alias_declaration"
        | "trait_item"
        | "trait_declaration"
        | "trait_definition"
        | "protocol_declaration" => {
            let name = find_child_text(node, "name", lines).unwrap_or_else(|| "<anonymous>".into());
            (OutlineKind::Interface, name, None)
        }
        "type_item" | "type_definition" | "typealias_declaration" | "alias_declaration" => {
            let name = find_child_text(node, "name", lines)
                // C/C++ `typedef int MyAlias;` names the alias in a `declarator`
                // field, not a `name` field.
                .or_else(|| c_declarator_child_name(node, lines))
                .unwrap_or_else(|| "<anonymous>".into());
            (OutlineKind::TypeAlias, name, None)
        }

        // Enums
        "enum_item" | "enum_declaration" | "enum_definition" | "enum_specifier" => {
            let name = find_child_text(node, "name", lines).unwrap_or_else(|| "<anonymous>".into());
            (OutlineKind::Enum, name, None)
        }

        // Impl blocks (Rust)
        "impl_item" => {
            let name = find_child_text(node, "type", lines).unwrap_or_else(|| "<impl>".into());
            (OutlineKind::Module, format!("impl {name}"), None)
        }

        // Objects (Scala companion objects, singletons; Kotlin object declarations)
        "object_declaration" | "object_definition" => {
            let name = find_child_text(node, "name", lines)
                .or_else(|| find_child_text(node, "identifier", lines))
                .unwrap_or_else(|| "<anonymous>".into());
            (OutlineKind::Module, name, None)
        }

        // Constants and variables
        "const_item" | "const_declaration" | "static_item" => {
            let name = find_child_text(node, "name", lines)
                .or_else(|| first_identifier_text(node, lines))
                .unwrap_or_else(|| "<const>".into());
            (OutlineKind::Constant, name, None)
        }
        "val_definition" => {
            let name = first_identifier_text(node, lines).unwrap_or_else(|| "<val>".into());
            (OutlineKind::ImmutableVariable, name, None)
        }
        "lexical_declaration" | "variable_declaration" | "var_definition" => {
            let name = first_identifier_text(node, lines).unwrap_or_else(|| "<var>".into());
            (OutlineKind::Variable, name, None)
        }

        // Properties (C#, Swift, Kotlin)
        "property_declaration" | "protocol_property_declaration" => {
            let name = find_child_text(node, "name", lines)
                .or_else(|| first_identifier_text(node, lines))
                .unwrap_or_else(|| "<property>".into());
            let sig = extract_signature(node, lines);
            (OutlineKind::Property, name, Some(sig))
        }

        // Imports — collect as a group.
        //
        // `preproc_include` is the C/C++ `#include`. It was missing, so a C or C++
        // outline showed a header's types but never what it included — even though
        // `extract_import_source` already knew how to parse the directive (the deps
        // tool reads it line-by-line rather than from the AST). The `<…>` / `"…"`
        // delimiters survive into the rendered group on purpose: they are what
        // distinguishes a system header from a project-relative one.
        "import_statement"
        | "import_declaration"
        | "import"
        | "use_declaration"
        | "namespace_use_declaration"
        | "use_item"
        | "using_directive"
        | "preproc_include" => {
            let text = node_text(node, lines);
            (OutlineKind::Import, text, None)
        }

        // Exports — `export` is a modifier on a wrapped declaration, not a
        // peer of `function`/`class`/`const`. Recurse into the inner
        // declaration so the entry renders with its real kind. Falling back to
        // `OutlineKind::Export` only when there is no nameable declaration
        // inside (`export { … }`, `export * from …`, `export default <expr>`).
        // Without this, `export_statement`'s `name` is the full source span
        // (already starts with `export `), and the renderer prepends the
        // `Export` kind_label `"export"` again — producing the doubled-keyword
        // outline header `export export async function foo(`.
        "export_statement" => {
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if let Some(mut inner) = node_to_entry(child, lines, lang, depth) {
                    // Extend the entry's range to cover the `export` keyword
                    // so the outline byte range still points at the statement.
                    inner.start_line = start_line;
                    return Some(inner);
                }
            }
            // No nameable inner declaration; strip leading `export ` so the
            // rendered name doesn't duplicate the `kind_label`.
            let raw = node_text(node, lines);
            let name = raw
                .strip_prefix("export ")
                .map(str::to_string)
                .unwrap_or(raw);
            (OutlineKind::Export, name, None)
        }

        // Module declarations
        "mod_item"
        | "module"
        | "namespace_declaration"
        | "namespace_definition"
        | "file_scoped_namespace_declaration" => {
            let name = find_child_text(node, "name", lines).unwrap_or_else(|| "<module>".into());
            (OutlineKind::Module, name, None)
        }

        // Elixir: all definitions are `call` nodes distinguished by target identifier
        "call" if lang == Lang::Elixir => {
            return elixir_call_to_entry(node, lines, lang, depth);
        }

        // Elixir: @type, @typep, @opaque are unary_operator nodes
        "unary_operator" if lang == Lang::Elixir => {
            return elixir_attr_to_entry(node, lines);
        }

        // Bash: top-level variable assignments (`MY_VAR=value`, `ARR[0]=value`)
        "variable_assignment" if lang == Lang::Bash => {
            let name = assignment_name(node, lines).unwrap_or_else(|| "<var>".into());
            (OutlineKind::Variable, name, None)
        }

        // Bash: top-level `export` / `declare` / `readonly` declarations. The name
        // is the `name` of the inner variable_assignment (`export FOO=bar`) or a
        // bare variable_name child (`export FOO`). Function-local `local`
        // declarations are nested in function bodies, so walk_top_level never
        // reaches them here. Multi-variable declarations surface their first name.
        "declaration_command" if lang == Lang::Bash => {
            let mut cursor = node.walk();
            let name = node
                .children(&mut cursor)
                .find_map(|child| match child.kind() {
                    "variable_assignment" => assignment_name(child, lines),
                    "variable_name" => Some(node_text(child, lines)),
                    _ => None,
                })?;
            (OutlineKind::Variable, name, None)
        }

        _ => return None,
    };

    // Collect children for classes, impls, modules, traits/interfaces
    let is_namespace = matches!(
        kind_str,
        "namespace_declaration" | "namespace_definition" | "file_scoped_namespace_declaration"
    );
    let children = if matches!(
        kind,
        OutlineKind::Class | OutlineKind::Struct | OutlineKind::Module | OutlineKind::Interface
    ) && depth < 1
    {
        // Namespaces are transparent wrappers — don't consume a depth level,
        // so classes inside namespaces still collect their methods.
        let child_depth = if is_namespace { depth } else { depth + 1 };
        collect_children(node, lines, lang, child_depth)
    } else {
        Vec::new()
    };

    // Extract doc comment if present
    let doc = extract_doc(node, lines);

    Some(OutlineEntry {
        kind,
        name,
        start_line,
        end_line,
        signature,
        children,
        doc,
    })
}

/// Collect child entries from a class/struct/impl body.
fn collect_children(
    node: tree_sitter::Node,
    lines: &[&str],
    lang: Lang,
    depth: usize,
) -> Vec<OutlineEntry> {
    let mut children = Vec::new();
    let mut cursor = node.walk();

    // Look for a body node first (C# uses `declaration_list` instead of `*_body`/`*_block`;
    // C/C++ class and struct bodies are a `field_declaration_list`).
    //
    // `compound_statement` is C/C++-only here: it is where a macro-misparsed class
    // head keeps its members. It must stay gated, because a *braced* PHP namespace
    // (`namespace App { … }`) is a Module entry whose body is also a
    // `compound_statement` — matching it ungated pulled PHP namespace members into
    // the outline with placeholder names and a trait mislabelled as an interface.
    let cpp_family = matches!(lang, Lang::C | Lang::Cpp);
    let body = node.children(&mut cursor).find(|c| {
        let k = c.kind();
        k.contains("body")
            || k.contains("block")
            || k == "declaration_list"
            || k == "field_declaration_list"
            || (cpp_family && k == "compound_statement")
    });

    // Constructors and destructors of a misparsed class survive only as recovery
    // artifacts, and telling one from a macro invocation needs the class's real name.
    let misparsed_class = if cpp_family {
        cpp_misparsed_class_name(node, lines)
    } else {
        None
    };

    let parent = body.unwrap_or(node);
    let mut cursor2 = parent.walk();

    for child in parent.children(&mut cursor2) {
        collect_member(
            child,
            lines,
            lang,
            depth,
            misparsed_class.as_deref(),
            &mut children,
        );
    }

    children
}

/// Emit the outline entries for one node in a class/impl/module body.
///
/// Most members map one-to-one through `node_to_entry`. Two shapes do not, both from
/// C/C++ class bodies: an access specifier wraps the members that follow it, and a
/// misparsed body can pack several members into a single `ERROR` — so this pushes
/// into a vector rather than returning one entry.
fn collect_member(
    child: tree_sitter::Node,
    lines: &[&str],
    lang: Lang,
    depth: usize,
    misparsed_class: Option<&str>,
    out: &mut Vec<OutlineEntry>,
) {
    // `public:` / `private:` inside a misparsed class body parse as a
    // `labeled_statement` wrapping the members that follow it. Flatten it so those
    // members are collected rather than hidden behind the access specifier — one
    // labeled_statement can hold several of them.
    if child.kind() == "labeled_statement" && matches!(lang, Lang::C | Lang::Cpp) {
        let mut inner = child.walk();
        for grandchild in child.children(&mut inner) {
            collect_member(grandchild, lines, lang, depth, misparsed_class, out);
        }
        return;
    }
    if let Some(class) = misparsed_class {
        // A brace body loose in a class body belongs to the member just recovered:
        // recovery splits an inline-bodied constructor into a head it cannot read and
        // a `compound_statement` sibling. Without this the entry's range is the head
        // alone, and `blast_radius`' signature window — `start_line ..= start_line+3`,
        // clamped to `end_line` — collapses to a single line, so edits inside the
        // constructor find no callers where the same class unmisparsed does.
        if child.kind() == "compound_statement" {
            if let Some(last) = out.last_mut() {
                let body_end = child.end_position().row as u32 + 1;
                if last.kind == OutlineKind::Function && last.end_line < body_end {
                    last.end_line = body_end;
                }
            }
            return;
        }
        if push_misparsed_members(child, lines, class, out) {
            return;
        }
    }
    out.extend(node_to_entries(child, lines, lang, depth));
    // A nested type definition consumes its whole `declaration`, so a member declared
    // right after it (`class Inner { … }; Outer();`) is never reached. Emitted after
    // the type rather than before it, so the outline keeps source order.
    if let Some(class) = misparsed_class {
        push_members_after_nested_type(child, lines, class, out);
    }
}

/// Recursion cap for the misparsed-body walk, matching `MAX_DECLARATOR_DEPTH`'s
/// reasoning: `outline::generate` is a fuzz target, and while the deepest `ERROR`
/// nesting observed in practice is two, nothing in error recovery promises a bound.
const MAX_MISPARSE_DEPTH: usize = 16;

/// Push any constructors and destructors hidden inside a recovery artifact.
///
/// Returns true when `node` is such an artifact and has been consumed — including
/// when it yields nothing, since neither an `ERROR` nor an `expression_statement` is
/// ever a member in its own right.
///
/// One `ERROR` can cover several members (`explicit Widget(int); ~Widget();` repairs
/// into a single one), so this recurses and can push more than one entry. Each entry
/// takes its range from the node that actually matched, not from the enclosing
/// `ERROR`, which may span the whole run.
fn push_misparsed_members(
    node: tree_sitter::Node,
    lines: &[&str],
    class: &str,
    out: &mut Vec<OutlineEntry>,
) -> bool {
    push_misparsed_members_at(node, lines, class, out, 0)
}

fn push_misparsed_members_at(
    node: tree_sitter::Node,
    lines: &[&str],
    class: &str,
    out: &mut Vec<OutlineEntry>,
    depth: usize,
) -> bool {
    if depth > MAX_MISPARSE_DEPTH {
        return false;
    }
    match node.kind() {
        // `Widget();` — the statement, not the call inside it, so the entry's range
        // covers the trailing `;` the same way the unmisparsed `declaration` does.
        "expression_statement" => {
            if let Some(name) = statement_member_name(node, lines, class) {
                out.push(misparsed_entry(node, name, lines));
            }
            true
        }
        // `constexpr Widget();` — a qualifier keeps recovery from reading a call and
        // leaves a `declaration` whose `type` is the class name and whose declarator
        // is zero-width, which the ordinary declaration arm cannot name.
        "declaration" => {
            if let Some(name) = qualified_ctor_name(node, lines, class) {
                out.push(misparsed_entry(node, name, lines));
                return true;
            }
            false
        }
        "ERROR" => {
            let mut cursor = node.walk();
            let children: Vec<tree_sitter::Node> = node.children(&mut cursor).collect();
            for child in children {
                match misparsed_member_name(child, lines, class) {
                    Some(name) => out.push(misparsed_entry(child, name, lines)),
                    None => {
                        push_misparsed_members_at(child, lines, class, out, depth + 1);
                    }
                }
            }
            true
        }
        _ => false,
    }
}

/// The member an `expression_statement` in a misparsed class body declares.
///
/// `Widget() = default;` and `Widget(const Widget&) = delete;` — the two most common
/// ways a modern C++ class spells a constructor — wrap the call in an
/// `assignment_expression`, so the call is a *grandchild*. Recovery only produces
/// that when the member is the first after an access specifier; anywhere else the
/// same source parses as a `function_definition` and never reaches here, which is why
/// the gap was position-dependent.
fn statement_member_name(node: tree_sitter::Node, lines: &[&str], class: &str) -> Option<String> {
    let mut cursor = node.walk();
    let children: Vec<tree_sitter::Node> = node.children(&mut cursor).collect();
    for child in children {
        if let Some(name) = misparsed_member_name(child, lines, class) {
            return Some(name);
        }
        if child.kind() == "assignment_expression" {
            let mut inner = child.walk();
            let nested: Vec<tree_sitter::Node> = child.children(&mut inner).collect();
            for grandchild in nested {
                if let Some(name) = misparsed_member_name(grandchild, lines, class) {
                    return Some(name);
                }
            }
        }
    }
    None
}

/// Name of the constructor a qualifier-prefixed `declaration` declares, if any.
///
/// `constexpr Widget();` leaves the class name in the `type` field and a zero-width
/// identifier as the declarator, so `c_declarator_name` yields nothing. Requiring
/// that emptiness is what keeps an ordinary member of the class's own type
/// (`Widget Other;`) — same `type`, real declarator — from being read as one.
fn qualified_ctor_name(node: tree_sitter::Node, lines: &[&str], class: &str) -> Option<String> {
    let type_node = node.child_by_field_name("type")?;
    if type_node.kind() != "type_identifier" || node_text_simple(type_node, lines) != class {
        return None;
    }
    let named = node
        .child_by_field_name("declarator")
        .and_then(|d| c_declarator_name(d, lines));
    named.is_none().then(|| class.to_string())
}

/// Push a member declared immediately after a nested type inside a misparsed body.
///
/// `class Inner { … }; Outer();` repairs into a *single* `declaration` holding the
/// nested type and the constructor's declarator, and the declaration arm returns the
/// nested type early — so the constructor is never looked at. Gated on the nested
/// type to make double-emission impossible: a declaration that arm handles any other
/// way never reaches here.
fn push_members_after_nested_type(
    node: tree_sitter::Node,
    lines: &[&str],
    class: &str,
    out: &mut Vec<OutlineEntry>,
) {
    if node.kind() != "declaration" {
        return;
    }
    let Some(type_node) = node.child_by_field_name("type") else {
        return;
    };
    if !is_named_bodied_specifier(type_node) {
        return;
    }
    let mut cursor = node.walk();
    let children: Vec<tree_sitter::Node> = node.children(&mut cursor).collect();
    for child in children {
        if let Some(name) = misparsed_member_name(child, lines, class) {
            out.push(misparsed_entry(child, name, lines));
        }
    }
}

/// Build the outline entry for a constructor or destructor recovered from a
/// misparsed class body, matching what the same member yields when the class parses.
fn misparsed_entry(node: tree_sitter::Node, name: String, lines: &[&str]) -> OutlineEntry {
    OutlineEntry {
        kind: OutlineKind::Function,
        name,
        start_line: node.start_position().row as u32 + 1,
        end_line: node.end_position().row as u32 + 1,
        signature: Some(extract_signature(node, lines)),
        children: Vec::new(),
        doc: misparsed_doc(node, lines),
    }
}

/// Doc comment for a recovered member, looking through the `ERROR`s wrapping it.
///
/// `extract_doc` reads the previous *sibling*, but a destructor is matched at the
/// `function_declarator` inside an `ERROR` while its comment is a sibling of that
/// `ERROR` — one level further out. Only `ERROR` is stepped through, so this cannot
/// reach past the artifact and attach an unrelated comment.
fn misparsed_doc(node: tree_sitter::Node, lines: &[&str]) -> Option<String> {
    let mut cur = node;
    for _ in 0..MAX_MISPARSE_DEPTH {
        if let Some(doc) = extract_doc(cur, lines) {
            return Some(doc);
        }
        if cur.prev_sibling().is_some() {
            return None;
        }
        let parent = cur.parent()?;
        if parent.kind() != "ERROR" {
            return None;
        }
        cur = parent;
    }
    None
}

/// Extract the first line as a function signature (name + params + return type).
fn extract_signature(node: tree_sitter::Node, lines: &[&str]) -> String {
    let start_row = node.start_position().row;
    if start_row < lines.len() {
        // BOM-aware, because `str::trim` is not. Signatures are rendered text: they reach
        // `tilth_diff`'s `[~:sig]` lines (via `diff::overlay`), `ResolvedCallee.signature` in
        // search's `-- calls --` block and grok's `## callees`. A definition on line 1 of a
        // BOM'd file therefore printed the glyph inside its own signature — on both sides of
        // a diff.
        //
        // Belt-and-braces since #88: both routes into `walk_top_level` now hand it stripped
        // content — `get_outline_entries` strips (which is what `diff::overlay` reaches), and
        // `read::outline::generate` has since #42 — so no live caller can put a BOM in front
        // of this. Kept for the same reason `search::rank`'s strips are: the two callers are
        // the invariant, not this line, and a third added later would reintroduce the glyph
        // silently.
        let line = crate::lang::outline::trim_start_bom_aware(lines[start_row]).trim_end();
        // Truncate at opening brace
        if let Some(pos) = line.find('{') {
            return line[..pos].trim().to_string();
        }
        if line.ends_with(':') {
            // Python — truncate at trailing colon (for `def foo(x: int):` etc.)
            if let Some(pos) = line.rfind(':') {
                return line[..pos].trim().to_string();
            }
        }
        // Elixir — truncate at ` do` (block form) or `, do:` (keyword form).
        // Safe for other languages: C/Java/Go/Rust hit the `{` branch above,
        // Python hits the `:` branch. Only Elixir uses ` do` as a block delimiter.
        if let Some(pos) = line.rfind(" do") {
            let after = &line[pos + 3..];
            if after.is_empty() || after.starts_with('\n') {
                return line[..pos].trim().to_string();
            }
        }
        if let Some(pos) = line.find(", do:") {
            return line[..pos].trim().to_string();
        }
        // Full first line, truncated
        if line.len() > 120 {
            format!("{}...", crate::types::truncate_str(line, 117))
        } else {
            line.to_string()
        }
    } else {
        String::new()
    }
}

/// Find a named child and return its text.
fn find_child_text(node: tree_sitter::Node, field: &str, lines: &[&str]) -> Option<String> {
    node.child_by_field_name(field).map(|n| node_text(n, lines))
}

/// Resolve the name a C/C++ node declares through its `declarator` chain.
/// `void Holder::Work() {}` → `"Work"`, `typedef int Alias;` → `"Alias"`.
/// Returns `None` for nodes with no `declarator` field, so grammars that name
/// their declarations with a `name` field are unaffected.
fn c_declarator_child_name(node: tree_sitter::Node, lines: &[&str]) -> Option<String> {
    c_declarator_name(node.child_by_field_name("declarator")?, lines)
}

/// Resolve the variable name from an assignment `name` field, unwrapping a
/// `subscript` (`ARR[0]=x`) to its base `variable_name` so the symbol
/// surfaces as `ARR`, not `ARR[0]`.
fn assignment_name(node: tree_sitter::Node, lines: &[&str]) -> Option<String> {
    let name = node.child_by_field_name("name")?;
    if name.kind() == "subscript" {
        let mut cursor = name.walk();
        let base = name
            .children(&mut cursor)
            .find(|c| c.kind() == "variable_name")
            .unwrap_or(name);
        Some(node_text(base, lines))
    } else {
        Some(node_text(name, lines))
    }
}

/// Get the text of a node, truncated to the first line.
fn node_text(node: tree_sitter::Node, lines: &[&str]) -> String {
    let row = node.start_position().row;
    let col_start = node.start_position().column;
    let end_row = node.end_position().row;

    if row < lines.len() {
        if row == end_row {
            let col_end = node.end_position().column.min(lines[row].len());
            lines[row][col_start..col_end].to_string()
        } else {
            // Multi-line — take first line only, truncated
            let text = &lines[row][col_start..];
            if text.len() > 80 {
                format!("{}...", crate::types::truncate_str(text, 77))
            } else {
                text.to_string()
            }
        }
    } else {
        String::new()
    }
}

/// Find the first identifier-like child.
/// Recurses one level through declarators and `variable_declaration` nodes to find
/// the actual identifier inside wrapper nodes (e.g. Kotlin `property_declaration`
/// → `variable_declaration` → `simple_identifier`).
fn first_identifier_text(node: tree_sitter::Node, lines: &[&str]) -> Option<String> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        let kind = child.kind();
        if kind.contains("identifier") || kind.contains("name") {
            let text = node_text(child, lines);
            if !text.is_empty() {
                return Some(text);
            }
        }
        // Recurse one level through wrapper nodes (variable_declarator, variable_declaration)
        if kind.contains("declarator") || kind.contains("declaration") {
            let mut inner = child.walk();
            for grandchild in child.children(&mut inner) {
                if grandchild.kind().contains("identifier") {
                    let text = node_text(grandchild, lines);
                    if !text.is_empty() {
                        return Some(text);
                    }
                }
            }
        }
    }
    None
}

/// Extract a doc comment from the previous sibling.
fn extract_doc(node: tree_sitter::Node, lines: &[&str]) -> Option<String> {
    let prev = node.prev_sibling()?;
    let kind = prev.kind();
    if kind.contains("comment") || kind.contains("doc") {
        let text = node_text(prev, lines);
        // Order matters: the longer markers must be tried before `//`, and `//` has to
        // be stripped at all — `format_entry` re-prefixes the doc with `// `, so a
        // plain `// comment` rendered as `class Widget  // // comment`. C and C++ use
        // `//` far more than `///`, so the doubling became the common case once C/C++
        // entries started carrying names (and therefore docs) at all.
        let trimmed = text
            .trim_start_matches("///")
            .trim_start_matches("//!")
            .trim_start_matches("/**")
            .trim_start_matches("//")
            .trim_start_matches('#')
            .trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Elixir-specific outline helpers
// ---------------------------------------------------------------------------

/// Elixir function-like definition keywords that produce `OutlineKind::Function`.
/// This is the subset of definition keywords handled uniformly (extract function
/// name from arguments). Container keywords (`defmodule`, `defprotocol`, `defimpl`,
/// `defstruct`, `defexception`) have their own match arms in `elixir_call_to_entry`.
/// See also `ELIXIR_DEFINITION_TARGETS` in `treesitter.rs` for the complete set.
const ELIXIR_DEF_KEYWORDS: &[&str] = &[
    "def",
    "defp",
    "defmacro",
    "defmacrop",
    "defguard",
    "defguardp",
    "defdelegate",
];

/// Convert an Elixir `call` node to an outline entry.
///
/// In the Elixir tree-sitter grammar, `defmodule`, `def`, `defp`, `defstruct`,
/// etc. are all `call` nodes whose `target` field is an identifier like `"def"`.
fn elixir_call_to_entry(
    node: tree_sitter::Node,
    lines: &[&str],
    lang: Lang,
    depth: usize,
) -> Option<OutlineEntry> {
    let target = node.child_by_field_name("target")?;
    let keyword = node_text(target, lines);
    let start_line = node.start_position().row as u32 + 1;
    let end_line = node.end_position().row as u32 + 1;

    let (kind, name, signature) = match keyword.as_str() {
        "defmodule" => {
            let name = elixir_first_arg_text(node, lines)?;
            (OutlineKind::Module, name, None)
        }
        kw if ELIXIR_DEF_KEYWORDS.contains(&kw) => {
            let name = elixir_func_name(node, lines)?;
            let sig = extract_signature(node, lines);
            (OutlineKind::Function, name, Some(sig))
        }
        "defstruct" | "defexception" => (OutlineKind::Struct, keyword.clone(), None),
        "defprotocol" => {
            let name = elixir_first_arg_text(node, lines)?;
            (OutlineKind::Interface, name, None)
        }
        "defimpl" => {
            let name = elixir_first_arg_text(node, lines)?;
            (OutlineKind::Module, format!("impl {name}"), None)
        }
        "use" | "import" | "alias" | "require" => {
            let text = node_text(node, lines);
            (OutlineKind::Import, text, None)
        }
        _ => return None,
    };

    // Collect children for modules, protocols, impls
    let children = if matches!(kind, OutlineKind::Module | OutlineKind::Interface) && depth < 1 {
        elixir_collect_children(node, lines, lang, depth + 1)
    } else {
        Vec::new()
    };

    // Extract @doc / @moduledoc from previous sibling
    let doc = elixir_extract_doc(node, lines);

    Some(OutlineEntry {
        kind,
        name,
        start_line,
        end_line,
        signature,
        children,
        doc,
    })
}

/// Convert an Elixir `unary_operator` node (`@type`, `@typep`, `@opaque`) to an outline entry.
fn elixir_attr_to_entry(node: tree_sitter::Node, lines: &[&str]) -> Option<OutlineEntry> {
    let operand = node.child_by_field_name("operand")?;
    if operand.kind() != "call" {
        return None;
    }
    let target = operand.child_by_field_name("target")?;
    let attr_name = node_text(target, lines);
    let start_line = node.start_position().row as u32 + 1;
    let end_line = node.end_position().row as u32 + 1;
    match attr_name.as_str() {
        "type" | "typep" | "opaque" => {
            let name = elixir_type_name(operand, lines)?;
            let sig = node_text(node, lines);
            Some(OutlineEntry {
                kind: OutlineKind::TypeAlias,
                name,
                start_line,
                end_line,
                signature: Some(sig),
                children: Vec::new(),
                doc: None,
            })
        }
        "callback" | "macrocallback" => {
            let name = elixir_callback_name(operand, lines)?;
            let sig = node_text(node, lines);
            Some(OutlineEntry {
                kind: OutlineKind::Function,
                name,
                start_line,
                end_line,
                signature: Some(sig),
                children: Vec::new(),
                doc: None,
            })
        }
        _ => None,
    }
}

/// Extract the first argument text from an Elixir call node.
/// For `defmodule Foo.Bar do ... end`, returns `"Foo.Bar"`.
fn elixir_first_arg_text(node: tree_sitter::Node, lines: &[&str]) -> Option<String> {
    let args = super::treesitter::elixir_arguments(node)?;
    let mut cursor = args.walk();
    for child in args.children(&mut cursor) {
        if child.is_named() {
            return Some(node_text(child, lines));
        }
    }
    None
}

/// Extract function name from an Elixir `def`/`defp` call node.
///
/// For `def greet(name) do ... end`, the AST is:
///   call[target=def] → arguments → call[target=greet] → arguments → ...
/// For `def greet(name), do: ...` (keyword form), same structure.
fn elixir_func_name(node: tree_sitter::Node, lines: &[&str]) -> Option<String> {
    let args = super::treesitter::elixir_arguments(node)?;
    let mut cursor = args.walk();
    for child in args.children(&mut cursor) {
        if !child.is_named() {
            continue;
        }
        return super::treesitter::elixir_extract_func_head_name(child, lines);
    }
    None
}

/// Extract type name from an Elixir `@type` call.
/// For `@type t :: %{...}`, the call operand is `type t :: %{...}`,
/// and we extract `t` from the first argument.
fn elixir_type_name(call: tree_sitter::Node, lines: &[&str]) -> Option<String> {
    let args = super::treesitter::elixir_arguments(call)?;
    let mut cursor = args.walk();
    for child in args.children(&mut cursor) {
        if !child.is_named() {
            continue;
        }
        // `type t :: ...` → binary_operator with left=identifier
        if child.kind() == "binary_operator" {
            if let Some(left) = child.child_by_field_name("left") {
                // left may be a call like `t()` or an identifier `t`
                if left.kind() == "call" {
                    if let Some(target) = left.child_by_field_name("target") {
                        return Some(node_text(target, lines));
                    }
                }
                return Some(node_text(left, lines));
            }
        }
        // Bare identifier
        if child.kind() == "identifier" {
            return Some(node_text(child, lines));
        }
    }
    None
}

/// Extract callback name from an Elixir `@callback` call.
/// For `@callback handle_event(event :: term()) :: :ok`, the call operand is
/// `callback handle_event(...) :: :ok`. The arguments contain a `binary_operator`
/// with `::`, whose left side is a `call` with target = the callback name.
fn elixir_callback_name(call: tree_sitter::Node, lines: &[&str]) -> Option<String> {
    let args = super::treesitter::elixir_arguments(call)?;
    let mut cursor = args.walk();
    for child in args.children(&mut cursor) {
        if !child.is_named() {
            continue;
        }
        if child.kind() == "binary_operator" {
            // `handle_event(...) :: return_type` → left is the function head
            if let Some(left) = child.child_by_field_name("left") {
                return super::treesitter::elixir_extract_func_head_name(left, lines);
            }
        }
        // Bare callback without return type spec (unlikely but handle it)
        return super::treesitter::elixir_extract_func_head_name(child, lines);
    }
    None
}

/// Collect child entries from an Elixir module/protocol/impl `do_block`.
///
/// This intentionally includes `use`/`alias`/`import`/`require` as import entries
/// inside module outlines. In Elixir these are structural — `use GenServer` injects
/// callbacks, `alias Foo.Bar` affects name resolution — so they provide useful
/// context alongside function definitions.
fn elixir_collect_children(
    node: tree_sitter::Node,
    lines: &[&str],
    lang: Lang,
    depth: usize,
) -> Vec<OutlineEntry> {
    let mut children = Vec::new();
    let mut cursor = node.walk();

    // Find the do_block child
    let Some(do_block) = node.children(&mut cursor).find(|c| c.kind() == "do_block") else {
        return children;
    };

    let mut cursor2 = do_block.walk();
    for child in do_block.children(&mut cursor2) {
        if let Some(entry) = node_to_entry(child, lines, lang, depth) {
            children.push(entry);
        }
    }

    children
}

/// Extract @doc or @moduledoc text from the previous sibling of an Elixir definition.
///
/// In Elixir, `@doc "text"` is a `unary_operator` node. We check if the
/// previous sibling is such a node and extract the string content.
fn elixir_extract_doc(node: tree_sitter::Node, lines: &[&str]) -> Option<String> {
    let prev = node.prev_sibling()?;
    if prev.kind() != "unary_operator" {
        return None;
    }
    let operand = prev.child_by_field_name("operand")?;
    if operand.kind() != "call" {
        return None;
    }
    let target = operand.child_by_field_name("target")?;
    let attr = node_text(target, lines);
    if attr != "doc" && attr != "moduledoc" {
        return None;
    }
    // Get the doc argument — use tree-sitter node types to handle all forms:
    //   `@doc "text"`           → string node
    //   `@doc """heredoc"""`    → string node (multi-line)
    //   `@doc ~S"""sigil"""`    → sigil node
    //   `@doc ~s"""sigil"""`    → sigil node
    //   `@doc false`            → boolean node (suppress docs)
    let args = super::treesitter::elixir_arguments(operand)?;
    let mut cursor = args.walk();
    for child in args.children(&mut cursor) {
        if !child.is_named() {
            continue;
        }
        match child.kind() {
            // `@doc false` suppresses documentation
            "boolean" => return None,
            // Regular string (`"text"`, `"""heredoc"""`) or sigil (`~S"""..."""`, `~s"""..."""`)
            "string" | "sigil" => {
                return elixir_extract_doc_string(child, lines);
            }
            _ => {}
        }
    }
    None
}

/// Extract the first meaningful line from an Elixir doc string or sigil node.
///
/// For single-line strings (`"text"`), returns the content without quotes.
/// For heredocs/sigils (`"""..."""`, `~S"""..."""`), returns the first
/// non-empty content line. Uses tree-sitter source lines rather than
/// fragile string trimming.
fn elixir_extract_doc_string(node: tree_sitter::Node, lines: &[&str]) -> Option<String> {
    let start_row = node.start_position().row;
    let end_row = node.end_position().row;

    if start_row == end_row {
        // Single-line: `"text"` or `~s"text"` — strip delimiters and sigil prefix
        let mut text = node_text(node, lines);
        // Strip sigil prefix (~s, ~S, etc.) if present
        if text.starts_with('~') && text.len() >= 2 {
            text = text[2..].to_string();
        }
        let trimmed = text.trim_matches('"').trim();
        if trimmed.is_empty() {
            return None;
        }
        return Some(trimmed.to_string());
    }

    // Multi-line (heredoc or sigil): scan interior lines for first non-empty content
    for row in (start_row + 1)..end_row {
        if row >= lines.len() {
            break;
        }
        let line = lines[row].trim();
        if !line.is_empty() && line != "\"\"\"" {
            return Some(line.to_string());
        }
    }
    None
}

/// Extract the source module name from an import statement text.
/// Handles: `use std::fs;` → `std::fs`, `import X from "react"` → `react`,
/// `from collections import X` → `collections`
///
/// The `lang` parameter is needed to disambiguate `use` (Rust path vs Elixir module)
/// and `import` (JS/TS `from` syntax vs Elixir/Python/Go bare module name).
pub(crate) fn extract_import_source(text: &str, lang: Option<crate::types::Lang>) -> String {
    // BOM-aware at the front so a line-1 import in a BOM'd file extracts the same source
    // as it would without one — and, more importantly, so extraction cannot disagree with
    // `is_import_line`, which uses the same helper. See `trim_start_bom_aware`.
    let trimmed = trim_start_bom_aware(text).trim_end().trim_end_matches(';');

    // Bash: `source ./lib.sh`, `. ./lib.sh`, or tab-separated variants
    if lang == Some(crate::types::Lang::Bash) {
        let after = trimmed
            .strip_prefix("source")
            .or_else(|| trimmed.strip_prefix('.'))
            .filter(|rest| rest.starts_with(char::is_whitespace))
            .map_or(trimmed, str::trim_start);
        // Skip variable-expanded paths (contain `$`)
        if after.contains('$') {
            return String::new();
        }
        return after.trim_matches(|c| c == '"' || c == '\'').to_string();
    }

    // Elixir: `use GenServer`, `import Kernel`, `alias Foo.Bar`, `require Logger`
    // Must be checked before the Rust `use` and JS `import` branches.
    if lang == Some(crate::types::Lang::Elixir) {
        for prefix in &["use ", "import ", "alias ", "require "] {
            if let Some(rest) = trimmed.strip_prefix(prefix) {
                return rest.split(',').next().unwrap_or(rest).trim().to_string();
            }
        }
        return trimmed.to_string();
    }

    // Rust: `use foo::bar` → `foo::bar`
    if let Some(rest) = trimmed.strip_prefix("use ") {
        return rest
            .split('{')
            .next()
            .unwrap_or(rest)
            .trim()
            .trim_end_matches("::")
            .to_string();
    }

    // JS/TS: `import ... from "source"` or `import "source"`
    if trimmed.starts_with("import") {
        if let Some(from_pos) = trimmed.find("from ") {
            let source = &trimmed[from_pos + 5..];
            return source
                .trim()
                .trim_matches(|c| c == '"' || c == '\'' || c == ';')
                .to_string();
        }
        // Direct import: `import "source"`
        let after = trimmed.strip_prefix("import ").unwrap_or("");
        return after
            .trim()
            .trim_matches(|c| c == '"' || c == '\'' || c == ';')
            .to_string();
    }

    // Python: `from module import ...` or `import module`
    if let Some(rest) = trimmed.strip_prefix("from ") {
        return rest.split_whitespace().next().unwrap_or("").to_string();
    }
    if let Some(rest) = trimmed.strip_prefix("import ") {
        return rest.split_whitespace().next().unwrap_or("").to_string();
    }

    // C/C++: #include "file.h" or #include <header>
    if let Some(rest) = c_include_directive_rest(trimmed) {
        return c_include_header_name(rest);
    }

    // Go: `import "source"` — already handled above via "import"
    // Fallback: first meaningful token
    trimmed
        .split_whitespace()
        .last()
        .unwrap_or(trimmed)
        .to_string()
}

/// A leading UTF-8 BOM removed, and nothing else touched.
///
/// The counterpart to `trim_start_bom_aware`, for callers that hand their content to a
/// *parser* rather than testing it for a line prefix. Those callers must not have their
/// leading whitespace eaten — it is either legal input the parser already handles (JSON,
/// YAML) or load-bearing indentation — so the two cannot be the same function.
///
/// `serde_json` rejects a leading BOM outright rather than skipping it, which is not a
/// quirk of ours to work around elsewhere: it is why a BOM'd `.json` file outlined as
/// `[parse error: expected value at line 1 column 1]`, why a BOM'd `package.json` dropped
/// the entire manifest block from the initialize fingerprint, and why `tilth install`
/// aborted on a BOM'd host config claiming it was "invalid JSON" when it was valid JSON.
/// The `toml` crate strips a BOM itself, so TOML never needed this; calling it there
/// anyway is a harmless no-op and keeps the treatment uniform.
///
/// Repeats are stripped for the same reason `trim_start_bom_aware` does it — a tool that
/// prepends a BOM without checking for an existing one leaves two, and serde rejects the
/// second exactly as it rejected the first.
pub(crate) fn strip_bom(content: &str) -> &str {
    content.trim_start_matches('\u{feff}')
}

/// `strip_bom` for callers holding raw bytes rather than a `&str`.
///
/// Several markdown paths parse the mmap directly and so cannot use the `&str` form. The
/// three on the read side use this and must agree, because one stripping while another does
/// not is how an outline came to advertise a heading anchor that the section resolver then
/// denied: `read::outline::generate`'s markdown arm, `resolve_heading`, and
/// `suggest_headings`. Two more parse markdown on the *search* side — `search::symbol`'s
/// definition matcher and `search::mod`'s enclosing-scope lookup — and both strip as well,
/// via the `&str` form `strip_bom`. All five agree, so no surface disagrees with another
/// about a doubled-BOM file's first heading (#51/#87).
///
/// A BOM contains no newline, so removing it never shifts a line number — only column 0 of
/// row 0, which no caller tests.
///
/// Repeats are stripped for the same reason `strip_bom` strips them: a tool that prepends a
/// BOM without checking for an existing one leaves two. That is not hypothetical here —
/// tree-sitter-md skips a *single* leading BOM by itself, so one BOM parses correctly with
/// or without this, but two make it parse the first heading as a paragraph, dropping that
/// heading from the outline (the rest survive).
pub(crate) fn strip_bom_bytes(buf: &[u8]) -> &[u8] {
    const BOM: &[u8] = &[0xEF, 0xBB, 0xBF];
    let mut rest = buf;
    while let Some(stripped) = rest.strip_prefix(BOM) {
        rest = stripped;
    }
    rest
}

/// Leading whitespace stripped, plus a UTF-8 BOM if one is sitting in front of it.
///
/// `str::trim_start` trims Unicode `White_Space`, and U+FEFF is *not* in that class — it
/// was removed from it years ago. So a file saved with a BOM yields a first line whose
/// text begins with U+FEFF, every `starts_with`/`strip_prefix` test against it fails, and
/// line 1 is not an import to any consumer: it contributes to neither `uses_local` nor
/// `uses_external` and vanishes from `tilth_deps` with no warning. Roughly 3% of files in
/// two large C++ trees carry a BOM; the exposure is worse for Rust, Python and TypeScript,
/// where line 1 is far more often the first import (58% of tilth's own Rust files open
/// with `use `).
///
/// Only line 1 of a file can be affected — a BOM occurs once, at file start — but this is
/// the one line-prefix decision shared by every language arm, so one helper covers all of
/// them. Detection (`read::imports::is_import_line`) and extraction
/// (`extract_import_source`) must both use it. Fixing only detection would leave the BOM'd
/// line passing detection and then missing every per-language `strip_prefix` in extraction,
/// falling through to the generic last-token fallback: `from .mod_a import X` would yield
/// the dependency name `X`, and `import { A } from './mod';` would yield `'./mod';`. The
/// fallback happens to be right for Rust, so the damage is language-dependent — but where
/// it lands it converts a silent drop into a bogus entry, which is not an improvement.
///
/// Whitespace on both sides of the BOM is handled, since a BOM is a byte-order marker
/// rather than a syntactic element and nothing forbids `\u{feff}    use foo;`. Repeats are
/// handled too: prepending a BOM to an already-BOM'd file is a real artifact of tools that
/// do not check for one first, and stopping after the first would leave the second in place
/// and drop the import exactly as before.
pub(crate) fn trim_start_bom_aware(line: &str) -> &str {
    // One pass over "whitespace or BOM" rather than chained trims, so any interleaving of
    // the two is consumed. `str::trim_start` is defined as `char::is_whitespace`, so this
    // is exactly `trim_start` widened by U+FEFF and nothing else.
    line.trim_start_matches(|c: char| c.is_whitespace() || c == '\u{feff}')
}

/// The text following a C/C++ `#include` directive, or `None` when the line is not one.
///
/// Whitespace is legal between the `#` and the directive name — `# include "X.h"`, and
/// equally `#\tinclude` — and is not rare in older codebases. Requiring `#include` as a
/// single token meant such a line was not treated as an import by *any* consumer: it
/// contributed to neither `uses_local` nor `uses_external` and vanished from `tilth_deps`
/// with no warning.
///
/// Detection (`read::imports::is_import_line`) and extraction (`extract_import_source`)
/// both route through here so they cannot disagree about which lines are includes. That
/// they were separate judgements is exactly how the trailing-comment bug managed to drop
/// includes silently: one function said "import", the other produced a name that resolved
/// to nothing, and the line fell between the two buckets.
///
/// `#include_next` is left for `c_include_header_name` to strip, so the returned text is
/// everything after `include` either way.
pub(crate) fn c_include_directive_rest(line: &str) -> Option<&str> {
    trim_start_bom_aware(line)
        .strip_prefix('#')?
        .trim_start()
        .strip_prefix("include")
}

/// The delimited header name in the text following `#include`, delimiters kept.
///
/// Everything after the closing delimiter is discarded. That matters because it is
/// legal — and in some codebases habitual — to comment an include:
///
/// ```text
/// #include "Widget.h" // forward decls only
/// ```
///
/// Returning the whole remainder made the header name `"Widget.h" // forward decls only`,
/// which resolves to nothing on disk. `is_external` still saw a leading quote, so the
/// include was neither local nor external and vanished from `tilth_deps` with no warning:
/// a file whose every include carried a trailing comment reported no dependencies at all.
///
/// The `"…"` / `<…>` delimiters are preserved because `is_external` distinguishes a system
/// header from a project-relative one by the opening delimiter.
///
/// A comment *before* the header name (`#include /* why */ "Widget.h"`) is skipped for the
/// same reason: left in place it made the text start with `/`, which is neither delimiter,
/// and the include took the pass-through path below and was dropped exactly as a trailing
/// comment used to be.
///
/// `#include_next` — real in glibc and gcc system headers — is accepted too.
/// `c_include_directive_rest` matches on `include` alone, so it arrives here regardless;
/// recognising it costs one `strip_prefix` and the alternative is silently discarding it.
///
/// Text that still does not begin with a delimiter is returned trimmed and unchanged. An
/// `#include SOME_MACRO` has no header name to find and inventing one would be worse. Note
/// what happens to such a value downstream, because it is not obvious: `is_external` sees no
/// leading quote and routes it to the external bucket, where `is_valid_module_path` accepts
/// it — no space, alphanumeric first character — so `tilth_deps` reports the macro *name* as
/// an external dependency. Passing the text through is honest about having no answer, but
/// the result is a bogus-looking entry rather than a dropped one. Verified, not assumed:
/// `#include SOME_MACRO` renders as `SOME_MACRO` under `## Uses (external)`.
fn c_include_header_name(after_include: &str) -> String {
    // `#include_next "x.h"` arrives as `_next "x.h"`.
    let rest = after_include.strip_prefix("_next").unwrap_or(after_include);
    let rest = skip_leading_comments(rest.trim());
    let close = match rest.chars().next() {
        Some('"') => '"',
        Some('<') => '>',
        _ => return rest.to_string(),
    };
    match rest[1..].find(close) {
        // +2: one for the opening delimiter, one to include the closing one. Both
        // delimiters are single-byte ASCII and `find` searches within `rest[1..]`, so this
        // always lands on a char boundary even for a non-ASCII header name.
        Some(end) => rest[..end + 2].to_string(),
        // Unterminated. Pass it through rather than inventing a boundary.
        None => rest.to_string(),
    }
}

/// Strip any `/* … */` comments at the start of `s`, plus the whitespace around them.
///
/// A `//` comment cannot precede the header name — everything after it is comment — so only
/// the block form is worth handling. An unterminated block comment consumes the rest.
fn skip_leading_comments(s: &str) -> &str {
    let mut cur = s.trim_start();
    while let Some(after_open) = cur.strip_prefix("/*") {
        cur = match after_open.find("*/") {
            Some(end) => after_open[end + 2..].trim_start(),
            None => return "",
        };
    }
    cur
}

/// Get structured outline entries for file content.
///
/// The BOM is stripped here rather than at each caller (#88). This is the shared entry point
/// for `deps`' exported-symbol extraction, `diff`'s structural overlays and `read`'s
/// `[signature]` view, and it was the one outline path that parsed unstripped — `read::outline::
/// generate` has stripped since #42.
///
/// **A single BOM was never the problem, and that is why this went unnoticed.** Measured across
/// all nineteen `Lang` variants: every code grammar skips one leading U+FEFF by itself, so names,
/// line numbers and signatures are identical with it and without. The damage needs a *doubled*
/// BOM — a real artifact of a tool that prepends one without checking for an existing one, and
/// the same spelling #51 needed to expose tree-sitter-md. Two of nineteen grammars break on it,
/// in two different ways:
///
///   * **Kotlin** drops the line-1 definition from the outline entirely (one entry becomes zero).
///     For `deps` that means Phase 1 under-counts exported symbols and Phase 3's reverse search
///     never looks for them, so `tilth_deps` silently under-reports its blast radius.
///   * **Bash** fuses the BOM onto the *name*, yielding `\u{feff}line_one`. Worse than dropping
///     it: the symbol is searched for under a name no call site spells, so the miss looks like a
///     real absence of dependents, and the U+FEFF rides into any surface rendering the name.
///
/// Safe at this depth because an `OutlineEntry` carries line numbers and no byte offsets, and a
/// BOM contains no newline — so stripping shifts nothing a caller could hold against the
/// unstripped content it still owns. `mcp::tools::read::read_signature_file` relies on exactly
/// that: it keeps hashing its own unstripped lines for the `{line}:{hash}|` anchors while reading
/// entries from here, so the anchor contract of the `read:signature` surface is untouched — as
/// `bom_surfaces`' `KeepsForAnchors` row for it still asserts.
pub fn get_outline_entries(content: &str, lang: Lang) -> Vec<OutlineEntry> {
    let Some(ts_lang) = outline_language(lang) else {
        return Vec::new();
    };

    // Once, before both the parse and the line split — they must see the same string, or the
    // rows tree-sitter reports would index into a `lines` that no longer matches them.
    let content = strip_bom(content);

    // Budgeted: `diff` builds overlays with `par_iter`, calling this twice per changed file
    // (old and new content), so this is a walk-time transient tree like the search paths (#70).
    let Some(tree) = crate::lang::parse_budget::parse_budgeted(content, Some(lang), &ts_lang)
    else {
        return Vec::new();
    };

    let lines: Vec<&str> = content.lines().collect();
    walk_top_level(tree.root_node(), &lines, lang)
}

#[cfg(test)]
mod outline_entry_bom_tests {
    use super::get_outline_entries;
    use crate::types::Lang;

    /// A definition on line 1 for every language that has a grammar, plus the three that do not.
    ///
    /// Line 1 specifically: a BOM occurs once, at file start, so it is the only line any of this
    /// can reach. Where a language forbids a definition there — Go needs its `package` clause, PHP
    /// its open tag — the definition sits as early as the grammar allows and the row still proves
    /// the file's *first* bytes do not perturb what follows.
    const CASES: &[(Lang, &str)] = &[
        (Lang::Rust, "pub fn line_one() -> u32 {\n    1\n}\n"),
        (
            Lang::TypeScript,
            "export function lineOne(): number {\n  return 1;\n}\n",
        ),
        (
            Lang::Tsx,
            "export function LineOne(): JSX.Element {\n  return <div />;\n}\n",
        ),
        (
            Lang::JavaScript,
            "export function lineOne() { return 1; }\n",
        ),
        (Lang::Python, "def line_one():\n    return 1\n"),
        (
            Lang::Go,
            "package p\n\nfunc LineThree() int {\n\treturn 3\n}\n",
        ),
        (Lang::Java, "class LineOne {\n  int m() { return 1; }\n}\n"),
        (Lang::Scala, "class LineOne {\n  def m(): Int = 1\n}\n"),
        (Lang::C, "int line_one(void) { return 1; }\n"),
        (Lang::Cpp, "int line_one() { return 1; }\n"),
        (Lang::Ruby, "class LineOne\n  def m\n    1\n  end\nend\n"),
        (Lang::Php, "<?php\nfunction line_two() { return 2; }\n"),
        (Lang::Swift, "func lineOne() -> Int { return 1 }\n"),
        (Lang::Kotlin, "fun lineOne(): Int { return 1 }\n"),
        (
            Lang::CSharp,
            "class LineOne {\n  int M() { return 1; }\n}\n",
        ),
        (Lang::Elixir, "defmodule LineOne do\n  def m, do: 1\nend\n"),
        (Lang::Bash, "line_one() {\n  echo 1\n}\n"),
        (Lang::Dockerfile, "FROM scratch\n"),
        (Lang::Make, "line_one:\n\techo 1\n"),
    ];

    /// `n` BOMs, built from their bytes per the #35/#41 convention.
    fn with_boms(n: usize, s: &str) -> String {
        let mut v = Vec::new();
        for _ in 0..n {
            v.extend_from_slice(&[0xEF, 0xBB, 0xBF]);
        }
        v.extend_from_slice(s.as_bytes());
        String::from_utf8(v).unwrap()
    }

    /// Compare the whole entry, not just the name — a shifted line or a lost signature is the
    /// same class of wrong answer and neither shows up in a name-only check.
    fn describe(src: &str, lang: Lang) -> Vec<String> {
        get_outline_entries(src, lang)
            .iter()
            .map(|e| {
                format!(
                    "{:?}:{}@{}-{}:{:?}",
                    e.kind, e.name, e.start_line, e.end_line, e.signature
                )
            })
            .collect()
    }

    /// #88: a BOM'd file must outline exactly as its BOM-free twin, in every language.
    ///
    /// Asserted for **every** `Lang`, not only the two that were broken, and as parity against
    /// the BOM-free spelling rather than against expected entries — so it keeps holding as the
    /// grammars are bumped, and a grammar that starts mishandling a BOM fails here rather than
    /// silently under-reporting through `deps`.
    ///
    /// Two BOMs, not one, is the load-bearing half. Every code grammar skips a single leading
    /// U+FEFF by itself, so a one-BOM fixture passes with or without the strip and would be the
    /// fake coverage `mcp::bom_surfaces` exists to kill. Both spellings are checked, and the
    /// doubled one is what fails without the fix: `Kotlin` returned **zero** entries for a
    /// one-definition file, and `Bash` named its function `\u{feff}line_one`.
    #[test]
    fn a_bom_does_not_change_the_outline_entries_in_any_language() {
        for (lang, src) in CASES {
            let plain = describe(src, *lang);
            for n in [1, 2] {
                assert_eq!(
                    describe(&with_boms(n, src), *lang),
                    plain,
                    "{n} BOM(s) changed the outline entries for {lang:?}"
                );
            }
        }
    }

    /// The fixtures have to actually produce entries, or the parity above is `[] == []`.
    ///
    /// Three languages legitimately yield none — `Dockerfile` and `Make` have no grammar at all
    /// (`outline_language` returns `None`), and the Ruby arm emits no top-level entry for a class.
    /// Naming them here is what stops a *fourth* joining them unnoticed and quietly turning its
    /// parity row into a comparison of two empty vectors.
    #[test]
    fn every_language_whose_outline_can_be_empty_is_named() {
        const KNOWN_EMPTY: &[Lang] = &[Lang::Dockerfile, Lang::Make, Lang::Ruby];
        for (lang, src) in CASES {
            let empty = describe(src, *lang).is_empty();
            assert_eq!(
                empty,
                KNOWN_EMPTY.contains(lang),
                "{lang:?} renders {} outline entries, but KNOWN_EMPTY says the opposite — either \
                 the fixture stopped exercising the parity check above, or a grammar gained \
                 support and the list is stale",
                if empty { "no" } else { "some" }
            );
        }
    }
}

#[cfg(test)]
mod c_include_tests {
    use super::{c_include_header_name, extract_import_source};
    use crate::types::Lang;

    /// Everything after the closing delimiter must be discarded. Returning the whole
    /// remainder of the line made a commented include's "path" include the comment, which
    /// resolved to nothing on disk while `is_external` still saw a leading quote — so the
    /// include landed in neither the local nor the external bucket and disappeared from
    /// `tilth_deps` silently. A file whose every include carried a trailing comment
    /// reported no dependencies at all.
    #[test]
    fn trailing_comment_is_not_part_of_the_header_name() {
        for line in [
            "#include \"Widget.h\" // forward decls only",
            "#include \"Widget.h\" /* forward decls only */",
            "#include \"Widget.h\"\t// tab-separated",
            "#include \"Widget.h\"   ",
            "#include\"Widget.h\"// no spaces anywhere",
        ] {
            assert_eq!(
                extract_import_source(line, Some(Lang::Cpp)),
                "\"Widget.h\"",
                "line: {line}"
            );
        }
    }

    /// The delimiters are load-bearing: `is_external` tells a system header from a
    /// project-relative one by the opening one.
    #[test]
    fn delimiters_are_preserved_for_both_forms() {
        assert_eq!(
            extract_import_source("#include <vector> // std", Some(Lang::Cpp)),
            "<vector>"
        );
        assert_eq!(
            extract_import_source("#include \"a/b.h\"", Some(Lang::Cpp)),
            "\"a/b.h\""
        );
    }

    /// A comment character inside the header name is part of the path, not a comment.
    #[test]
    fn slashes_inside_the_delimiters_survive() {
        assert_eq!(
            extract_import_source("#include \"a/b/c.h\" // note", Some(Lang::Cpp)),
            "\"a/b/c.h\""
        );
    }

    /// No delimiter and no close: pass through rather than invent a boundary. An
    /// `#include SOME_MACRO` has no header name, and an unterminated one is a broken file.
    #[test]
    fn undelimited_and_unterminated_forms_pass_through() {
        assert_eq!(c_include_header_name(" SOME_MACRO"), "SOME_MACRO");
        assert_eq!(
            c_include_header_name(" \"unterminated.h"),
            "\"unterminated.h"
        );
        assert_eq!(c_include_header_name(" <unterminated.h"), "<unterminated.h");
        assert_eq!(c_include_header_name(""), "");
    }

    /// A comment before the header name took the pass-through path — the text started with
    /// `/`, not a delimiter — and was dropped exactly as a trailing comment used to be.
    #[test]
    fn comment_before_the_header_name_is_skipped() {
        for line in [
            "#include /* why */ \"Widget.h\"",
            "#include /*a*/ /*b*/ \"Widget.h\" // and after",
            "#include/*tight*/\"Widget.h\"",
        ] {
            assert_eq!(
                extract_import_source(line, Some(Lang::Cpp)),
                "\"Widget.h\"",
                "line: {line}"
            );
        }
        assert_eq!(
            extract_import_source("#include /* why */ <vector>", Some(Lang::Cpp)),
            "<vector>"
        );
        // An unterminated block comment swallows the line rather than yielding a bogus name.
        assert_eq!(
            extract_import_source("#include /* never closed", Some(Lang::Cpp)),
            ""
        );
    }

    /// `#include_next` reaches this code because `is_import_line` matches on `#include`
    /// alone. It is a real directive in glibc and gcc headers; recognising it beats
    /// silently discarding it.
    #[test]
    fn include_next_is_recognised() {
        assert_eq!(
            extract_import_source("#include_next \"limits.h\" // chain", Some(Lang::Cpp)),
            "\"limits.h\""
        );
        assert_eq!(
            extract_import_source("#include_next <stdio.h>", Some(Lang::Cpp)),
            "<stdio.h>"
        );
    }

    /// Whitespace between the `#` and the directive name is legal C and habitual in some
    /// older codebases. Requiring `#include` as a single token meant such a line was not an
    /// include to *any* consumer — it reached neither `uses_local` nor `uses_external` and
    /// disappeared from `tilth_deps` with no warning.
    #[test]
    fn whitespace_after_the_hash_is_allowed() {
        for line in [
            "# include \"Widget.h\"",
            "#  include \"Widget.h\"",
            "#\tinclude \"Widget.h\"",
            "  # include \"Widget.h\" // note",
        ] {
            assert_eq!(
                extract_import_source(line, Some(Lang::Cpp)),
                "\"Widget.h\"",
                "line: {line}"
            );
        }
        assert_eq!(
            extract_import_source("# include_next <stdio.h>", Some(Lang::Cpp)),
            "<stdio.h>"
        );
    }

    /// The `#` alone is not enough — other directives must not be mistaken for includes and
    /// routed to `c_include_header_name`, which would hand back a bogus header name.
    #[test]
    fn other_preprocessor_directives_are_not_includes() {
        for line in ["#pragma once", "# define INCLUDE_GUARD 1", "#ifndef X_H"] {
            assert!(
                super::c_include_directive_rest(line).is_none(),
                "must not be read as an include: {line}"
            );
        }
    }

    /// A non-ASCII header name must not panic the byte slicing in `c_include_header_name`.
    #[test]
    fn non_ascii_header_names_do_not_panic() {
        for (line, want) in [
            ("#include \"café.h\" // note", "\"café.h\""),
            ("#include <café世界.h>", "<café世界.h>"),
            ("#include \"😀/😀.h\"", "\"😀/😀.h\""),
        ] {
            assert_eq!(extract_import_source(line, Some(Lang::Cpp)), want);
        }
    }
}

#[cfg(test)]
mod markdown_helper_tests {
    use super::{heading_level, heading_text, parse_markdown};

    /// Walk the tree and collect every `atx_heading`/`setext_heading` node.
    fn collect_headings(tree: &tree_sitter::Tree) -> Vec<tree_sitter::Node<'_>> {
        let mut out = Vec::new();
        let mut cursor = tree.walk();
        let mut stack = vec![tree.root_node()];
        while let Some(node) = stack.pop() {
            if matches!(node.kind(), "atx_heading" | "setext_heading") {
                out.push(node);
            }
            for child in node.children(&mut cursor) {
                stack.push(child);
            }
        }
        out.sort_by_key(|n| n.start_position().row);
        out
    }

    #[test]
    fn parse_returns_block_tree_with_sections() {
        let src = "# Top\n\ncontent\n\n## Sub\n\nmore\n";
        let tree = parse_markdown(src).unwrap();
        let root = tree.root_node();
        assert_eq!(root.kind(), "document");
        // The document contains at least one section node.
        let mut cursor = root.walk();
        let has_section = root.children(&mut cursor).any(|c| c.kind() == "section");
        assert!(has_section, "expected document to contain section children");
    }

    #[test]
    fn fenced_code_blocks_do_not_emit_headings() {
        // The whole point: a `# foo` inside a fenced code block must NOT be
        // parsed as an atx_heading. The hand-rolled scanners had to track
        // fence state manually; the AST does this for free.
        let src = "# Real\n\n```python\n# fake heading\nprint('x')\n```\n\n## Also Real\n";
        let tree = parse_markdown(src).unwrap();
        let headings = collect_headings(&tree);
        let lines: Vec<&str> = src.lines().collect();
        let texts: Vec<String> = headings.iter().map(|n| heading_text(*n, &lines)).collect();
        assert_eq!(texts, vec!["Real".to_string(), "Also Real".to_string()]);
    }

    #[test]
    fn tilde_fences_are_recognised() {
        let src = "# Real\n\n~~~\n# inside tilde fence\n~~~\n\n## Other\n";
        let tree = parse_markdown(src).unwrap();
        let lines: Vec<&str> = src.lines().collect();
        let headings = collect_headings(&tree);
        let texts: Vec<String> = headings.iter().map(|n| heading_text(*n, &lines)).collect();
        assert_eq!(texts, vec!["Real".to_string(), "Other".to_string()]);
    }

    #[test]
    fn level_extraction_covers_h1_through_h6() {
        let src = "# A\n\n## B\n\n### C\n\n#### D\n\n##### E\n\n###### F\n";
        let tree = parse_markdown(src).unwrap();
        let headings = collect_headings(&tree);
        let levels: Vec<u8> = headings.iter().filter_map(|n| heading_level(*n)).collect();
        assert_eq!(levels, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn trailing_atx_close_hashes_are_stripped() {
        let src = "## Foo ##\n";
        let tree = parse_markdown(src).unwrap();
        let lines: Vec<&str> = src.lines().collect();
        let headings = collect_headings(&tree);
        assert_eq!(heading_text(headings[0], &lines), "Foo");
    }
}

#[cfg(test)]
mod bash_outline_tests {
    use super::{extract_import_source, get_outline_entries};
    use crate::search::callees::extract_callee_names;
    use crate::types::{Lang, OutlineKind};

    // Fixture covering both function syntaxes, top-level vars, and a nested local.
    const BASH_FIXTURE: &str = r#"MY_CONST=hello
DEBUG_MODE=0

greet() { echo "hi $1"; }

function cleanup {
    rm -f /tmp/x
}

main() {
    greet world
    cleanup
    local y=1
}
"#;

    #[test]
    fn bash_outline_functions_and_vars() {
        let entries = get_outline_entries(BASH_FIXTURE, Lang::Bash);

        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();

        // All three functions must appear
        assert!(
            names.contains(&"greet"),
            "expected greet in outline, got: {names:?}"
        );
        assert!(
            names.contains(&"cleanup"),
            "expected cleanup in outline, got: {names:?}"
        );
        assert!(
            names.contains(&"main"),
            "expected main in outline, got: {names:?}"
        );

        // Top-level variables must appear
        assert!(
            names.contains(&"MY_CONST"),
            "expected MY_CONST in outline, got: {names:?}"
        );
        assert!(
            names.contains(&"DEBUG_MODE"),
            "expected DEBUG_MODE in outline, got: {names:?}"
        );

        // Functions must have Function kind
        for fname in &["greet", "cleanup", "main"] {
            let entry = entries.iter().find(|e| e.name == *fname).unwrap();
            assert_eq!(
                entry.kind,
                OutlineKind::Function,
                "{fname} should be OutlineKind::Function"
            );
        }

        // Variables must have Variable kind
        for vname in &["MY_CONST", "DEBUG_MODE"] {
            let entry = entries.iter().find(|e| e.name == *vname).unwrap();
            assert_eq!(
                entry.kind,
                OutlineKind::Variable,
                "{vname} should be OutlineKind::Variable"
            );
        }

        // Nested `local y=1` must NOT appear at the top level
        assert!(
            !names.contains(&"y"),
            "nested local 'y' must not appear in top-level outline, got: {names:?}"
        );
    }

    #[test]
    fn bash_callee_names_for_main() {
        // Derive main's range from the outline so the test can't silently drift
        // if the fixture is edited.
        let main = get_outline_entries(BASH_FIXTURE, Lang::Bash)
            .into_iter()
            .find(|e| e.name == "main")
            .expect("main must be in the outline");
        let names = extract_callee_names(
            BASH_FIXTURE,
            Lang::Bash,
            Some((main.start_line, main.end_line)),
        );

        assert!(
            names.contains(&"greet".to_string()),
            "expected greet as callee, got: {names:?}"
        );
        assert!(
            names.contains(&"cleanup".to_string()),
            "expected cleanup as callee, got: {names:?}"
        );
        // echo is called inside greet, outside main's range, so it is absent here.
    }

    #[test]
    fn bash_outline_surfaces_declarations_and_hyphenated_names() {
        // export/declare/readonly declarations must surface (the common config
        // pattern), and hyphenated function names must be captured whole.
        let src = "export E_VAR=1\n\
                   declare -r D_VAR=2\n\
                   readonly R_VAR=3\n\
                   deploy-app() { :; }\n";
        let entries = get_outline_entries(src, Lang::Bash);
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();

        for v in ["E_VAR", "D_VAR", "R_VAR"] {
            let e = entries
                .iter()
                .find(|e| e.name == v)
                .unwrap_or_else(|| panic!("{v} should be outlined, got: {names:?}"));
            assert_eq!(e.kind, OutlineKind::Variable, "{v} should be a Variable");
        }
        let dep = entries
            .iter()
            .find(|e| e.name == "deploy-app")
            .unwrap_or_else(|| panic!("deploy-app should be outlined whole, got: {names:?}"));
        assert_eq!(dep.kind, OutlineKind::Function);
    }

    #[test]
    fn bash_extract_import_source_source_keyword() {
        let line = "source ./lib/utils.sh";
        let result = extract_import_source(line, Some(Lang::Bash));
        assert_eq!(result, "./lib/utils.sh");
    }

    #[test]
    fn bash_extract_import_source_dot_keyword() {
        let line = ". ./config.sh";
        let result = extract_import_source(line, Some(Lang::Bash));
        assert_eq!(result, "./config.sh");
    }

    #[test]
    fn bash_extract_import_source_quoted() {
        let line = r#"source "./lib/helpers.sh""#;
        let result = extract_import_source(line, Some(Lang::Bash));
        assert_eq!(result, "./lib/helpers.sh");
    }

    #[test]
    fn bash_extract_import_source_variable_expanded_returns_empty() {
        let line = r#"source "$DIR/lib.sh""#;
        let result = extract_import_source(line, Some(Lang::Bash));
        assert!(
            result.is_empty(),
            "variable-expanded source should return empty, got: {result:?}"
        );
    }

    #[test]
    fn bash_subscript_assignment_surfaces_base_name() {
        // `ARR[0]=hello` should appear as `ARR` (Variable), not `ARR[0]`.
        let entries = get_outline_entries("ARR[0]=hello\n", Lang::Bash);
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(
            names.contains(&"ARR"),
            "expected ARR in outline, got: {names:?}"
        );
        assert!(
            !names.contains(&"ARR[0]"),
            "ARR[0] must not appear verbatim in outline, got: {names:?}"
        );
        let entry = entries.iter().find(|e| e.name == "ARR").unwrap();
        assert_eq!(
            entry.kind,
            OutlineKind::Variable,
            "ARR should be OutlineKind::Variable"
        );
    }

    #[test]
    fn bash_extract_import_source_tab_separated() {
        // `source\t./lib.sh` (tab separator) must be parsed correctly.
        let result = extract_import_source("source\t./lib/utils.sh", Some(Lang::Bash));
        assert_eq!(result, "./lib/utils.sh");
    }
}
