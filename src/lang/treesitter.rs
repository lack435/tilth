//! Shared tree-sitter utilities used by symbol search and caller search.

use crate::types::Lang;

/// Definition node kinds across tree-sitter grammars.
///
/// Kind strings are shared across every grammar tilth ships, so a string added
/// here applies to all of them. `definition_kinds_are_not_ambiguous_across_grammars`
/// pins which grammars own each entry; kinds that mean different things in
/// different grammars live in `C_FAMILY_DEFINITION_KINDS` instead.
///
/// Private on purpose: every consumer goes through `is_definition_node`, which layers
/// on the language and body rules a bare membership test cannot express.
const DEFINITION_KINDS: &[&str] = &[
    // Functions
    "function_declaration",
    "function_definition",
    "function_item",
    "method_definition",
    "method_declaration",
    // Classes, structs & Kotlin objects
    "class_declaration",
    "class_definition",
    "struct_item",
    "object_declaration",
    // C/C++ type specifiers. `class_specifier` is C++-only; the other three are
    // shared by tree-sitter-c and tree-sitter-cpp and mean the same thing in both.
    // Only counted when the node carries a `body` — see `is_definition_node`.
    "class_specifier",
    "struct_specifier",
    "union_specifier",
    "enum_specifier",
    // C++ `using` aliases. C++-only.
    //
    // `template_declaration` is deliberately NOT here. It carries no name of its own
    // and wraps a declaration that is already a definition kind, so registering both
    // reported one template twice whenever the `template <…>` clause sat on its own
    // line — the normal spelling in real C++. It is a transparent wrapper instead
    // (`is_transparent_wrapper`), so the walk reaches the inner declaration without
    // spending a depth level and only that inner node registers.
    "alias_declaration",
    // Interfaces & types (TS)
    "interface_declaration",
    "trait_declaration",
    "type_alias_declaration",
    "type_item",
    // Enums
    "enum_item",
    "enum_declaration",
    // Variables, constants & properties (Kotlin, C#, Swift)
    "lexical_declaration",
    "variable_declaration",
    "variable_assignment", // Bash top-level assignments (bash-only today; a future grammar reusing this node kind would inherit definition_weight 60)
    "const_item",
    "const_declaration",
    "static_item",
    "property_declaration",
    // Rust-specific
    "trait_item",
    "impl_item",
    "mod_item",
    "namespace_definition",
    // Python
    "decorated_definition",
    // Go
    "type_declaration",
    // Go type declarations; also C/C++ `typedef` (tree-sitter-c/cpp
    // `type_definition`) and Scala `type` aliases, which share the kind string
    // and are all type definitions.
    "type_definition",
    // Exports
    "export_statement",
];

/// Definition kinds that only count in the C/C++ family.
///
/// `field_declaration` is a class/struct member declaration in tree-sitter-c and
/// tree-sitter-cpp — the node holding `void DoThing();` or `int Count;` inside a
/// class body, which is where C++ declares its members. But the same kind string
/// also exists in the Rust, Go, Java and C# grammars, where it means one field of
/// a struct/record. Registering it in `DEFINITION_KINDS` would report every Rust
/// and Go struct field as a top-level definition, so it is gated on language
/// instead.
const C_FAMILY_DEFINITION_KINDS: &[&str] = &["field_declaration"];

/// C/C++ type specifiers that are a definition only when they carry a `body`.
///
/// Without one the node is an *elaborated type specifier* — a reference to a type,
/// not a definition of it: `class Fwd;` (forward declaration), `class Fwd* p;`
/// (type reference), `friend class Other;`. Counting those would report a
/// definition at every forward declaration in every header.
pub(crate) const SPECIFIER_KINDS: &[&str] = &[
    "class_specifier",
    "struct_specifier",
    "union_specifier",
    "enum_specifier",
];

/// True when `node` is a C/C++ type specifier that actually defines the type — i.e.
/// carries a `body`. Shared so the outline and the definition walk cannot drift on
/// what counts as a definition.
pub(crate) fn is_bodied_specifier(node: tree_sitter::Node) -> bool {
    SPECIFIER_KINDS.contains(&node.kind()) && node.child_by_field_name("body").is_some()
}

/// True when `node` is a bodied specifier that also has a `name`.
///
/// An anonymous one (`struct { int a; } inst;`, `typedef struct { … } Foo;`) has a body
/// but nothing to call it, so preferring it over the declarator would trade a
/// searchable identifier for `<anonymous>`.
pub(crate) fn is_named_bodied_specifier(node: tree_sitter::Node) -> bool {
    is_bodied_specifier(node) && node.child_by_field_name("name").is_some()
}

/// C/C++ declarator node kinds — the chain wrapping a declared name.
///
/// `void Holder::Work()` nests as `function_declarator` → `qualified_identifier`
/// → `identifier`; pointers, references and arrays add further layers.
const C_DECLARATOR_KINDS: &[&str] = &[
    "function_declarator",
    "pointer_declarator",
    "reference_declarator",
    "array_declarator",
    "parenthesized_declarator",
    "init_declarator",
    "qualified_identifier",
];

/// True when `node` is a definition node for `lang`.
///
/// Wraps the flat `DEFINITION_KINDS` membership test with the rules that need more
/// than the kind string: the language-scoped kinds in `C_FAMILY_DEFINITION_KINDS`,
/// and the body requirement that separates a C/C++ type *definition* from a
/// reference to a type.
pub(crate) fn is_definition_node(node: tree_sitter::Node, lang: Option<Lang>) -> bool {
    let kind = node.kind();
    if C_FAMILY_DEFINITION_KINDS.contains(&kind) {
        return matches!(lang, Some(Lang::C | Lang::Cpp));
    }
    // A macro in a class head can misparse a class definition into a `declaration`
    // (see `is_cpp_misparsed_class_head`). Only that shape counts — an ordinary
    // C/C++ `declaration` is a local variable or a prototype, not a definition, and
    // `declaration` is not in `DEFINITION_KINDS` for exactly that reason.
    if kind == "declaration" {
        // Two shapes make a C/C++ `declaration` a definition. Everything else — a local
        // variable, a prototype, a global — is not, which is why `declaration` is
        // absent from `DEFINITION_KINDS`.
        //
        // 1. A macro in a class head can misparse a class definition into one
        //    (see `is_cpp_misparsed_class_head`).
        // 2. A *qualified data* declarator means the declaration defines something
        //    declared elsewhere: `int Widget::sCount = 0;` (or without an initialiser,
        //    which is equally a definition), the out-of-line definition of a static
        //    member.
        //
        // No language gate is needed, but only by cross-grammar coincidence, so it is
        // worth spelling out: `declaration` also exists in the JS, TS, TSX, Java, C#
        // and Kotlin grammars. Kotlin has no `declarator` field at all, so the chain
        // walk stops immediately; Java has a `declarator` field but no
        // `qualified_identifier` kind; the rest have neither. C and C++ are the only
        // grammars where both exist, so the test below can only ever fire for them.
        //
        // Case 2 needs both extra conditions. A qualified *function* declarator is not
        // a definition — it is `friend void Helper::Assist();`, which declares a
        // member of another class that is defined elsewhere; counting it made the
        // scope annotator attribute `Assist` to the befriending class. And an `extern`
        // storage class marks a declaration of something defined elsewhere by
        // definition. (A variable whose *type* is qualified — `Foo::Bar x;` — keeps
        // the qualifier on the `type` field, not the declarator chain, so it never
        // reaches this test.)
        if is_cpp_misparsed_class_head(node) {
            return true;
        }
        // 3. A member template declared inside a class body: `template <typename T> void
        //    Apply(T V);` nests as `field_declaration_list` → `template_declaration` →
        //    `declaration`, where a plain member would be a `field_declaration` (already
        //    covered by `C_FAMILY_DEFINITION_KINDS`). The wrapper is transparent, so this
        //    is the only place such a member can be recognised. Anchored on
        //    `field_declaration_list` so a *free* template prototype at namespace scope
        //    stays a declaration, consistent with plain prototypes.
        if is_cpp_member_template_declaration(node) {
            return true;
        }
        return declarator_chain_has_kind(node, "qualified_identifier")
            && !declarator_chain_has_function(node)
            && !has_extern_storage_class(node);
    }
    if !DEFINITION_KINDS.contains(&kind) {
        return false;
    }
    if SPECIFIER_KINDS.contains(&kind) {
        return node.child_by_field_name("body").is_some();
    }
    true
}

/// Recursion cap for the C/C++ declarator walk.
///
/// The walk always descends, so it cannot cycle — but the depth is attacker- (and
/// fuzzer-) controlled: `int` + N `*` + `p;` nests N `pointer_declarator`s, and
/// `outline::generate` is a fuzz target (`fuzz/fuzz_targets/outline.rs`, oss-fuzz)
/// reachable with inputs well under libFuzzer's default 4 KB `max_len`. Real C++
/// never approaches this; anything deeper is not worth a name.
const MAX_DECLARATOR_DEPTH: usize = 64;

/// Walk a C/C++ declarator chain down to the name it declares.
///
/// `void Holder::StaticWork()` → `"StaticWork"`, `int* Buffer` → `"Buffer"`,
/// `void Work()` inside a class body → `"Work"`. Returns the trailing segment of a
/// qualified name, matching how tilth names methods in every other language (a
/// Rust `impl` method resolves as `bar`, not `Foo::bar`); grok's own
/// `split_qualified` retry relies on the same convention.
pub(crate) fn c_declarator_name(node: tree_sitter::Node, lines: &[&str]) -> Option<String> {
    c_declarator_name_at(node, lines, 0)
}

fn c_declarator_name_at(node: tree_sitter::Node, lines: &[&str], depth: usize) -> Option<String> {
    if depth > MAX_DECLARATOR_DEPTH {
        return None;
    }
    match node.kind() {
        "identifier" | "field_identifier" | "type_identifier" | "destructor_name"
        | "operator_name" => {
            let text = node_text_simple(node, lines);
            (!text.is_empty()).then_some(text)
        }
        // `A::B::c` — recurse on `name`, which may itself be qualified.
        "qualified_identifier" => node
            .child_by_field_name("name")
            .and_then(|n| c_declarator_name_at(n, lines, depth + 1)),
        // `(*Cb)` in `typedef void (*Cb)(int);` — a `parenthesized_declarator` wraps
        // its inner declarator as an *unnamed* child, so `child_by_field_name` finds
        // nothing and the generic arm below would give up. Without this, function
        // pointer typedefs fall back to the raw declarator text.
        "parenthesized_declarator" => {
            let mut cursor = node.walk();
            let inner: Vec<tree_sitter::Node> = node
                .children(&mut cursor)
                .filter(tree_sitter::Node::is_named)
                .collect();
            inner
                .into_iter()
                .find_map(|c| c_declarator_name_at(c, lines, depth + 1))
        }
        // Every other declarator kind wraps the next one down.
        _ => inner_declarator(node).and_then(|d| c_declarator_name_at(d, lines, depth + 1)),
    }
}

/// The next declarator down a C/C++ declarator chain.
///
/// Most kinds expose the declarator they wrap as a `declarator` field, but
/// `reference_declarator` — `T& Get()`, `T&& Take()` — holds it as an *unnamed*
/// child, the same way `parenthesized_declarator` does. A bare field probe therefore
/// stops one link short of the name on every reference return, which is not a corner:
/// it is how a C++ singleton accessor, `at()`, `front()` and `operator[]` are all
/// spelled. The symbol resolved to nothing, so it was never indexed as a definition —
/// silently, since text search still matched the line and reported it as a usage.
fn inner_declarator(node: tree_sitter::Node) -> Option<tree_sitter::Node> {
    if let Some(declarator) = node.child_by_field_name("declarator") {
        return Some(declarator);
    }
    if !matches!(
        node.kind(),
        "reference_declarator" | "parenthesized_declarator"
    ) {
        return None;
    }
    let mut cursor = node.walk();
    // Bind rather than returning in tail position: `children` borrows `cursor`.
    let found = node.children(&mut cursor).find(tree_sitter::Node::is_named);
    found
}

/// True when `node` is a class head that an unknown macro made tree-sitter misparse.
///
/// `class LIBFOO_API Widget : public Base { … };` does not parse as a class.
/// tree-sitter-cpp has no way to know `LIBFOO_API` is a macro, so it reads it as the
/// class name and the real name as a declarator, producing a `function_definition`
/// (or, with multiple inheritance, a `declaration`) whose `type` is a *bodyless*
/// `class_specifier` — one named for the macro.
///
/// This is ordinary C++, not a framework quirk: an export macro between `class` and
/// the class name is how essentially every Windows C++ library spells its dllexport
/// attribute (`MYLIB_API`, `FOO_EXPORT`, `DLLEXPORT`). Detection is on AST shape
/// alone — no macro name is ever matched.
///
/// Four conditions separate this from the valid C++ that shares parts of the shape:
///   * a bodyless specifier as the `type`, and
///   * no `function_declarator` anywhere in the declarator chain — a real function
///     returning an elaborated type (`class Foo bar() { … }`) always has one, and
///   * a brace-delimited body somewhere in the node, and
///   * for `declaration` an `ERROR` child, which only appears when recovery had to
///     repair a malformed head (`class Foo* p;` parses cleanly and has none).
///
/// The brace requirement is what distinguishes *defining* a type from *referencing*
/// one, and it is load-bearing: an attribute macro between the type and the variable
/// name — `class Foo PACKED_ATTR bar;`, `struct FVector ALIGN16 Position;` — produces
/// the identical bodyless-specifier-plus-ERROR shape while declaring an ordinary
/// variable. Without the brace check those register as a class definition named after
/// the macro, in exactly the macro-heavy codebases this feature targets.
///
/// Residual known gap: a brace-*initialised* variable behind such a macro
/// (`struct FVector ALIGN16 P{0,0,0};`) still matches. That is a narrower shape than
/// what the check rejects, and the alternative — resolving whether the specifier
/// names a pre-existing type — needs a symbol table tilth does not have.
fn is_cpp_misparsed_class_head(node: tree_sitter::Node) -> bool {
    if !matches!(node.kind(), "function_definition" | "declaration") {
        return false;
    }
    // `SPECIFIER_KINDS` are C/C++-only node kinds, so this is inherently scoped to
    // the C/C++ grammars even though `declaration` exists in several others.
    let Some(type_node) = node.child_by_field_name("type") else {
        return false;
    };
    if !SPECIFIER_KINDS.contains(&type_node.kind())
        || type_node.child_by_field_name("body").is_some()
    {
        return false;
    }
    if declarator_chain_has_function(node) {
        return false;
    }
    if node.kind() == "declaration" && !has_error_child(node) {
        return false;
    }
    has_brace_body(node)
}

/// True when `node` encloses a brace-delimited body.
///
/// Where the body lands depends on how error recovery repaired the head, so all the
/// forms it takes in practice are accepted: a `compound_statement` (the
/// `function_definition` shape), an `initializer_list` (multiple inheritance, where
/// the body is re-read as a brace initialiser), or a bare `{` token stranded inside
/// an `ERROR`. Bounded depth — the body is never more than a couple of levels down,
/// and this runs per candidate node.
fn has_brace_body(node: tree_sitter::Node) -> bool {
    fn walk(node: tree_sitter::Node, depth: usize) -> bool {
        if depth > 3 {
            return false;
        }
        let mut cursor = node.walk();
        let found = node.children(&mut cursor).any(|c| {
            matches!(
                c.kind(),
                "compound_statement" | "field_declaration_list" | "initializer_list" | "{"
            ) || walk(c, depth + 1)
        });
        found
    }
    walk(node, 0)
}

/// Name of the class a C++ export macro caused tree-sitter to misparse, if any.
///
/// The real class name is the first identifier after the (macro-named) type
/// specifier. Which child holds it depends on how error recovery repaired the head
/// — for `class API Foo final : public Bar` it is the `declarator`, while for
/// `class API Foo final : public LongerBase` recovery instead swallows
/// `Foo final : public` into an `ERROR` and leaves the *base* class as the
/// declarator. Reading in source order and descending into an `ERROR` gets the
/// right name in both, rather than trusting one particular recovery shape.
pub(crate) fn cpp_misparsed_class_name(node: tree_sitter::Node, lines: &[&str]) -> Option<String> {
    if !is_cpp_misparsed_class_head(node) {
        return None;
    }
    let type_node = node.child_by_field_name("type")?;
    let mut cursor = node.walk();
    let mut past_type = false;
    for child in node.children(&mut cursor) {
        if child == type_node {
            past_type = true;
            continue;
        }
        if !past_type {
            continue;
        }
        let candidate = match child.kind() {
            "identifier" => Some(node_text_simple(child, lines)),
            "ERROR" => first_identifier_child(child, lines),
            _ => None,
        };
        if let Some(name) = candidate.filter(|n| !n.is_empty()) {
            return Some(name);
        }
    }
    None
}

/// First direct `identifier` child of `node`, if any.
fn first_identifier_child(node: tree_sitter::Node, lines: &[&str]) -> Option<String> {
    let mut cursor = node.walk();
    // Bind rather than returning the expression directly: `children` borrows `cursor`,
    // and the temporary would outlive it in tail position.
    let found = node
        .children(&mut cursor)
        .find(|c| c.kind() == "identifier")
        .map(|c| node_text_simple(c, lines));
    found
}

/// How far below its `compound_statement` body a misparsed class member can sit.
///
/// An access specifier wraps what follows it in a `labeled_statement`, and recovery
/// packs members it could not repair into nested `ERROR`s — `ERROR` inside `ERROR`
/// inside a `labeled_statement` is the deepest nesting observed.
const MISPARSED_BODY_DEPTH: usize = 4;

/// Name of the macro-misparsed class whose body encloses `node`, if any.
///
/// Members of such a body are not all direct children of the `compound_statement`:
/// an access specifier wraps the members after it, and recovery buries the ones it
/// could not repair in `ERROR`s. Walking up through just those two wrappers keeps
/// this precise — anything else in the chain means `node` is not a class member —
/// and cheap enough to ask per candidate, which is why the enclosing class does not
/// have to be threaded through every outline call.
pub(crate) fn enclosing_misparsed_class_name(
    node: tree_sitter::Node,
    lines: &[&str],
) -> Option<String> {
    let mut cur = node;
    for _ in 0..MISPARSED_BODY_DEPTH {
        let parent = cur.parent()?;
        if parent.kind() == "compound_statement" {
            return cpp_misparsed_class_name(parent.parent()?, lines);
        }
        if !matches!(parent.kind(), "ERROR" | "labeled_statement") {
            return None;
        }
        cur = parent;
    }
    None
}

/// The member a recovery artifact inside a misparsed class body declares, by name.
///
/// Constructors and destructors have no return type, so once the class head has
/// misparsed there is nothing left for tree-sitter to read them as a declaration —
/// a constructor becomes a *call*, and a destructor a stranded declarator inside an
/// `ERROR`. Both are ordinary members that the outline (and through it deps' exported
/// symbols and blast radius) would otherwise lose entirely.
///
/// `class` is the enclosing class's real name, from `cpp_misparsed_class_name`. It is
/// what separates a constructor from a zero-argument macro invocation
/// (`GENERATED_BODY()`), which recovery reshapes into the identical call — the same
/// rule `is_cpp_macro_invocation` applies to bodies that parsed cleanly, which cannot
/// see this shape.
///
/// Known residual: when a misparsed class has an inline-bodied *method*, recovery
/// reads that body's opening brace as an initialiser and the statements after the
/// first escape upward into the class body, where they are indistinguishable from
/// members. A leaked `Widget();` then reads as a constructor. The escaped statements
/// corrupt that outline wholesale — the class's own range is wrong too — so the fix
/// belongs with the escape, not here.
pub(crate) fn misparsed_member_name(
    node: tree_sitter::Node,
    lines: &[&str],
    class: &str,
) -> Option<String> {
    match node.kind() {
        // `Widget();` and `Widget(int a, float b);` — a call, with the arguments
        // re-read as an expression list.
        "call_expression" => {
            let callee = node.child_by_field_name("function")?;
            let name = node_text_simple(callee, lines);
            (callee.kind() == "identifier" && name == class).then_some(name)
        }
        // `~Widget();`, and `explicit Widget(int a);` where a specifier keyword kept
        // recovery from reaching the call reading above. Both name themselves through
        // the declarator chain, the same way they do when the class parses cleanly —
        // so a destructor renders exactly as written rather than reconstructed.
        "function_declarator" => {
            let inner = node.child_by_field_name("declarator")?;
            let name = c_declarator_name(inner, lines)?;
            (inner.kind() == "destructor_name" || name == class).then_some(name)
        }
        _ => None,
    }
}

/// True when a C/C++ `declaration` is really a macro invocation rather than a member.
///
/// A *zero-argument* macro invocation inside a class body — `GENERATED_BODY()` — parses as
/// a `declaration` with a `function_declarator` and **no `type`**, which is exactly how a
/// constructor or destructor parses too, since those have no return type either. The one
/// thing that separates them is the name: a constructor is named for its class, a
/// destructor is a `destructor_name`, and anything else with that shape is a macro.
///
/// Only that form. An argument-carrying macro (`DECLARE_DELEGATE(FOnHit);`) parses with the
/// macro name as the `type` and a `parenthesized_declarator`, so it never reaches the name
/// comparison below — it is excluded from outlines for unrelated reasons.
///
/// This matters beyond cosmetics. Treated as a member, such a macro becomes an "exported
/// symbol" of the header, and `tilth_deps` then reports every *other* file that invokes
/// the same macro as a dependent — a code-generating C++ framework makes that every
/// header in the project, none of which include the file.
///
/// Detection is on shape and the enclosing type's name only; no macro name is matched.
pub(crate) fn is_cpp_macro_invocation(node: tree_sitter::Node, lines: &[&str]) -> bool {
    if node.kind() != "declaration" || node.child_by_field_name("type").is_some() {
        return false;
    }
    let Some(declarator) = node.child_by_field_name("declarator") else {
        return false;
    };
    if declarator.kind() != "function_declarator" {
        return false;
    }
    // A destructor is a member, and is unambiguous.
    let inner = declarator.child_by_field_name("declarator");
    if inner.is_some_and(|d| d.kind() == "destructor_name") {
        return false;
    }
    let Some(name) = inner.and_then(|d| c_declarator_name(d, lines)) else {
        return false;
    };
    // A constructor is named for the type that encloses it. Every branch that cannot
    // establish that comparison answers "not a macro", because the cost is asymmetric:
    // a macro left in an outline is noise, a constructor dropped from one is a member
    // that vanishes from search, deps' exported symbols and blast radius.
    match enclosing_type_specifier(node) {
        // No enclosing type at all — a constructor is impossible here, so a typeless
        // declaration shaped like a call is a macro.
        None => true,
        Some(spec) => match spec.child_by_field_name("name") {
            // Anonymous type, or a class head tree-sitter repaired into something with
            // no `name` field. Nothing to compare against. Defensive rather than
            // observed: an anonymous type cannot have a constructor, and in the known
            // export-macro misparse the constructor becomes an `expression_statement`
            // that never reaches this predicate. Kept because the asymmetry above says
            // to keep the member whenever the comparison cannot be made.
            None => false,
            Some(name_node) => match trailing_type_identifier(name_node, lines) {
                Some(ty) => ty != name,
                // Unrecognised name shape.
                None => false,
            },
        },
    }
}

/// The nearest enclosing C/C++ type specifier node, if any.
fn enclosing_type_specifier(node: tree_sitter::Node) -> Option<tree_sitter::Node> {
    let mut cur = node.parent();
    while let Some(p) = cur {
        if SPECIFIER_KINDS.contains(&p.kind()) {
            return Some(p);
        }
        cur = p.parent();
    }
    None
}

/// The bare identifier at the end of a type name.
///
/// A specifier's `name` field is not always a plain `type_identifier`: an explicit or
/// partial specialization names a `template_type` (`Box<int>`, `Box<T*>`) and an
/// out-of-namespace definition names a `qualified_identifier` (`Outer::Inner`). Comparing
/// a constructor's bare `Box` or `Inner` against that whole text never matches, which
/// classified those constructors as macros and dropped them.
fn trailing_type_identifier(node: tree_sitter::Node, lines: &[&str]) -> Option<String> {
    let mut cur = node;
    for _ in 0..MAX_DECLARATOR_DEPTH {
        match cur.kind() {
            "type_identifier" | "identifier" => return Some(node_text_simple(cur, lines)),
            "template_type" | "qualified_identifier" => {
                cur = cur.child_by_field_name("name")?;
            }
            _ => return None,
        }
    }
    None
}

/// True when `node` is a `declaration` for a member template inside a class body —
/// `field_declaration_list` → `template_declaration` → `declaration`.
fn is_cpp_member_template_declaration(node: tree_sitter::Node) -> bool {
    node.parent().is_some_and(|p| {
        p.kind() == "template_declaration"
            && p.parent()
                .is_some_and(|g| g.kind() == "field_declaration_list")
    })
}

/// True when `node` carries an `extern` storage class. Such a declaration names
/// something defined in another translation unit, so it is never itself a definition.
fn has_extern_storage_class(node: tree_sitter::Node) -> bool {
    let mut cursor = node.walk();
    let found = node.children(&mut cursor).any(|c| {
        c.kind() == "storage_class_specifier" && c.child(0).is_some_and(|k| k.kind() == "extern")
    });
    found
}

/// True when any `ERROR` node sits directly inside `node`.
fn has_error_child(node: tree_sitter::Node) -> bool {
    let mut cursor = node.walk();
    let found = node.children(&mut cursor).any(|c| c.kind() == "ERROR");
    found
}

/// True when `node`'s C/C++ declarator chain contains a node of `kind`.
/// Walks the chain, so layers added by pointers, references and arrays are seen too.
///
/// Bounded by `MAX_DECLARATOR_DEPTH` for the same reason `c_declarator_name` is: the
/// nesting is input-controlled and `outline::generate` is a fuzz target.
fn declarator_chain_has_kind(node: tree_sitter::Node, kind: &str) -> bool {
    let mut current = inner_declarator(node);
    for _ in 0..MAX_DECLARATOR_DEPTH {
        let Some(n) = current else { return false };
        if n.kind() == kind {
            return true;
        }
        current = inner_declarator(n);
    }
    false
}

/// True when `node`'s C/C++ declarator chain declares a function — the marker that
/// separates a real definition (`class Foo bar() { … }`) from a misparsed class head.
pub(crate) fn declarator_chain_has_function(node: tree_sitter::Node) -> bool {
    declarator_chain_has_kind(node, "function_declarator")
}

/// Extract the name defined by a tree-sitter definition node.
///
/// Walks standard field names (`name`, `identifier`, `declarator`) and handles
/// nested declarators and export statements.
pub(crate) fn extract_definition_name(node: tree_sitter::Node, lines: &[&str]) -> Option<String> {
    // C/C++ wrap the declared name in a declarator chain rather than exposing a
    // `name` field, so resolve those before the generic field probe below. Without
    // this, the `declarator` arm of that probe returns the declarator's raw source
    // text — `"Holder::StaticWork()"` for `void Holder::StaticWork() {}`, parens and
    // qualifier included — which never equals the symbol being searched for. That is
    // why C and C++ function definitions did not resolve by name at all.
    //
    // Gating on the child's kind rather than on a language parameter keeps this
    // scoped. Three grammars expose a `declarator` field — c, cpp and java — but
    // Java's is a `variable_declarator`, which is not in `C_DECLARATOR_KINDS`, so it
    // keeps taking the generic path below. (`qualified_identifier` is also a Kotlin
    // kind, but Kotlin has no `declarator` field, so it is never reached from here.)
    // Python's `function_definition` and PHP's expose a `name` field and no
    // `declarator`, so they fall through untouched.
    if let Some(declarator) = node.child_by_field_name("declarator") {
        if C_DECLARATOR_KINDS.contains(&declarator.kind()) {
            if let Some(name) = c_declarator_name(declarator, lines) {
                return Some(name);
            }
        }
    }

    // A C++ class nested inside another class is a `field_declaration` whose `type`
    // is the inner `class_specifier` (`class Outer { class Inner { … }; };`). The
    // specifier itself sits one AST level deeper than the definition walk descends,
    // so name it from here.
    if node.kind() == "field_declaration" {
        if let Some(type_node) = node.child_by_field_name("type") {
            if is_bodied_specifier(type_node) {
                return extract_definition_name(type_node, lines);
            }
        }
    }

    // A C++ export macro in a class head misparses into a `function_definition`
    // that actually declares a class — resolve it to the class name.
    if let Some(name) = cpp_misparsed_class_name(node, lines) {
        return Some(name);
    }

    // Try standard field names
    for field in &["name", "identifier", "declarator"] {
        if let Some(child) = node.child_by_field_name(field) {
            let text = node_text_simple(child, lines);
            if !text.is_empty() {
                // For variable_declarator, get the identifier inside
                if child.kind().contains("declarator") {
                    if let Some(id) = child.child_by_field_name("name") {
                        return Some(node_text_simple(id, lines));
                    }
                }
                return Some(text);
            }
        }
    }

    // For export_statement, check the declaration child
    if node.kind() == "export_statement" {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if DEFINITION_KINDS.contains(&child.kind()) {
                return extract_definition_name(child, lines);
            }
        }
    }

    // JS/TS `lexical_declaration` and C# `variable_declaration` store the
    // identifier inside a `variable_declarator` child (field "declarations" /
    // unnamed children), not as a direct named field on the declaration node.
    // Walk children to find the first `variable_declarator` and pull its `name`.
    if node.kind() == "lexical_declaration" || node.kind() == "variable_declaration" {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "variable_declarator" {
                if let Some(name_node) = child.child_by_field_name("name") {
                    let text = node_text_simple(name_node, lines);
                    if !text.is_empty() {
                        return Some(text);
                    }
                }
            }
        }
    }

    None
}

/// Get the text of a single-line node from pre-split source lines.
///
/// Returns the text slice for single-line nodes, or the text from the start
/// column to end-of-line for multi-line nodes.
pub(crate) fn node_text_simple(node: tree_sitter::Node, lines: &[&str]) -> String {
    let row = node.start_position().row;
    let col_start = node.start_position().column;
    let end_row = node.end_position().row;
    if row < lines.len() && row == end_row {
        let col_end = node.end_position().column.min(lines[row].len());
        lines[row][col_start..col_end].to_string()
    } else if row < lines.len() {
        lines[row][col_start..].to_string()
    } else {
        String::new()
    }
}

/// Extract trait name from Rust `impl Trait for Type` node.
/// Returns None for inherent impls (no trait).
pub(crate) fn extract_impl_trait(node: tree_sitter::Node, lines: &[&str]) -> Option<String> {
    let trait_node = node.child_by_field_name("trait")?;
    Some(node_text_simple(trait_node, lines))
}

/// Extract implementing type from Rust `impl ... for Type` node.
pub(crate) fn extract_impl_type(node: tree_sitter::Node, lines: &[&str]) -> Option<String> {
    let type_node = node.child_by_field_name("type")?;
    Some(node_text_simple(type_node, lines))
}

/// Extract implemented interface names from TS/Java class declaration.
/// Walks `implements_clause` (TS) and `super_interfaces` (Java) children.
pub(crate) fn extract_implemented_interfaces(
    node: tree_sitter::Node,
    lines: &[&str],
) -> Vec<String> {
    let mut interfaces = Vec::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "implements_clause" || child.kind() == "super_interfaces" {
            let mut inner = child.walk();
            for ident in child.children(&mut inner) {
                if ident.kind().contains("identifier") {
                    let text = node_text_simple(ident, lines);
                    if !text.is_empty() {
                        interfaces.push(text);
                    }
                }
            }
        }
    }
    interfaces
}

// ---------------------------------------------------------------------------
// Elixir-specific definition helpers
// ---------------------------------------------------------------------------

/// Elixir call-node target identifiers that define named symbols.
/// This is the complete set used for definition detection in symbol search/index.
/// See also `ELIXIR_DEF_KEYWORDS` in `outline.rs` which is the subset of
/// function-like keywords (excludes container keywords like `defmodule`,
/// `defprotocol`, `defimpl`, `defstruct`, `defexception` that have their own
/// outline handling).
const ELIXIR_DEFINITION_TARGETS: &[&str] = &[
    "defmodule",
    "def",
    "defp",
    "defmacro",
    "defmacrop",
    "defguard",
    "defguardp",
    "defdelegate",
    "defstruct",
    "defexception",
    "defprotocol",
    "defimpl",
];

/// Find the `arguments` child of an Elixir `call` node.
/// In tree-sitter-elixir, `arguments` is a node kind, not a named field,
/// so `child_by_field_name("arguments")` doesn't work.
pub(crate) fn elixir_arguments(node: tree_sitter::Node) -> Option<tree_sitter::Node> {
    let mut cursor = node.walk();
    // Node is Copy (arena index) — the returned node survives cursor drop.
    let result = node.children(&mut cursor).find(|c| c.kind() == "arguments");
    result
}

/// Check if a tree-sitter node is an Elixir definition.
/// In Elixir all definitions are `call` nodes whose `target` identifier
/// is one of `defmodule`, `def`, `defp`, etc.
pub(crate) fn is_elixir_definition(node: tree_sitter::Node, lines: &[&str]) -> bool {
    if node.kind() != "call" {
        return false;
    }
    let Some(target) = node.child_by_field_name("target") else {
        return false;
    };
    let kw = node_text_simple(target, lines);
    ELIXIR_DEFINITION_TARGETS.contains(&kw.as_str())
}

/// Extract the defined name from an Elixir definition `call` node.
///
/// - `defmodule Foo.Bar do...end` → `"Foo.Bar"`
/// - `def greet(name) do...end`  → `"greet"`
/// - `defstruct [:a, :b]`       → `"defstruct"`
pub(crate) fn extract_elixir_definition_name(
    node: tree_sitter::Node,
    lines: &[&str],
) -> Option<String> {
    let target = node.child_by_field_name("target")?;
    let kw = node_text_simple(target, lines);
    let args = elixir_arguments(node)?;

    match kw.as_str() {
        "defmodule" | "defprotocol" | "defimpl" => {
            // First named child of arguments is the module/protocol alias.
            // For `defimpl Printable, for: User`, this returns "Printable" (the
            // protocol name), not "User" (the implementing type). Searching for
            // the protocol name will find both the protocol and all its impls.
            let mut cursor = args.walk();
            for child in args.children(&mut cursor) {
                if child.is_named() {
                    return Some(node_text_simple(child, lines));
                }
            }
            None
        }
        "def" | "defp" | "defmacro" | "defmacrop" | "defguard" | "defguardp" | "defdelegate" => {
            // First named child is:
            //   `call`              — normal: `def greet(name)`
            //   `identifier`        — no-arg: `def bar, do: :ok`
            //   `binary_operator`   — guard:  `def foo(x) when x > 0`
            let mut cursor = args.walk();
            for child in args.children(&mut cursor) {
                if !child.is_named() {
                    continue;
                }
                return elixir_extract_func_head_name(child, lines);
            }
            None
        }
        // In Elixir, a struct IS its enclosing module (`%MyModule{}`), and only
        // one struct per module is allowed. There's no standalone struct name to
        // extract, so we index the keyword itself. Search for the struct by its
        // module name instead.
        "defstruct" | "defexception" => Some(kw.clone()),
        _ => None,
    }
}

/// Extract function name from the first argument of a `def`/`defp`/`defmacro` call.
///
/// The first argument can be:
/// - `call` node: `def greet(name)` → target is `greet`
/// - `identifier` node: `def bar, do: :ok` → text is `bar`
/// - `binary_operator` with `when`: `def foo(x) when x > 0` → unwrap left, then recurse
pub(crate) fn elixir_extract_func_head_name(
    node: tree_sitter::Node,
    lines: &[&str],
) -> Option<String> {
    match node.kind() {
        "call" => node
            .child_by_field_name("target")
            .map(|t| node_text_simple(t, lines)),
        "identifier" => Some(node_text_simple(node, lines)),
        "binary_operator" => {
            // Guard clause: `foo(x) when x > 0` → left is the function head
            let left = node.child_by_field_name("left")?;
            elixir_extract_func_head_name(left, lines)
        }
        _ => None,
    }
}

/// Semantic weight for Elixir definition keywords.
pub(crate) fn elixir_definition_weight(node: tree_sitter::Node, lines: &[&str]) -> u16 {
    let Some(target) = node.child_by_field_name("target") else {
        return 50;
    };
    let kw = node_text_simple(target, lines);
    match kw.as_str() {
        "defmodule" | "defprotocol" | "def" | "defp" | "defmacro" | "defmacrop" | "defguard"
        | "defguardp" | "defdelegate" => 100,
        "defimpl" => 90,
        "defstruct" | "defexception" => 80,
        _ => 50,
    }
}

/// Semantic weight for a definition *node*, for the cases where the kind string alone
/// is not enough to say how a definition should rank.
///
/// Two C/C++ kinds each cover two very different things, and weighting them by kind
/// forced one answer on both:
///
///   * `field_declaration` is `void Work();` *and* `int Count;`. The first is a method
///     declaration whose real definition is usually out-of-line at 100, so 70 keeps it
///     just below. The second is a data member — a variable — and weighting it like a
///     near-type meant a common member name competed with the type and function
///     definitions in a definitions-first view. 40 is the plain-variable tier, and it
///     also falls under `stratum_for_display`'s 60 cutoff, so a data member sorts below
///     real definitions and is dropped before one when the match cap bites.
///   * `declaration` is a macro-misparsed class head (a type, 100) *or* an out-of-line
///     static member definition (data, 80 — the tier `static_item` and
///     `const_declaration` use in other languages).
pub(crate) fn definition_weight_for(node: tree_sitter::Node) -> u16 {
    match node.kind() {
        "field_declaration" => member_weight(node),
        "declaration" => {
            if is_cpp_misparsed_class_head(node) {
                100
            } else if is_cpp_member_template_declaration(node) {
                // A member template — `template <typename T> void Apply(T V);` — is a
                // member declaration that happens to arrive as a `declaration` rather
                // than a `field_declaration`. Weigh it as the member it is, not as the
                // out-of-line data definition the arm below covers.
                member_weight(node)
            } else {
                80
            }
        }
        kind => definition_weight(kind),
    }
}

/// Weight for a C/C++ class member: 70 for a method declaration, 40 for data.
///
/// A `parenthesized_declarator` in the chain means the function part belongs to the
/// *type*, not to the declaration — `void (*Cb)(int);` is a function-pointer data member,
/// not a method — so it is data despite containing a `function_declarator`.
fn member_weight(node: tree_sitter::Node) -> u16 {
    let is_method = declarator_chain_has_function(node)
        && !declarator_chain_has_kind(node, "parenthesized_declarator");
    if is_method {
        70
    } else {
        40
    }
}

/// Semantic weight for definition kinds. Primary declarations rank highest.
///
/// Prefer `definition_weight_for` when a node is available — two C/C++ kinds need it.
pub(crate) fn definition_weight(kind: &str) -> u16 {
    match kind {
        "function_declaration"
        | "function_definition"
        | "function_item"
        | "method_definition"
        | "method_declaration"
        | "class_declaration"
        | "class_definition"
        | "struct_item"
        | "interface_declaration"
        | "trait_declaration"
        | "trait_item"
        | "enum_item"
        | "enum_declaration"
        | "type_item"
        | "type_declaration"
        | "type_definition"
        | "class_specifier"
        | "struct_specifier"
        | "union_specifier"
        | "enum_specifier"
        | "alias_declaration"
        // `declaration` and `field_declaration` are listed here as a safe default only.
        // Production never reads their value from this table: `definition_weight_for`
        // intercepts both, because each covers two constructs that must rank differently
        // (a class head vs an out-of-line data definition; a method declaration vs a data
        // member). Anything reaching these arms is a caller that bypassed
        // `definition_weight_for` — prefer that function whenever a node is available.
        | "declaration"
        | "decorated_definition" => 100,
        "impl_item" | "object_declaration" => 90,
        "const_item" | "const_declaration" | "static_item" => 80,
        "field_declaration" | "mod_item" | "namespace_definition" | "property_declaration" => 70,
        "lexical_declaration" | "variable_declaration" => 40,
        "variable_assignment" => 60,
        "export_statement" => 30,
        _ => 50,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang::outline::outline_language;
    use crate::types::Lang;

    #[test]
    fn definition_weight_covers_every_tier() {
        // 100 — primary declarations, one per source language shape
        // (Rust function_item/enum_item, TS class_declaration/interface_declaration, Python decorated_definition).
        assert_eq!(definition_weight("function_item"), 100);
        assert_eq!(definition_weight("class_declaration"), 100);
        assert_eq!(definition_weight("interface_declaration"), 100);
        assert_eq!(definition_weight("enum_item"), 100);
        assert_eq!(definition_weight("decorated_definition"), 100);
        // 90 — impls / object-like declarations (Rust impl_item, Kotlin object_declaration).
        assert_eq!(definition_weight("impl_item"), 90);
        assert_eq!(definition_weight("object_declaration"), 90);
        // 80 — const/static.
        assert_eq!(definition_weight("const_item"), 80);
        assert_eq!(definition_weight("static_item"), 80);
        // 70 — module/namespace/property.
        assert_eq!(definition_weight("mod_item"), 70);
        assert_eq!(definition_weight("property_declaration"), 70);
        // 60 — Bash top-level assignment (special-cased above the 40 tier).
        assert_eq!(definition_weight("variable_assignment"), 60);
        // 40 — plain variable declarations (JS/TS lexical_declaration, C#/Kotlin variable_declaration).
        assert_eq!(definition_weight("lexical_declaration"), 40);
        assert_eq!(definition_weight("variable_declaration"), 40);
        // 30 — export wrapper (unwrapped recursively by extract_definition_name).
        assert_eq!(definition_weight("export_statement"), 30);
        // 50 — unrecognized kind falls to the default tier, not 0.
        assert_eq!(definition_weight("comment"), 50);
        // C/C++ type definitions rank with the other primary declarations, so a
        // class definition outranks its own usages.
        for kind in [
            "class_specifier",
            "struct_specifier",
            "union_specifier",
            "enum_specifier",
            "alias_declaration",
            "type_definition",
        ] {
            assert_eq!(definition_weight(kind), 100, "{kind} should be a 100-tier");
        }
        // `template_declaration` is intentionally NOT a definition kind — it is a
        // transparent wrapper, so its weight is never consulted.
        assert!(!DEFINITION_KINDS.contains(&"template_declaration"));
        // A C++ member *declaration* sits below the out-of-line definition (100)
        // that usually accompanies it, but above the doc-heading tier (30).
        assert_eq!(definition_weight("field_declaration"), 70);
    }

    /// `field_declaration` and `declaration` each cover two different things, so their
    /// weight cannot come from the kind string alone. A data member weighted like a
    /// near-type crowded real type and function definitions out of a definitions-first
    /// view whenever a common member name was searched.
    #[test]
    fn definition_weight_for_separates_members_from_methods() {
        let src = "class Holder {\npublic:\n\tvoid Work();\n\tint Count;\n};\n";
        let tree = parse(src, Lang::Cpp);
        let mut cursor = tree.root_node().walk();
        let mut fields = Vec::new();
        let mut stack = vec![tree.root_node()];
        while let Some(n) = stack.pop() {
            if n.kind() == "field_declaration" {
                fields.push(n);
            }
            stack.extend(n.children(&mut cursor));
        }
        fields.sort_by_key(|n| n.start_position().row);
        assert_eq!(fields.len(), 2, "expected the method and the data member");

        // `void Work();` — a method declaration. Its real definition is usually
        // out-of-line at 100, so this sits just below.
        assert_eq!(definition_weight_for(fields[0]), 70);
        // `int Count;` — a data member, weighted as the variable it is. Below
        // `stratum_for_display`'s 60 cutoff, so it never outranks a real definition.
        assert_eq!(definition_weight_for(fields[1]), 40);
        assert!(definition_weight_for(fields[1]) < 60);
    }

    #[test]
    fn definition_weight_for_separates_class_heads_from_static_members() {
        // Multiple inheritance on purpose: that is the recovery shape that actually
        // parses as a `declaration`, so this exercises the branch. The single-base form
        // parses as a `function_definition` and would take its 100 from the kind table,
        // testing nothing about `definition_weight_for`.
        let head = "class API Widget : public B1, public B2 { public: void W(); };\n";
        let tree = parse(head, Lang::Cpp);
        let node = tree.root_node().named_child(0).expect("top-level node");
        assert_eq!(
            node.kind(),
            "declaration",
            "fixture must hit the declaration arm"
        );
        assert_eq!(
            definition_weight_for(node),
            100,
            "a macro-misparsed class head is a type definition"
        );

        let member = "int Widget::sCount = 0;\n";
        let tree = parse(member, Lang::Cpp);
        let node = tree.root_node().named_child(0).expect("declaration");
        assert_eq!(
            definition_weight_for(node),
            80,
            "an out-of-line static member is data — the tier static_item uses"
        );
    }

    /// A member template is a member declaration that arrives as a `declaration` rather
    /// than a `field_declaration`, so it must be weighed as the member it is — otherwise
    /// a templated method lands in the out-of-line data tier (80) and outranks the plain
    /// method declarations beside it.
    #[test]
    fn definition_weight_for_member_template_is_weighed_as_a_member() {
        let src = "class Holder {\npublic:\ntemplate <typename T>\nvoid Apply(T V);\n};\n";
        let tree = parse(src, Lang::Cpp);
        let node = find_by_kind(tree.root_node(), "declaration");
        assert_eq!(
            definition_weight_for(node),
            70,
            "a member template is a method"
        );
    }

    /// A `function_declarator` inside a `parenthesized_declarator` belongs to the *type*,
    /// not to the declaration: `void (*Cb)(int);` is a function-pointer data member, not a
    /// method. Callback members are common in C-ish headers.
    #[test]
    fn definition_weight_for_function_pointer_member_is_data() {
        let cases: &[(&str, u16)] = &[
            ("class H { void (*Cb)(int); };\n", 40),
            ("class H { void Work(); };\n", 70),
            ("class H { int Arr[10]; };\n", 40),
            ("class H { Widget* Get(); };\n", 70),
        ];
        for (src, want) in cases {
            let tree = parse(src, Lang::Cpp);
            let node = find_by_kind(tree.root_node(), "field_declaration");
            assert_eq!(definition_weight_for(node), *want, "{src:?}");
        }
    }

    #[test]
    fn definition_weight_for_falls_through_to_the_kind_table() {
        let src = "class Widget { int x; };\n";
        let tree = parse(src, Lang::Cpp);
        let node = find_by_kind(tree.root_node(), "class_specifier");
        assert_eq!(
            definition_weight_for(node),
            definition_weight("class_specifier")
        );
    }

    /// The kind strings in `DEFINITION_KINDS` are matched against every grammar
    /// tilth ships, so a string that means one thing in tree-sitter-cpp and another
    /// elsewhere silently changes behaviour for that other language. This pins which
    /// grammars own each C/C++ kind that was added for C++ type support, and — the
    /// actual guard — that `field_declaration` is *not* in the global list: it is a
    /// class member in C/C++ but a struct/record field in Rust, Go, Java and C#, so
    /// registering it globally would report every Rust and Go struct field as a
    /// top-level definition.
    #[test]
    fn definition_kinds_are_not_ambiguous_across_grammars() {
        // Every grammar that could plausibly own one of these strings must be in this
        // list, or the test cannot see the collision it claims to rule out.
        let grammars: Vec<(&str, tree_sitter::Language)> = vec![
            ("rust", tree_sitter_rust::LANGUAGE.into()),
            ("go", tree_sitter_go::LANGUAGE.into()),
            ("java", tree_sitter_java::LANGUAGE.into()),
            ("csharp", tree_sitter_c_sharp::LANGUAGE.into()),
            ("c", tree_sitter_c::LANGUAGE.into()),
            ("cpp", tree_sitter_cpp::LANGUAGE.into()),
            ("scala", tree_sitter_scala::LANGUAGE.into()),
            ("php", tree_sitter_php::LANGUAGE_PHP.into()),
            ("kotlin", tree_sitter_kotlin_ng::LANGUAGE.into()),
            (
                "typescript",
                tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            ),
            ("javascript", tree_sitter_javascript::LANGUAGE.into()),
            ("python", tree_sitter_python::LANGUAGE.into()),
            ("swift", tree_sitter_swift::LANGUAGE.into()),
        ];
        let owners = |kind: &str| -> Vec<&str> {
            grammars
                .iter()
                .filter(|(_, l)| l.id_for_node_kind(kind, true) != 0)
                .map(|(n, _)| *n)
                .collect()
        };

        // `type_definition` is the one added kind that is NOT C/C++-exclusive: Scala
        // owns it too, for `type Alias = String`. Registering it is intentional and
        // makes Scala type aliases resolve as definitions, which they are — but it is
        // a real cross-language behaviour change, so it is called out here rather than
        // swept into the "C/C++-exclusive" claim below.
        assert!(DEFINITION_KINDS.contains(&"type_definition"));
        let td = owners("type_definition");
        assert!(
            td.contains(&"scala"),
            "expected scala to share type_definition (owners: {td:?}); the Scala \
             behaviour change this kind causes is deliberate — see the comment above"
        );

        // `field_declaration` is shared, hence language-scoped rather than global.
        assert!(
            !DEFINITION_KINDS.contains(&"field_declaration"),
            "field_declaration must stay out of the global list — it is a struct \
             field in Rust/Go/Java/C#, not a definition"
        );
        assert!(C_FAMILY_DEFINITION_KINDS.contains(&"field_declaration"));
        let shared = owners("field_declaration");
        for lang in ["rust", "go", "java", "csharp"] {
            assert!(
                shared.contains(&lang),
                "expected {lang} to define field_declaration (owners: {shared:?}); if a \
                 grammar dropped it, revisit whether the language gate is still needed"
            );
        }

        // The kinds added to the global list are C/C++-exclusive, so registering them
        // cannot change any other language's results.
        for kind in [
            "class_specifier",
            "struct_specifier",
            "union_specifier",
            "enum_specifier",
            "alias_declaration",
        ] {
            assert!(
                DEFINITION_KINDS.contains(&kind),
                "{kind} should be registered"
            );
            let o = owners(kind);
            assert!(
                o.iter().all(|g| *g == "c" || *g == "cpp"),
                "{kind} is not C/C++-exclusive (owners: {o:?}) — registering it globally \
                 would change behaviour for those grammars"
            );
        }
    }

    /// Parse `src` with `lang`'s grammar and return the owned tree.
    fn parse(src: &str, lang: Lang) -> tree_sitter::Tree {
        let language = outline_language(lang).expect("grammar available for test language");
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&language).expect("grammar loads");
        parser.parse(src, None).expect("parse succeeds")
    }

    /// Depth-first search for the first descendant node of the given kind.
    fn find_by_kind<'a>(root: tree_sitter::Node<'a>, kind: &str) -> tree_sitter::Node<'a> {
        let mut cursor = root.walk();
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            if node.kind() == kind {
                return node;
            }
            stack.extend(node.children(&mut cursor));
        }
        panic!("no {kind} node found in parsed tree");
    }

    #[test]
    fn extract_definition_name_rust_function_item() {
        let src = "fn greet(name: &str) -> String { name.to_string() }\n";
        let tree = parse(src, Lang::Rust);
        let lines: Vec<&str> = src.lines().collect();
        let node = find_by_kind(tree.root_node(), "function_item");
        assert_eq!(
            extract_definition_name(node, &lines),
            Some("greet".to_string())
        );
    }

    #[test]
    fn extract_definition_name_python_class_definition() {
        let src = "class Widget:\n    pass\n";
        let tree = parse(src, Lang::Python);
        let lines: Vec<&str> = src.lines().collect();
        let node = find_by_kind(tree.root_node(), "class_definition");
        assert_eq!(
            extract_definition_name(node, &lines),
            Some("Widget".to_string())
        );
    }

    #[test]
    fn extract_definition_name_unwraps_export_statement() {
        // export_statement has no "name"/"identifier"/"declarator" field of its
        // own — extract_definition_name must recurse into the wrapped
        // function_declaration to find the name (the node.kind() == "export_statement"
        // branch).
        let src = "export function handler() {}\n";
        let tree = parse(src, Lang::TypeScript);
        let lines: Vec<&str> = src.lines().collect();
        let node = find_by_kind(tree.root_node(), "export_statement");
        assert_eq!(
            extract_definition_name(node, &lines),
            Some("handler".to_string())
        );
    }

    #[test]
    fn extract_definition_name_walks_lexical_declaration_declarator() {
        // lexical_declaration stores its identifier inside a child
        // variable_declarator, not as a direct field on the declaration node —
        // exercises the dedicated child-walk branch.
        let src = "const total = 42;\n";
        let tree = parse(src, Lang::TypeScript);
        let lines: Vec<&str> = src.lines().collect();
        let node = find_by_kind(tree.root_node(), "lexical_declaration");
        assert_eq!(
            extract_definition_name(node, &lines),
            Some("total".to_string())
        );
    }

    #[test]
    fn extract_definition_name_returns_none_when_no_name_field_present() {
        // impl_item has no "name"/"identifier"/"declarator" field and isn't
        // handled by any of the special-cased branches — must fall through to
        // None rather than panic or return an empty string.
        let src = "impl Widget {}\n";
        let tree = parse(src, Lang::Rust);
        let lines: Vec<&str> = src.lines().collect();
        let node = find_by_kind(tree.root_node(), "impl_item");
        assert_eq!(extract_definition_name(node, &lines), None);
    }

    // -- C++ type support -------------------------------------------------
    //
    // Before these, `DEFINITION_KINDS` held no tree-sitter-cpp type node at all, so
    // no C++ class, struct, enum or typedef resolved — and because C/C++ name their
    // declarations through a declarator chain rather than a `name` field, even plain
    // *functions* resolved as `"Name()"` and so never matched a search.

    /// Name the definition of kind `kind` in `src`, parsed as C++.
    fn cpp_name(src: &str, kind: &str) -> Option<String> {
        let tree = parse(src, Lang::Cpp);
        let lines: Vec<&str> = src.lines().collect();
        let node = find_by_kind(tree.root_node(), kind);
        extract_definition_name(node, &lines)
    }

    #[test]
    fn extract_definition_name_cpp_plain_class() {
        assert_eq!(
            cpp_name(
                "class Widget { public: void Work(); };\n",
                "class_specifier"
            ),
            Some("Widget".to_string())
        );
    }

    #[test]
    fn extract_definition_name_cpp_class_with_final_and_base_clause() {
        // The form the original bug report blamed on a `final` parse failure. It
        // parses cleanly as a `class_specifier`; the real reason it never resolved is
        // that the kind was absent from `DEFINITION_KINDS`.
        let src = "class Base {};\nclass Widget final : public Base {};\n";
        let tree = parse(src, Lang::Cpp);
        let lines: Vec<&str> = src.lines().collect();
        let node = tree
            .root_node()
            .named_child(1)
            .expect("second top-level declaration");
        assert_eq!(node.kind(), "class_specifier");
        assert_eq!(
            extract_definition_name(node, &lines),
            Some("Widget".to_string())
        );
    }

    #[test]
    fn extract_definition_name_cpp_struct_union_and_scoped_enum() {
        assert_eq!(
            cpp_name("struct Point { int X; };\n", "struct_specifier"),
            Some("Point".to_string())
        );
        assert_eq!(
            cpp_name("union Value { int I; float F; };\n", "union_specifier"),
            Some("Value".to_string())
        );
        assert_eq!(
            cpp_name("enum Color { Red, Blue };\n", "enum_specifier"),
            Some("Color".to_string())
        );
        // `enum class` — the same node kind, with a `base` field for the underlying type.
        assert_eq!(
            cpp_name("enum class Mode : uint8_t { On, Off };\n", "enum_specifier"),
            Some("Mode".to_string())
        );
    }

    #[test]
    fn extract_definition_name_cpp_typedef_and_using_alias() {
        // `typedef` names the alias in a `declarator` field; `using` in a `name` field.
        assert_eq!(
            cpp_name("typedef unsigned int Handle;\n", "type_definition"),
            Some("Handle".to_string())
        );
        assert_eq!(
            cpp_name("using Callback = void(*)(int);\n", "alias_declaration"),
            Some("Callback".to_string())
        );
    }

    /// `template_declaration` is a transparent wrapper, not a definition: it carries no
    /// name of its own and encloses a declaration that is already a definition kind.
    /// Registering both reported one template *twice* whenever the `template <…>`
    /// clause sat on its own line — the normal spelling in real C++ — so only the inner
    /// declaration counts, and it is the inner one that carries the name.
    #[test]
    fn template_declaration_is_a_transparent_wrapper_not_a_definition() {
        let src = "template <typename T>\nclass Vector { public: void Push(T V); };\n";
        let tree = parse(src, Lang::Cpp);
        let lines: Vec<&str> = src.lines().collect();

        let wrapper = tree.root_node().named_child(0).expect("template");
        assert_eq!(wrapper.kind(), "template_declaration");
        assert!(
            !is_definition_node(wrapper, Some(Lang::Cpp)),
            "the wrapper must not register a definition of its own"
        );

        let inner = find_by_kind(tree.root_node(), "class_specifier");
        assert!(is_definition_node(inner, Some(Lang::Cpp)));
        assert_eq!(
            extract_definition_name(inner, &lines),
            Some("Vector".to_string())
        );
    }

    #[test]
    fn extract_definition_name_cpp_nested_class_via_member_declaration() {
        // A class nested in another class is a `field_declaration` wrapping the inner
        // `class_specifier`, which sits deeper than the definition walk descends —
        // so the member node itself has to resolve to the inner type's name.
        let src = "class Outer { public: class Inner { void Deep(); }; };\n";
        let tree = parse(src, Lang::Cpp);
        let lines: Vec<&str> = src.lines().collect();
        let node = find_by_kind(tree.root_node(), "field_declaration");
        assert_eq!(
            extract_definition_name(node, &lines),
            Some("Inner".to_string())
        );
    }

    #[test]
    fn extract_definition_name_cpp_member_declaration_and_out_of_line_definition() {
        // A member declared in a header is a `field_declaration`; its definition in
        // the .cpp is a `function_definition` with a qualified declarator. Both must
        // resolve to the bare member name, matching how every other language names
        // methods (a Rust `impl` method resolves as `bar`, not `Foo::bar`).
        assert_eq!(
            cpp_name(
                "class Holder { void MemberWork(); };\n",
                "field_declaration"
            ),
            Some("MemberWork".to_string())
        );
        assert_eq!(
            cpp_name("void Holder::MemberWork() {}\n", "function_definition"),
            Some("MemberWork".to_string())
        );
        // A free function, which resolved as `"Run()"` before the declarator walk.
        assert_eq!(
            cpp_name("void Run() {}\n", "function_definition"),
            Some("Run".to_string())
        );
        // Pointer return type — the declarator chain has an extra layer.
        assert_eq!(
            cpp_name(
                "int* Holder::Buffer() { return nullptr; }\n",
                "function_definition"
            ),
            Some("Buffer".to_string())
        );
    }

    #[test]
    fn is_definition_node_rejects_bodyless_specifier() {
        // A specifier with no body is an elaborated type specifier — a forward
        // declaration or a type reference, not a definition. Counting them would
        // report a definition at every forward declaration in every header.
        let src = "class Fwd;\nclass Fwd* Global;\nclass Real { int X; };\n";
        let tree = parse(src, Lang::Cpp);
        let root = tree.root_node();

        let fwd = root.named_child(0).expect("forward declaration");
        assert_eq!(fwd.kind(), "class_specifier");
        assert!(
            !is_definition_node(fwd, Some(Lang::Cpp)),
            "`class Fwd;` is a forward declaration, not a definition"
        );

        // `class Fwd* Global;` — the specifier inside an ordinary variable declaration.
        let decl = root.named_child(1).expect("variable declaration");
        assert_eq!(decl.kind(), "declaration");
        assert!(
            !is_definition_node(decl, Some(Lang::Cpp)),
            "a variable of an elaborated type is not a definition"
        );

        let real = root.named_child(2).expect("real class");
        assert!(
            is_definition_node(real, Some(Lang::Cpp)),
            "a class with a body is a definition"
        );
    }

    #[test]
    fn is_definition_node_scopes_field_declaration_to_c_family() {
        // Same kind string, opposite answers: a C++ class member is a definition, a
        // Rust struct field is not.
        let cpp_src = "class Holder { int Count; };\n";
        let cpp_tree = parse(cpp_src, Lang::Cpp);
        let cpp_field = find_by_kind(cpp_tree.root_node(), "field_declaration");
        assert!(is_definition_node(cpp_field, Some(Lang::Cpp)));

        let rust_src = "struct Holder { count: u32 }\n";
        let rust_tree = parse(rust_src, Lang::Rust);
        let rust_field = find_by_kind(rust_tree.root_node(), "field_declaration");
        assert!(
            !is_definition_node(rust_field, Some(Lang::Rust)),
            "a Rust struct field must not be reported as a definition"
        );
    }

    /// An export macro between `class` and the class name — how virtually every
    /// Windows C++ library spells dllexport — makes tree-sitter-cpp read the macro as
    /// the class name and the real name as a declarator. Which child ends up holding
    /// the real name depends on how error recovery repaired the head, so all the
    /// shapes that occur in practice are pinned here.
    #[test]
    fn cpp_misparsed_class_name_covers_every_macro_head_shape() {
        let cases: &[(&str, &str)] = &[
            ("class API Widget { public: void W(); };", "Widget"),
            ("class API Widget final { public: void W(); };", "Widget"),
            (
                "class API Widget : public Parent { public: void W(); };",
                "Widget",
            ),
            (
                "class API Widget final : public Parent { public: void W(); };",
                "Widget",
            ),
            // Same construct as the line above, but a longer base-class name tips
            // recovery into a different repair: it swallows `Widget final : public`
            // into an ERROR and leaves the *base* class as the declarator. Reading in
            // source order is what makes both shapes work, and this case is why
            // trusting any single recovery shape is not an option.
            (
                "class API Widget final : public VeryLongBaseClassNameHere { public: void W(); };",
                "Widget",
            ),
            // Multiple inheritance misparses into a `declaration`, not a
            // `function_definition`.
            (
                "class API Widget : public P1, public P2 { public: void W(); };",
                "Widget",
            ),
            ("struct API Point { int X; };", "Point"),
        ];
        for (src, expected) in cases {
            let owned = format!("{src}\n");
            let tree = parse(&owned, Lang::Cpp);
            let lines: Vec<&str> = owned.lines().collect();
            let node = tree.root_node().named_child(0).expect("a top-level node");
            assert_eq!(
                cpp_misparsed_class_name(node, &lines).as_deref(),
                Some(*expected),
                "wrong name for {src:?} (node kind {})",
                node.kind()
            );
            assert!(
                is_definition_node(node, Some(Lang::Cpp)),
                "{src:?} should be a definition"
            );
        }
    }

    /// Guards the premise of the test above: recovery really does repair these heads
    /// differently, so reading the name in source order is doing work that trusting
    /// any one shape would not. Purely renaming a fixture could otherwise collapse the
    /// cases onto one shape and leave that test passing while covering less.
    #[test]
    fn macro_class_head_recovery_shapes_are_genuinely_different() {
        // Child ordering of the misparsed node, as `field:kind` pairs.
        fn shape(src: &str) -> (String, Vec<String>) {
            let owned = format!("{src}\n");
            let tree = parse(&owned, Lang::Cpp);
            let node = tree.root_node().named_child(0).expect("a top-level node");
            let mut cursor = node.walk();
            let mut children = Vec::new();
            if cursor.goto_first_child() {
                loop {
                    let field = cursor
                        .field_name()
                        .map(|f| format!("{f}:"))
                        .unwrap_or_default();
                    children.push(format!("{field}{}", cursor.node().kind()));
                    if !cursor.goto_next_sibling() {
                        break;
                    }
                }
            }
            (node.kind().to_string(), children)
        }

        let (short_kind, short) =
            shape("class API Widget final : public Parent { public: void W(); };");
        let (long_kind, long) = shape(
            "class API Widget final : public VeryLongBaseClassNameHere { public: void W(); };",
        );
        let (multi_kind, _) =
            shape("class API Widget : public P1, public P2 { public: void W(); };");

        // Identical construct, different base-class name: the short form leaves the
        // real class name as the `declarator`, the long form buries it in an `ERROR`
        // and promotes the *base* class to `declarator` instead.
        assert_eq!(short_kind, "function_definition");
        assert_eq!(long_kind, "function_definition");
        assert_ne!(
            short, long,
            "the two base-class spellings must still produce different repairs, or the \
             source-order name lookup is no longer being exercised"
        );
        let idx = |v: &[String], k: &str| v.iter().position(|c| c == k);
        assert!(
            idx(&short, "declarator:identifier") < idx(&short, "ERROR"),
            "short form should keep the class name as the declarator: {short:?}"
        );
        assert!(
            idx(&long, "ERROR") < idx(&long, "declarator:identifier"),
            "long form should bury the class name in the ERROR: {long:?}"
        );
        // Multiple inheritance is a third shape entirely.
        assert_eq!(multi_kind, "declaration");
    }

    /// Tripwire. Recovering a misparsed class's constructors and destructors means
    /// pattern-matching tree-sitter-cpp's error recovery, which is not a stable
    /// contract — the head repair already differs with the length of the base-class
    /// name. This pins the body shapes `misparsed_member_name` relies on so a grammar
    /// bump fails here, loudly, instead of silently dropping members again.
    ///
    /// Full *paths* from the class body down to each matched node, not a set of
    /// `parent>child` kind pairs: the walk in `push_misparsed_members` descends only
    /// `ERROR`, and `enclosing_misparsed_class_name` climbs at most
    /// `MISPARSED_BODY_DEPTH`. A repair that kept every node kind but nested them one
    /// level deeper, or introduced a wrapper kind between them, would break the fix
    /// while leaving a kind-pair assertion green.
    ///
    /// If this fires: re-dump the trees for these sources and update
    /// `misparsed_member_name` (and this test) to the new shapes. The behaviour to
    /// restore is `cpp_outline_export_macro_parity_across_recovery_shapes`.
    #[test]
    fn misparsed_class_body_repair_shapes_are_unchanged() {
        /// Root-to-leaf kind paths for every node in the tree, as `"a>b>c"`.
        fn paths(src: &str) -> Vec<String> {
            fn walk(node: tree_sitter::Node, prefix: &str, out: &mut Vec<String>) {
                let here = if prefix.is_empty() {
                    node.kind().to_string()
                } else {
                    format!("{prefix}>{}", node.kind())
                };
                out.push(here.clone());
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    walk(child, &here, out);
                }
            }
            let tree = parse(src, Lang::Cpp);
            let mut out = Vec::new();
            walk(tree.root_node(), "", &mut out);
            out
        }

        /// Asserts exactly one path ends with `suffix`, so a shape cannot be satisfied
        /// by two unrelated nodes elsewhere in the tree.
        fn pin(paths: &[String], suffix: &str, what: &str) {
            let hits: Vec<&String> = paths.iter().filter(|p| p.ends_with(suffix)).collect();
            assert_eq!(
                hits.len(),
                1,
                "{what}: expected exactly one path ending {suffix:?}, found {hits:?}"
            );
        }

        // The head misparses into a `function_definition` whose body is a statement
        // block rather than a `field_declaration_list`. Everything below follows from
        // that: statements, not member declarations.
        let declared = paths(
            "class API Widget : public Base\n{\npublic:\n\tWidget();\n\t~Widget();\n\tint Value;\n};\n",
        );
        // A constructor has no return type, so recovery reads it as a call — under the
        // access specifier's `labeled_statement`, which `collect_member` flattens.
        pin(
            &declared,
            "function_definition>compound_statement>labeled_statement>expression_statement>call_expression",
            "declared constructor",
        );
        // A destructor is left as a bare declarator inside an ERROR, a *direct* child
        // of the body — one level, which is what the ERROR recursion assumes.
        pin(
            &declared,
            "function_definition>compound_statement>ERROR>function_declarator>destructor_name",
            "declared destructor",
        );
        // A data member stays a `declaration`, which is why it needs reclassifying from
        // local variable to property rather than recovering. Two `parent()` steps from
        // here to the class head is what `enclosing_misparsed_class_name` walks.
        pin(
            &declared,
            "function_definition>compound_statement>declaration",
            "data member",
        );

        // A specifier keyword blocks the call reading and leaves the constructor as a
        // declarator instead — the second shape `misparsed_member_name` handles.
        let specified = paths("class API Widget\n{\n\texplicit Widget(int A);\n};\n");
        pin(
            &specified,
            "compound_statement>ERROR>function_declarator>identifier",
            "`explicit` constructor",
        );

        // `= default` wraps the call one level deeper, in an assignment_expression —
        // the reason `statement_member_name` looks past the statement's own children.
        let defaulted =
            paths("class API Widget\n{\npublic:\n\tWidget() = default;\n\tint Value;\n};\n");
        pin(
            &defaulted,
            "labeled_statement>expression_statement>assignment_expression>call_expression",
            "`= default` constructor",
        );

        // `constexpr` leaves the class name in the `type` field with a zero-width
        // declarator — no call and no usable declarator name, hence `qualified_ctor_name`.
        let qualified = paths("class API Widget\n{\npublic:\n\tconstexpr Widget();\n};\n");
        pin(
            &qualified,
            "labeled_statement>declaration>parenthesized_declarator>identifier",
            "`constexpr` constructor",
        );

        // A nested type and the constructor after it share one `declaration`, which is
        // why the constructor has to be emitted separately, after the type.
        let nested =
            paths("class API Outer\n{\npublic:\n\tclass Inner { int X; };\n\tOuter();\n};\n");
        pin(
            &nested,
            "labeled_statement>declaration>function_declarator>identifier",
            "constructor after a nested type",
        );
    }

    /// An attribute macro sits between the type and the *variable* name here, not
    /// between `class` and the type name — an ordinary variable declaration that
    /// produces the identical bodyless-specifier-plus-`ERROR` shape as a misparsed
    /// class head. What separates them is the brace body: defining a type has one,
    /// referencing a type does not. Without that check these registered a class
    /// definition named after the macro (or, for `MACRO(1)`, emitted the macro as the
    /// symbol while naming the *variable* as the class).
    #[test]
    fn cpp_misparsed_class_name_rejects_macro_attributed_variables() {
        for src in [
            "class Foo PACKED_ATTR bar;\n",
            "struct FVector ALIGN16 Position;\n",
            "enum EMode DEPRECATED value;\n",
            "struct S MACRO(1) inst;\n",
        ] {
            let tree = parse(src, Lang::Cpp);
            let lines: Vec<&str> = src.lines().collect();
            let node = tree.root_node().named_child(0).expect("a top-level node");
            assert_eq!(
                cpp_misparsed_class_name(node, &lines),
                None,
                "{src:?} declares a variable, not a class"
            );
            assert!(
                !is_definition_node(node, Some(Lang::Cpp)),
                "{src:?} must not be a definition"
            );
        }
    }

    /// `template <typename T> class Fwd;` is a forward declaration exactly as
    /// `class Fwd;` is, but `template_declaration` is not itself a specifier, so the
    /// body gate has to reach through it. Forward-declared templates fill
    /// `<iosfwd>`-style headers and any codebase with a container library; each one
    /// used to register a definition that tied the real one at weight 100.
    /// A forward-declared template is a forward declaration exactly as `class Fwd;` is.
    /// Since the wrapper is transparent, the body gate on the inner specifier is what
    /// rejects it — asserted here on the inner node, because that is where the decision
    /// is now made. Forward-declared templates fill `<iosfwd>`-style headers, and each
    /// one used to register a definition that tied the real one at weight 100.
    #[test]
    fn forward_declared_template_is_not_a_definition() {
        let cases: &[(&str, bool)] = &[
            ("template <typename T> class Fwd;\n", false),
            ("template <typename T> struct TIsArray;\n", false),
            (
                "template <typename T> class TArray { public: void Add(T V); };\n",
                true,
            ),
        ];
        for (src, want) in cases {
            let tree = parse(src, Lang::Cpp);
            let inner = find_by_kind(
                tree.root_node(),
                if src.contains("struct") {
                    "struct_specifier"
                } else {
                    "class_specifier"
                },
            );
            assert_eq!(is_definition_node(inner, Some(Lang::Cpp)), *want, "{src:?}");
        }
        // A templated *function* definition still registers, via `function_definition`.
        let src = "template <typename T> void Swap(T& A, T& B) {}\n";
        let tree = parse(src, Lang::Cpp);
        let inner = find_by_kind(tree.root_node(), "function_definition");
        assert!(is_definition_node(inner, Some(Lang::Cpp)));
    }

    /// A qualified *function* declarator is a `friend` declaration — it declares a
    /// member of another class, defined elsewhere — not a definition. Counting it made
    /// the scope annotator attribute the function to the befriending class. `extern`
    /// likewise names something defined in another translation unit.
    #[test]
    fn qualified_declarator_rule_rejects_friends_and_extern() {
        let cases = [
            "struct S { friend void Helper::Assist(); };\n",
            "struct S { friend int Other::Compute(int); };\n",
            "extern int Widget::sExtern;\n",
            "void Foo::Bar();\n",
        ];
        for src in cases {
            let tree = parse(src, Lang::Cpp);
            let mut found_definition = false;
            let mut stack = vec![tree.root_node()];
            let mut cursor = tree.root_node().walk();
            while let Some(n) = stack.pop() {
                if n.kind() == "declaration" && is_definition_node(n, Some(Lang::Cpp)) {
                    found_definition = true;
                }
                stack.extend(n.children(&mut cursor));
            }
            assert!(
                !found_definition,
                "{src:?} must not produce a definition-bearing `declaration`"
            );
        }
    }

    /// `c_declarator_name` descends a chain whose depth is input-controlled, and
    /// `read::outline::generate` is an oss-fuzz target, so an unbounded walk was a
    /// reachable stack overflow (`int` + ~1700 `*` + `p;` crashed). The cap makes deep
    /// chains simply unnameable instead. Depth here is far past anything real and well
    /// inside libFuzzer's default 4 KB input budget.
    #[test]
    fn deep_declarator_chain_is_bounded_not_a_stack_overflow() {
        let src = format!("int {}p;\n", "*".repeat(5000));
        let entries = crate::lang::outline::get_outline_entries(&src, Lang::Cpp);
        // The point is that this returns rather than aborting the process; a chain
        // this deep yields no usable name, so no entry is emitted.
        assert!(
            entries.iter().all(|e| e.name != "p"),
            "a 5000-deep declarator should not resolve to a name"
        );
        // Nested array declarators and qualified names take the same path.
        let arrays = format!("int p{};\n", "[1]".repeat(5000));
        let _ = crate::lang::outline::get_outline_entries(&arrays, Lang::Cpp);
        let quals = format!("void {}f() {{}}\n", "A::".repeat(5000));
        let _ = crate::lang::outline::get_outline_entries(&quals, Lang::Cpp);
    }

    /// `reference_declarator` holds its inner declarator as an *unnamed* child, so the
    /// field probe that walks every other declarator kind stopped one link short and
    /// the name resolved to nothing. Every reference return was affected — a C++
    /// singleton accessor, `at()`, `front()`, `operator[]` — and the symbol was simply
    /// never indexed as a definition. Silent, because text search still matched the
    /// line and labelled it a usage.
    #[test]
    fn reference_returns_resolve_their_name() {
        let cases: &[(&str, &str)] = &[
            ("Holder& FreeRef() { return L; }", "FreeRef"),
            ("Holder&& FreeMove() { return {}; }", "FreeMove"),
            ("const Holder& FreeConst() { return L; }", "FreeConst"),
            ("Holder* FreePtr() { return nullptr; }", "FreePtr"),
            // A reference to a pointer, and a pointer chain, both stack layers on.
            ("Holder*& RefToPtr() { return P; }", "RefToPtr"),
            ("Holder** PtrToPtr() { return nullptr; }", "PtrToPtr"),
            // Out-of-line definitions resolve to the trailing segment.
            ("Holder& Outer::Member() { return L; }", "Member"),
        ];
        for (src, expected) in cases {
            let owned = format!("{src}\n");
            let tree = parse(&owned, Lang::Cpp);
            let lines: Vec<&str> = owned.lines().collect();
            let node = tree.root_node().named_child(0).expect("a top-level node");
            assert_eq!(
                extract_definition_name(node, &lines).as_deref(),
                Some(*expected),
                "wrong name for {src:?} (node kind {})",
                node.kind()
            );
            assert!(
                is_definition_node(node, Some(Lang::Cpp)),
                "{src:?} must be a definition"
            );
        }
    }

    /// The same missing link broke the *other* chain walk: a reference-returning
    /// prototype has a `function_declarator` in its chain, but the walk could not reach
    /// it, so the member was classified as data rather than as a function.
    #[test]
    fn reference_returning_prototype_is_seen_as_a_function() {
        let cases = [
            "struct S { Holder& Get(); };",
            "struct S { Holder&& Take(); };",
            "struct S { const Holder& Peek() const; };",
        ];
        for src in cases {
            let owned = format!("{src}\n");
            let tree = parse(&owned, Lang::Cpp);
            let spec = tree.root_node().named_child(0).expect("a top-level node");
            let body = spec
                .child_by_field_name("body")
                .expect("a field_declaration_list");
            let member = body.named_child(0).expect("one member");
            assert!(
                declarator_chain_has_function(member),
                "{src:?} member ({}) must read as a function",
                member.kind()
            );
        }
    }

    /// A `parenthesized_declarator` wraps its inner declarator as an *unnamed* child,
    /// so `child_by_field_name("declarator")` finds nothing. Without walking named
    /// children, function-pointer typedefs fell back to the raw declarator text —
    /// the exact failure the declarator walk exists to eliminate.
    #[test]
    fn extract_definition_name_cpp_function_pointer_typedef() {
        assert_eq!(
            cpp_name("typedef void (*Cb)(int, const char*);\n", "type_definition"),
            Some("Cb".to_string())
        );
        assert_eq!(
            cpp_name("int (*fp)(void);\n", "declaration"),
            Some("fp".to_string())
        );
    }

    /// Error recovery yields `function_definition` for most macro-class heads but
    /// `declaration` for multiple inheritance. Both are the same construct and must
    /// weigh the same, or a class's rank would depend on its base-class count.
    #[test]
    fn misparsed_class_head_weight_is_independent_of_recovery_shape() {
        let cases = [
            "class API W { public: void M(); };\n",
            "class API W : public B { public: void M(); };\n",
            "class API W : public B1, public B2 { public: void M(); };\n",
        ];
        for src in cases {
            let tree = parse(src, Lang::Cpp);
            let node = tree.root_node().named_child(0).expect("top-level node");
            // `definition_weight_for`, not `definition_weight(kind)`: the kind table
            // cannot tell a class head from an out-of-line data definition, so asking it
            // would stop exercising the production path this test exists to pin.
            assert_eq!(
                definition_weight_for(node),
                100,
                "{src:?} (parsed as {}) should weigh 100",
                node.kind()
            );
        }
    }

    /// `int Widget::sCount = 0;` is the out-of-line definition of a static member — a
    /// real definition, but a `declaration` node, so it did not register and only the
    /// in-class declaration was findable. A *qualified* declarator is what marks it:
    /// ordinary locals and globals never have one, and a variable whose *type* is
    /// qualified keeps the qualifier on the `type` field instead.
    #[test]
    fn is_definition_node_accepts_out_of_line_static_member_definition() {
        let src = "int Widget::sCount = 0;\n";
        let tree = parse(src, Lang::Cpp);
        let lines: Vec<&str> = src.lines().collect();
        let node = tree.root_node().named_child(0).expect("declaration");
        assert_eq!(node.kind(), "declaration");
        assert!(is_definition_node(node, Some(Lang::Cpp)));
        assert_eq!(
            extract_definition_name(node, &lines),
            Some("sCount".to_string())
        );
    }

    #[test]
    fn is_definition_node_rejects_ordinary_declarations() {
        // The guard on the rule above: none of these may become definitions, or every
        // local variable in every C++ function would be reported as one.
        for src in [
            "int gGlobal = 1;\n",
            "Holder h;\n",
            "Foo::Bar qualifiedType;\n", // qualifier is on `type`, not the declarator
            "void Prototype(int x);\n",
            "static const char* kName = \"x\";\n",
        ] {
            let tree = parse(src, Lang::Cpp);
            let node = tree.root_node().named_child(0).expect("top-level node");
            assert!(
                !is_definition_node(node, Some(Lang::Cpp)),
                "{src:?} (kind {}) must not be a definition",
                node.kind()
            );
        }
    }

    #[test]
    fn cpp_misparsed_class_name_rejects_real_declarations() {
        // A function returning an elaborated type has the same bodyless-specifier
        // `type` as a misparsed class head; the `function_declarator` is what tells
        // them apart. A variable of an elaborated type has no ERROR node.
        for src in [
            "class Legit ReturnsElaborated() { return Legit(); }\n",
            "class Legit* GlobalPtr;\n",
            "class Real { int X; };\n",
            "void Run() {}\n",
        ] {
            let tree = parse(src, Lang::Cpp);
            let lines: Vec<&str> = src.lines().collect();
            let node = tree.root_node().named_child(0).expect("a top-level node");
            assert_eq!(
                cpp_misparsed_class_name(node, &lines),
                None,
                "{src:?} must not be treated as a misparsed class head"
            );
        }
    }
}
