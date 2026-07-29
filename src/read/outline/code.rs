use crate::lang::outline::{extract_import_source, outline_language, walk_top_level};
use crate::types::{Lang, OutlineEntry, OutlineKind};

/// Generate a code outline using tree-sitter. Walks top-level AST nodes,
/// emitting signatures without bodies.
#[must_use]
pub fn outline(content: &str, lang: Lang, max_lines: usize) -> String {
    let Some(language) = outline_language(lang) else {
        return fallback_outline(content, max_lines);
    };

    let mut parser = tree_sitter::Parser::new();
    if parser.set_language(&language).is_err() {
        return fallback_outline(content, max_lines);
    }

    let Some(tree) = parser.parse(content, None) else {
        return fallback_outline(content, max_lines);
    };

    let root = tree.root_node();
    let lines: Vec<&str> = content.lines().collect();
    let entries = walk_top_level(root, &lines, lang);

    format_entries(&entries, &lines, max_lines, lang)
}

/// Format outline entries into the spec'd output format.
fn format_entries(
    entries: &[OutlineEntry],
    _lines: &[&str],
    max_lines: usize,
    lang: Lang,
) -> String {
    let mut out = Vec::new();
    let mut import_groups: Vec<&str> = Vec::new();
    // Track the start line of the first import in the current group.
    let mut import_group_start: u32 = 1;

    for entry in entries {
        if out.len() >= max_lines {
            break;
        }

        match entry.kind {
            OutlineKind::Import => {
                if import_groups.is_empty() {
                    import_group_start = entry.start_line;
                }
                import_groups.push(&entry.name);
                continue;
            }
            _ => {
                // Flush any accumulated imports
                if !import_groups.is_empty() {
                    out.push(format_imports(&import_groups, import_group_start, lang));
                    import_groups.clear();
                }
            }
        }

        // Flatten namespace modules — hoist their children to top level
        // so classes inside namespaces show their methods at indent 1.
        if entry.kind == OutlineKind::Module && !entry.children.is_empty() {
            out.push(format_entry(entry, 0, lang));
            for child in &entry.children {
                if out.len() >= max_lines {
                    break;
                }
                out.push(format_entry(child, 1, lang));
                for grandchild in &child.children {
                    if out.len() >= max_lines {
                        break;
                    }
                    out.push(format_entry(grandchild, 2, lang));
                }
            }
        } else {
            out.push(format_entry(entry, 0, lang));
            for child in &entry.children {
                if out.len() >= max_lines {
                    break;
                }
                out.push(format_entry(child, 1, lang));
            }
        }
    }

    // Flush trailing imports
    if !import_groups.is_empty() {
        out.push(format_imports(&import_groups, import_group_start, lang));
    }

    out.join("\n")
}

/// Format a collapsed import summary grouped by source with counts.
/// Spec format: `imports: react(4), express(2), @/lib(3)`
fn format_imports(imports: &[&str], start: u32, lang: Lang) -> String {
    let count = imports.len();

    // Extract source modules and count occurrences
    let mut sources: Vec<String> = Vec::new();
    let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    for imp in imports {
        let source = extract_import_source(imp, Some(lang));
        *seen.entry(source.clone()).or_insert(0) += 1;
        if !sources.contains(&source) {
            sources.push(source);
        }
    }

    // Format as "source(count)" or just "source" if count is 1
    let mut parts: Vec<String> = Vec::new();
    for src in sources.iter().take(5) {
        let c = seen[src];
        if c > 1 {
            parts.push(format!("{src}({c})"));
        } else {
            parts.push(src.clone());
        }
    }

    let suffix = if count > 5 {
        format!(", ... ({count} total)")
    } else {
        String::new()
    };
    let condensed = parts.join(", ");
    format!("[{start}-]   imports: {condensed}{suffix}")
}

/// Format a single outline entry with optional indentation.
fn format_entry(entry: &OutlineEntry, indent: usize, lang: Lang) -> String {
    let prefix = "  ".repeat(indent);
    let range = if entry.start_line == entry.end_line {
        format!("[{}]", entry.start_line)
    } else {
        format!("[{}-{}]", entry.start_line, entry.end_line)
    };

    let kind_label = match entry.kind {
        OutlineKind::Function => {
            if lang == Lang::Scala {
                "def"
            } else if lang == Lang::Kotlin {
                "fun"
            } else {
                "fn"
            }
        }
        OutlineKind::Class => "class",
        OutlineKind::Struct => "struct",
        OutlineKind::Interface => {
            if lang == Lang::Scala {
                "trait"
            } else {
                "interface"
            }
        }
        OutlineKind::TypeAlias => "type",
        OutlineKind::Enum => "enum",
        OutlineKind::Constant => "const",
        OutlineKind::ImmutableVariable => "val",
        OutlineKind::Variable => {
            if lang == Lang::Scala {
                "var"
            } else {
                "let"
            }
        }
        OutlineKind::Export => "export",
        OutlineKind::Property => "prop",
        OutlineKind::Module => {
            if lang == Lang::Scala || lang == Lang::Kotlin {
                "object"
            } else {
                "mod"
            }
        }
        OutlineKind::Import => "import",
        OutlineKind::TestSuite => "suite",
        OutlineKind::TestCase => "test",
    };

    let sig = match &entry.signature {
        Some(s) => format!("\n{prefix}           {s}"),
        None => String::new(),
    };

    let doc = match &entry.doc {
        Some(d) => {
            let truncated = if d.len() > 60 {
                format!("{}...", crate::types::truncate_str(d, 57))
            } else {
                d.clone()
            };
            format!("  // {truncated}")
        }
        None => String::new(),
    };

    format!("{prefix}{range:<12} {kind_label} {}{sig}{doc}", entry.name)
}

/// Fallback when tree-sitter grammar isn't available.
fn fallback_outline(content: &str, _max_lines: usize) -> String {
    super::fallback::head_tail(content)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scala_outline_constructs() {
        let scala_code = r#"
package example

import scala.util.Try

trait DataSource {
  def load(): String
}

class Database {
  val connectionString = "jdbc:..."
  var connected = false
  
  def connect(): Unit = {}
}

object Database {
  def create(): Database = new Database()
}

enum Color {
  case Red, Green, Blue
}

type UserId = String
"#;

        let outline = outline(scala_code, Lang::Scala, 1000);

        assert!(outline.contains("trait DataSource"));
        assert!(outline.contains("class Database"));
        assert!(outline.contains("object Database"));
        assert!(outline.contains("enum Color"));
        assert!(outline.contains("type UserId"));
        assert!(outline.contains("val connectionString"));
        assert!(outline.contains("var connected"));
        assert!(outline.contains("def load"));
        assert!(outline.contains("def connect"));
        assert!(outline.contains("def create"));
    }

    #[test]
    fn php_outline_constructs() {
        let php_code = r#"<?php
namespace App\Services;

use App\Support\Client;

trait LogsQueries {
    public function log(string $query): void {}
}

class UserService {
    use LogsQueries;

    public function __construct(private Client $client) {}

    public function findUser(int $id): array {
        return $this->client->loadUser($id);
    }
}
"#;

        let outline = outline(php_code, Lang::Php, 1000);

        assert!(outline.contains("mod App\\Services"));
        assert!(outline.contains("imports: App\\Support\\Client"));
        assert!(outline.contains("interface LogsQueries"));
        assert!(outline.contains("class UserService"));
        assert!(outline.contains("fn findUser"));
    }

    #[test]
    fn kotlin_outline_constructs() {
        let kotlin_code = r#"
package com.example

import kotlin.collections.List
import kotlin.io.println

interface Drawable {
    fun draw()
}

data class Point(val x: Int, val y: Int)

class Canvas : Drawable {
    val width = 800
    var height = 600

    override fun draw() {
        println("Drawing")
    }

    fun resize(w: Int, h: Int) {}

    companion object {
        fun create(): Canvas = Canvas()
    }
}

object Registry {
    fun register(item: Drawable) {}
}

enum class Color {
    RED, GREEN, BLUE
}

fun String.isPalindrome(): Boolean = this == this.reversed()

fun main() {
    val canvas = Canvas()
    canvas.draw()
}
"#;

        let outline = outline(kotlin_code, Lang::Kotlin, 1000);

        // Imports
        assert!(
            outline.contains("imports:"),
            "should have collapsed imports"
        );
        // Interface (shown as class since Kotlin grammar uses class_declaration)
        assert!(outline.contains("class Drawable"), "should have Drawable");
        // Data class
        assert!(outline.contains("class Point"), "should have Point");
        // Regular class with methods
        assert!(outline.contains("class Canvas"), "should have Canvas");
        assert!(outline.contains("fun draw"), "should have draw method");
        assert!(outline.contains("fun resize"), "should have resize method");
        // Properties inside classes
        assert!(outline.contains("prop width"), "should have width property");
        assert!(
            outline.contains("prop height"),
            "should have height property"
        );
        // Object declaration
        assert!(
            outline.contains("object Registry"),
            "should have Registry object"
        );
        assert!(
            outline.contains("fun register"),
            "should have register method"
        );
        // Enum class
        assert!(outline.contains("class Color"), "should have Color enum");
        // Top-level functions
        assert!(
            outline.contains("fun isPalindrome"),
            "should have extension fun"
        );
        assert!(outline.contains("fun main"), "should have main");
        // Kotlin-specific labels
        assert!(outline.contains("fun "), "should use 'fun' not 'fn'");
        assert!(!outline.contains("fn "), "should not use 'fn' for Kotlin");
    }

    #[test]
    fn cpp_outline_constructs() {
        // Before C++ type support the outline for this file was almost entirely
        // `<anonymous>`: no type specifier had an arm, and C/C++ name their
        // declarations through a declarator chain rather than a `name` field, so even
        // the functions came out unnamed.
        let cpp_code = r"
#include <vector>

struct Point { int X; };
union Value { int I; float F; };
enum class Mode : uint8_t { On, Off };
typedef unsigned int Handle;
using Callback = void(*)(int);

template <typename T> class Vector { public: void Push(T V); };

namespace Outer {
class Widget final : public Base
{
public:
    void Work();
    int Count;
};
}

void Widget::Work() {}
";

        let outline = outline(cpp_code, Lang::Cpp, 1000);

        assert!(outline.contains("struct Point"), "actual:\n{outline}");
        assert!(outline.contains("struct Value"), "actual:\n{outline}");
        assert!(outline.contains("enum Mode"), "actual:\n{outline}");
        assert!(outline.contains("type Handle"), "actual:\n{outline}");
        assert!(outline.contains("type Callback"), "actual:\n{outline}");
        // `template <typename T> class Vector` renders as the class it declares, not
        // as an unnamed template wrapper.
        assert!(outline.contains("class Vector"), "actual:\n{outline}");
        assert!(outline.contains("mod Outer"), "actual:\n{outline}");
        assert!(outline.contains("class Widget"), "actual:\n{outline}");
        // Class members, reached through the `field_declaration_list` body.
        assert!(outline.contains("fn Work"), "actual:\n{outline}");
        assert!(outline.contains("prop Count"), "actual:\n{outline}");
        // The out-of-line definition resolves to the bare member name.
        assert!(
            !outline.contains("<anonymous>"),
            "no entry should be anonymous:\n{outline}"
        );
    }

    #[test]
    fn cpp_outline_names_class_behind_export_macro() {
        // `class MYLIB_API Widget : public Base` does not parse as a class —
        // tree-sitter-cpp reads the macro as the class name. The outline must still
        // name the class, and must not surface the macro as a type of its own.
        //
        // The bare `ANNOTATE()` / `BODY_MACRO()` call-shaped macros around and inside
        // the head are what code-generating C++ frameworks emit; they are here because
        // they change nothing, which is the point worth pinning.
        let cpp_code = "\
ANNOTATE()
class MYLIB_API Widget final : public Base
{
    BODY_MACRO()
public:
    void Work();
};
";
        let outline = outline(cpp_code, Lang::Cpp, 1000);
        assert!(outline.contains("class Widget"), "actual:\n{outline}");
        assert!(
            !outline.contains("MYLIB_API"),
            "the export macro must not appear as a type:\n{outline}"
        );
        // Members of the misparsed body are still collected.
        assert!(outline.contains("fn Work"), "actual:\n{outline}");
    }

    /// The export macro must cost nothing at all: the same class with and without it
    /// has to outline identically, member for member and kind for kind.
    ///
    /// It did not. Once the head misparses the body is a `compound_statement`, and a
    /// constructor — which has no return type to anchor a declaration — is re-read as
    /// a *call*, a destructor as a stranded declarator inside an `ERROR`. Neither
    /// reached an outline arm, so both vanished; and `int Value;` in a statement body
    /// is an ordinary local, so it came out `let` where the plain class gives `prop`.
    /// Losing a constructor is not cosmetic: it is also absent from deps' exported
    /// symbols and gets no blast radius when edited.
    #[test]
    fn cpp_outline_export_macro_costs_no_members() {
        let with_macro = "\
class MYLIB_API Widget : public Base
{
public:
    Widget();
    ~Widget();
    void Work();
    int Value;
};
";
        let plain = with_macro.replace("class MYLIB_API ", "class ");

        let macro_outline = outline(with_macro, Lang::Cpp, 1000);
        let plain_outline = outline(&plain, Lang::Cpp, 1000);
        assert_eq!(
            macro_outline, plain_outline,
            "export macro changed the outline\nwith macro:\n{macro_outline}\nplain:\n{plain_outline}"
        );
        for expected in [
            "class Widget",
            "fn Widget",
            "fn ~Widget",
            "fn Work",
            "prop Value",
        ] {
            assert!(
                macro_outline.contains(expected),
                "missing {expected:?}:\n{macro_outline}"
            );
        }
    }

    /// The constructor recovery reads a *call*, which is also exactly what a
    /// zero-argument macro invocation becomes — so the class's own name is the only
    /// thing separating them. `is_cpp_macro_invocation` enforces the same rule for
    /// bodies that parsed cleanly, but it cannot see this shape.
    #[test]
    fn cpp_outline_misparsed_body_keeps_macros_out() {
        let cpp_code = "\
class MYLIB_API Widget : public Base
{
    GENERATED_BODY()
public:
    Widget();
    void Work();
};
";
        let outline = outline(cpp_code, Lang::Cpp, 1000);
        assert!(outline.contains("fn Widget"), "actual:\n{outline}");
        assert!(
            !outline.contains("GENERATED_BODY"),
            "a macro invocation must not outline as a member:\n{outline}"
        );
    }

    /// Recovery does not repair every member into the same shape: a specifier keyword
    /// (`explicit`) or an inline body moves a constructor into a different artifact,
    /// and one `ERROR` can swallow two members at once.
    #[test]
    fn cpp_outline_export_macro_members_across_recovery_shapes() {
        let cases: &[(&str, &[&str])] = &[
            // `explicit` keeps recovery from reading a call; this `ERROR` holds both
            // the constructor and the destructor.
            (
                "class API Widget\n{\npublic:\n    Widget(int A, float B);\n    explicit Widget(int A);\n    ~Widget();\nprivate:\n    int Value;\n    static int Count;\n};\n",
                &["fn Widget", "fn ~Widget", "prop Value", "prop Count"],
            ),
            // Inline bodies: the destructor and method parse as real definitions, the
            // constructor still does not.
            (
                "class API Widget\n{\npublic:\n    Widget() {}\n    ~Widget() {}\n    void Work() { Value = 1; }\n    int Value;\n};\n",
                &["fn Widget", "fn ~Widget", "fn Work", "prop Value"],
            ),
            // No base class, no access specifier — the members sit directly in the body.
            (
                "struct API Point\n{\n    Point();\n    ~Point();\n    int X;\n};\n",
                &["fn Point", "fn ~Point", "prop X"],
            ),
            // A longer base-class name tips the head into the other repair (see
            // `macro_class_head_recovery_shapes_are_genuinely_different`).
            (
                "class API Widget : public VeryLongBaseClassNameHere\n{\npublic:\n    Widget();\n    virtual ~Widget();\n    int Value;\n};\n",
                &["fn Widget", "fn ~Widget", "prop Value"],
            ),
        ];
        for (src, expected) in cases {
            let rendered = outline(src, Lang::Cpp, 1000);
            for want in *expected {
                assert!(
                    rendered.contains(want),
                    "missing {want:?} for {src:?}:\n{rendered}"
                );
            }
        }
    }

    #[test]
    fn cpp_outline_omits_forward_declarations() {
        // A specifier with no body is an elaborated type specifier — a forward
        // declaration or a type reference, not a definition.
        let outline = outline(
            "class Fwd;\nclass Fwd* Global;\nclass Real { int X; };\n",
            Lang::Cpp,
            1000,
        );
        assert!(outline.contains("class Real"), "actual:\n{outline}");
        assert!(
            !outline.contains("class Fwd"),
            "forward declaration must not be outlined:\n{outline}"
        );
    }

    /// `#include` is `preproc_include`, which no outline arm matched — so a C or C++
    /// outline showed a header's types but never what it included, even though
    /// `extract_import_source` already knew how to parse the directive. The `<…>` /
    /// `"…"` delimiters are kept in the rendered group because they are what tells a
    /// system header from a project-relative one.
    #[test]
    fn cpp_outline_includes_preprocessor_includes() {
        let cpp_code = "#include <vector>\n\
                        #include <memory>\n\
                        #include \"Local.h\"\n\
                        \n\
                        class Widget { public: void Work(); };\n";
        let outline = outline(cpp_code, Lang::Cpp, 1000);
        assert!(
            outline.contains("imports:"),
            "C++ outline must surface #include lines:\n{outline}"
        );
        assert!(outline.contains("vector"), "actual:\n{outline}");
        assert!(outline.contains("memory"), "actual:\n{outline}");
        assert!(outline.contains("Local.h"), "actual:\n{outline}");
        // The types still render alongside.
        assert!(outline.contains("class Widget"), "actual:\n{outline}");
    }

    /// `struct S { int a; } sInstance;` declares a type and a variable in one node. The
    /// outline used to show only `sInstance`, leaving the type invisible even though
    /// symbol search found it — an outline/symbol disagreement on ordinary C.
    #[test]
    fn cpp_outline_surfaces_type_declared_with_a_variable() {
        let outline = outline("struct S { int a; } sInstance;\n", Lang::Cpp, 1000);
        assert!(
            outline.contains("struct S"),
            "the type must be outlined:\n{outline}"
        );
    }

    /// The type-over-variable preference must not fire for an *anonymous* specifier:
    /// `<anonymous>` is not searchable, whereas the variable name is, so preferring the
    /// type there trades a usable identifier for a placeholder.
    #[test]
    fn cpp_outline_keeps_variable_name_for_anonymous_specifier() {
        let anon = outline("struct { int a; } anonInst;\n", Lang::Cpp, 1000);
        assert!(
            anon.contains("anonInst"),
            "an anonymous struct must keep the searchable variable name:\n{anon}"
        );
        assert!(
            !anon.contains("<anonymous>"),
            "must not degrade to a placeholder:\n{anon}"
        );
        // The `typedef struct { … } Foo;` idiom — the dominant C spelling — is named by
        // the typedef and must be unaffected.
        let td = outline("typedef struct { int b; } Named;\n", Lang::Cpp, 1000);
        assert!(td.contains("Named"), "actual:\n{td}");
        assert!(!td.contains("<anonymous>"), "actual:\n{td}");
    }

    /// Platform guards are the dominant use of `#ifdef` in C/C++ headers, so anything
    /// inside one was invisible in the outline while `tilth_deps` — which reads lines
    /// rather than the AST — reported it. Conditional blocks are now transparent.
    #[test]
    fn cpp_outline_descends_into_conditional_compilation_blocks() {
        let cpp_code = "#include <always.h>\n\
                        #ifdef _WIN32\n\
                        #include <windows.h>\n\
                        class WinOnly { public: void Work(); };\n\
                        #else\n\
                        class PosixOnly { public: void Work(); };\n\
                        #endif\n";
        let outline = outline(cpp_code, Lang::Cpp, 1000);
        assert!(outline.contains("always.h"), "actual:\n{outline}");
        assert!(
            outline.contains("windows.h"),
            "a guarded include must be surfaced:\n{outline}"
        );
        assert!(
            outline.contains("class WinOnly"),
            "a guarded declaration must be surfaced:\n{outline}"
        );
        // Both arms are shown — tilth does not evaluate the preprocessor, and both
        // exist in the source.
        assert!(
            outline.contains("class PosixOnly"),
            "the #else arm must be surfaced too:\n{outline}"
        );
    }

    #[test]
    fn c_outline_includes_preprocessor_includes() {
        // Same arm serves C, whose outlines were equally include-blind.
        let outline = outline(
            "#include <stdio.h>\nstruct Point { int x; };\n",
            Lang::C,
            1000,
        );
        assert!(outline.contains("imports:"), "actual:\n{outline}");
        assert!(outline.contains("stdio.h"), "actual:\n{outline}");
    }

    /// A bare macro invocation in a class body is shaped exactly like a constructor
    /// declaration — a typeless `declaration` with a `function_declarator` — so outlining
    /// it named the macro as a member of the class. That is worse than cosmetic: the name
    /// became an "exported symbol" of the header, and `tilth_deps` then reported every
    /// other file invoking the same macro as a dependent. Constructors and destructors
    /// must still be outlined; only the name distinguishes them.
    #[test]
    fn cpp_outline_omits_macro_invocations_but_keeps_constructors() {
        let cpp_code = "\
class Widget
{
\tBODY_MACRO()
public:
\tWidget();
\t~Widget();
\texplicit Widget(int Times);
\tvoid Work();
};
";
        let outline = outline(cpp_code, Lang::Cpp, 1000);
        assert!(
            !outline.contains("BODY_MACRO"),
            "a macro invocation must not be outlined as a member:\n{outline}"
        );
        // Assert on the *rendered member line*, not on the bare name. `contains("Widget")`
        // is satisfied by the `class Widget` header the outline always emits, so it stayed
        // green under a mutation that dropped every C++ constructor — a review found
        // exactly that. `fn Widget` can only come from the member entry.
        assert_eq!(
            outline.matches("fn Widget").count(),
            2,
            "both constructors must be outlined:\n{outline}"
        );
        assert!(
            outline.contains("fn ~Widget"),
            "the destructor must still be outlined:\n{outline}"
        );
        assert!(outline.contains("fn Work"), "actual:\n{outline}");
    }

    /// A constructor is recognised by comparing its name to the enclosing type's. That
    /// comparison used the whole text of the specifier's `name` node, which is not always
    /// a bare identifier — so any class whose name is a `template_type` or a
    /// `qualified_identifier` had its constructor misread as a macro and dropped. The loss
    /// reached past outlines: `get_outline_entries` also feeds deps' exported symbols and
    /// blast radius, so an edit to such a constructor produced no blast radius at all.
    #[test]
    fn cpp_outline_keeps_constructors_of_specialized_and_qualified_classes() {
        for (src, ctor) in [
            (
                "template <> class Box<int> { public: Box(); void Put(int); };\n",
                "fn Box",
            ),
            (
                "template <typename T> class Box<T*> { public: Box(); void Put(T* p); };\n",
                "fn Box",
            ),
            (
                "class Outer::Inner { public: Inner(); void Go(); };\n",
                "fn Inner",
            ),
            (
                "namespace n { class Deep { public: Deep(); void Go(); }; }\n",
                "fn Deep",
            ),
        ] {
            let outline = outline(src, Lang::Cpp, 1000);
            assert!(
                outline.contains(ctor),
                "constructor dropped from `{src}`:\n{outline}"
            );
        }
    }

    /// A *braced* PHP namespace is a Module entry whose body is a
    /// `compound_statement` — the same kind a macro-misparsed C++ class body uses. When
    /// `collect_children`'s body finder matched that kind ungated, PHP namespace
    /// members started appearing with placeholder names (`prop <property>`,
    /// `const <const>`) and a `trait` mislabelled as an interface.
    #[test]
    fn php_braced_namespace_outline_unaffected_by_cpp_body_kinds() {
        let php_code = "<?php\nnamespace App {\n    class A { public $p = 1; }\n    trait T {}\n    const C = 1;\n}\n";
        let outline = outline(php_code, Lang::Php, 1000);
        assert!(outline.contains("mod App"), "actual:\n{outline}");
        assert!(
            !outline.contains("<property>") && !outline.contains("<const>"),
            "no placeholder member names should leak into PHP outlines:\n{outline}"
        );
        assert!(
            !outline.contains("interface T"),
            "a PHP trait must not be outlined as an interface:\n{outline}"
        );
    }

    /// `field_declaration` and `declaration` are outlined for C/C++ only — both kind
    /// strings also exist in the Rust, Go, Java and C# grammars, where surfacing
    /// every struct field would change those outlines.
    #[test]
    fn rust_outline_still_omits_struct_fields() {
        let outline = outline(
            "pub struct Holder {\n    count: u32,\n}\n",
            Lang::Rust,
            1000,
        );
        assert!(outline.contains("struct Holder"), "actual:\n{outline}");
        assert!(
            !outline.contains("count"),
            "Rust struct fields must stay out of the outline:\n{outline}"
        );
    }

    #[test]
    fn ts_export_outline_no_doubled_keyword() {
        // Regression: `export_statement` must recurse into the wrapped
        // declaration (function/class/lexical) and render with that
        // declaration's real kind label. Pre-fix, both the kind label and
        // the synthesized name began with `export `, so the renderer
        // emitted `export export async function foo(`. Leaf-fallback cases
        // (`export { ... }`, `export * from ...`) keep `OutlineKind::Export`
        // but strip the leading `export ` from the name to avoid duplication.
        let ts_code = r#"
export async function foo() {}

export class Foo {}

export const x = 1;

export { foo };

export * from './bar';

export default class Bar {}
"#;

        let outline = outline(ts_code, Lang::TypeScript, 1000);

        // No outline line may contain a doubled `export` keyword.
        for line in outline.lines() {
            assert!(
                !line.contains("export export"),
                "doubled `export` in outline line: {line:?}\nfull outline:\n{outline}"
            );
        }

        // Wrapped declarations must render with the real kind, not as
        // `OutlineKind::Export` carrying the entire source as `name`.
        assert!(
            outline.contains("fn foo"),
            "export async function foo should render as `fn foo`:\n{outline}"
        );
        assert!(
            outline.contains("class Foo"),
            "export class Foo should render as `class Foo`:\n{outline}"
        );
        assert!(
            outline.contains("class Bar"),
            "export default class Bar should render as `class Bar`:\n{outline}"
        );
    }
}
