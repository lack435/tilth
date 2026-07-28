pub mod detection;
pub mod outline;
pub mod treesitter;

use std::path::Path;

use crate::types::{FileType, Lang};

/// Detect file type by extension, then by name.
pub fn detect_file_type(path: &Path) -> FileType {
    match path.extension().and_then(|e| e.to_str()) {
        Some("ts") => FileType::Code(Lang::TypeScript),
        Some("tsx") => FileType::Code(Lang::Tsx),
        Some("js" | "jsx") => FileType::Code(Lang::JavaScript),
        Some("py" | "pyi") => FileType::Code(Lang::Python),
        Some("rs") => FileType::Code(Lang::Rust),
        Some("go") => FileType::Code(Lang::Go),
        Some("java") => FileType::Code(Lang::Java),
        Some("scala" | "sc") => FileType::Code(Lang::Scala),
        Some("c") => FileType::Code(Lang::C),
        // `.h` is parsed with the C++ grammar, not the C one. `.h` is the header
        // extension virtually all C++ projects use, and a C++ header parsed as C
        // misparses catastrophically: `class Foo { … };` becomes a
        // `function_definition` named `Foo` (the C grammar has no `class` keyword), so
        // every class in a header resolved as an anonymous function, and any form the
        // accident didn't cover — `class Foo final : public Bar` — failed to parse at
        // all.
        //
        // C++ is not a strict superset of C, so this is a trade, not a free win. For
        // outline and symbol purposes it is heavily one-sided: of the C-only
        // constructs, only a K&R-style definition (`int add(a, b) int a; int b; {…}`,
        // which tree-sitter-cpp reads as a variable) and an identifier that is a C++
        // keyword (`int template;`) resolve worse. `restrict`, `_Generic`, `_Atomic`,
        // designated initialisers, compound literals, VLAs, bitfields and anonymous
        // unions all parse identically under both grammars — and C headers gain
        // named structs, enums, typedefs and prototypes that the C-grammar path
        // reported as `<anonymous>`.
        Some("cpp" | "hpp" | "hh" | "hxx" | "cc" | "cxx" | "h") => FileType::Code(Lang::Cpp),
        Some("rb") => FileType::Code(Lang::Ruby),
        Some("php" | "phtml") => FileType::Code(Lang::Php),
        Some("swift") => FileType::Code(Lang::Swift),
        Some("kt" | "kts") => FileType::Code(Lang::Kotlin),
        Some("cs") => FileType::Code(Lang::CSharp),
        Some("ex" | "exs") => FileType::Code(Lang::Elixir),
        Some("sh" | "bash" | "bats") => FileType::Code(Lang::Bash),

        Some("md" | "mdx" | "rst") => FileType::Markdown,
        Some("json" | "yaml" | "yml" | "toml" | "xml" | "ini") => FileType::StructuredData,
        Some("csv" | "tsv") => FileType::Tabular,
        Some("log") => FileType::Log,

        None => file_type_from_name(path),
        _ => FileType::Other,
    }
}

fn file_type_from_name(path: &Path) -> FileType {
    match path.file_name().and_then(|n| n.to_str()) {
        Some("Dockerfile" | "Containerfile") => FileType::Code(Lang::Dockerfile),
        Some("Makefile" | "GNUmakefile") => FileType::Code(Lang::Make),
        Some("Vagrantfile" | "Rakefile") => FileType::Code(Lang::Ruby),
        Some(n) if n.starts_with(".env") => FileType::StructuredData,
        Some(".bashrc" | ".bash_profile" | ".bash_aliases" | ".profile") => {
            FileType::Code(Lang::Bash)
        }
        _ => FileType::Other,
    }
}

/// Find the nearest package root by looking for manifest files.
pub(crate) fn package_root(path: &Path) -> Option<&Path> {
    const MANIFESTS: &[&str] = &[
        "Cargo.toml",
        "package.json",
        "pyproject.toml",
        "setup.py",
        "go.mod",
        "pom.xml",
        "build.gradle",
        "build.sbt",
        "mix.exs",
        "composer.json", // PHP / Laravel
    ];
    let mut dir = path;
    loop {
        for m in MANIFESTS {
            if dir.join(m).exists() {
                return Some(dir);
            }
        }
        dir = dir.parent()?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `.h` must resolve to the C++ grammar. tree-sitter-cpp is a superset of
    /// tree-sitter-c, so routing C headers through it is harmless, while routing C++
    /// headers through the C grammar is not: the C grammar has no `class` keyword, so
    /// `class Widget { … };` misparses into a `function_definition` and every class
    /// declared in a header — which is where C++ declares them — became invisible.
    #[test]
    fn header_extensions_use_the_cpp_grammar() {
        for name in ["Probe.h", "Probe.hpp", "Probe.hh", "Probe.hxx"] {
            assert_eq!(
                detect_file_type(Path::new(name)),
                FileType::Code(Lang::Cpp),
                "{name} should be parsed as C++"
            );
        }
        // Implementation extensions are unchanged; only `.c` stays on the C grammar.
        assert_eq!(
            detect_file_type(Path::new("probe.c")),
            FileType::Code(Lang::C)
        );
        for name in ["Probe.cpp", "Probe.cc", "Probe.cxx"] {
            assert_eq!(
                detect_file_type(Path::new(name)),
                FileType::Code(Lang::Cpp),
                "{name} should be parsed as C++"
            );
        }
    }
}
