use std::path::PathBuf;
use std::time::SystemTime;

/// What kind of query the user issued.
#[derive(Debug)]
pub enum QueryType {
    FilePath(PathBuf),
    Glob(String),
    Symbol(String),
    /// Broad concept query — single lowercase word or multi-word phrase
    /// that likely refers to a feature/module/flow rather than an exact symbol.
    Concept(String),
    Content(String),
    /// Slash-wrapped regex: `/pattern/` → regex content search.
    Regex(String),
    /// Path-like query that didn't resolve — try symbol, then content.
    Fallthrough(String),
}

/// Programming language, carried through the type system so downstream
/// code never re-detects. Adding a language means adding an arm here
/// and the compiler tells you everywhere else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lang {
    Rust,
    TypeScript,
    Tsx,
    JavaScript,
    Python,
    Go,
    Java,
    Scala,
    C,
    Cpp,
    Ruby,
    Php,
    Swift,
    Kotlin,
    CSharp,
    Elixir,
    Bash,
    Dockerfile,
    Make,
}

impl Lang {
    /// Returns `true` if the language uses `'` for lifetime ticks (`'a`,
    /// `'static`) rather than as a string/char delimiter.
    ///
    /// Lexers that scan `'` must disambiguate a lifetime from a single-quoted
    /// literal; only Rust needs the lifetime branch.
    pub(crate) fn has_lifetimes(self) -> bool {
        match self {
            Lang::Rust => true,
            Lang::TypeScript
            | Lang::Tsx
            | Lang::JavaScript
            | Lang::Python
            | Lang::Go
            | Lang::Java
            | Lang::Scala
            | Lang::C
            | Lang::Cpp
            | Lang::Ruby
            | Lang::Php
            | Lang::Swift
            | Lang::Kotlin
            | Lang::CSharp
            | Lang::Elixir
            | Lang::Bash
            | Lang::Dockerfile
            | Lang::Make => false,
        }
    }
}

/// File type as detected by extension. Determines outline strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    Code(Lang),
    Markdown,
    StructuredData,
    Tabular,
    Log,
    Other,
}

/// What the output contains — shown in the header bracket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewMode {
    Full,
    Outline,
    Signature,
    Keys,
    // Reserved/roadmap: planned head+tail view mode, not yet wired.
    #[allow(dead_code)]
    HeadTail,
    Empty,
    Generated,
    Minified,
    // Reserved/roadmap: binary file view variant, not yet wired.
    #[allow(dead_code)]
    Binary,
    // Reserved/roadmap: error view variant, not yet wired.
    #[allow(dead_code)]
    Error,
    Section,
    Stripped,
}

impl std::fmt::Display for ViewMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full => write!(f, "full"),
            Self::Outline => write!(f, "outline"),
            Self::Signature => write!(f, "signature"),
            Self::Keys => write!(f, "keys"),
            Self::HeadTail => write!(f, "head+tail"),
            Self::Empty => write!(f, "empty"),
            Self::Generated => write!(f, "generated — skipped"),
            Self::Minified => write!(f, "minified — skipped"),
            Self::Binary => write!(f, "skipped"),
            Self::Error => write!(f, "error"),
            Self::Section => write!(f, "section"),
            Self::Stripped => write!(f, "stripped"),
        }
    }
}

/// A single search match, carrying enough context for ranking and display.
#[derive(Debug, Clone)]
pub struct Match {
    pub path: PathBuf,
    pub line: u32,
    /// The matched line, as displayed *and* as ranked. Build it with [`match_text`] — never
    /// from a raw line — so that a leading UTF-8 BOM cannot reach either use. See that
    /// function for why the two cannot be separated.
    pub text: String,
    pub is_definition: bool,
    pub exact: bool,
    pub file_lines: u32,
    pub mtime: SystemTime,
    /// Line range of the enclosing definition node (for expand).
    /// Populated by tree-sitter for definitions; None for usages.
    pub def_range: Option<(u32, u32)>,
    /// The defined symbol name (populated from AST during definition detection).
    pub def_name: Option<String>,
    /// Semantic weight for definition kinds. 0 for usages.
    pub def_weight: u16,
    /// For impl/implements matches: the trait or interface being implemented.
    /// None for primary definitions and plain usages.
    pub impl_target: Option<String>,
}

/// Build a [`Match::text`] from a raw file line: BOM removed, trailing whitespace removed.
///
/// One helper for every search backend, because `Match.text` is read by two kinds of
/// consumer and a BOM breaks both — which is what made #51 more than the cosmetic issue it
/// was filed as:
///
///   * **Display.** The `-> [1]` preview and the fenced expanded block print it verbatim, so
///     a BOM'd line 1 rendered a stray glyph and read as a defect in the file rather than in
///     tilth.
///   * **Ranking.** Four terms in `search::rank` test the start of this string —
///     `definition_kind_boost` (`starts_with("pub fn ")` and friends), `exported_api_boost`
///     (`"pub "` / `"export "`), `incidental_text_penalty` (`starts_with("//")`) and
///     `multi_word_boost` (first whole word) — and each reaches it through `str::trim_start`
///     or `str::trim`, neither of which removes U+FEFF. Measured: a line-1 `pub fn`
///     definition scored 1630 against 1760 for the identical BOM-free line, losing the kind
///     boost and the exported-API boost, so it sorted *below* the same code in a file without
///     a BOM. A line-1 `//` comment escaped the incidental-text penalty and sorted above
///     where it belonged.
///
/// So stripping here is not "perturbing the ranking inputs" — the concern #51 was filed
/// with. Those comparisons were already wrong for a BOM'd line; this makes them right.
/// Stripping at the render sites instead would have fixed the glyph and left the mis-ranking
/// in place.
///
/// `trim_end` matches what every construction site already did, so it changes nothing on its
/// own; it lives here only so the two operations cannot drift apart between backends.
pub(crate) fn match_text(line: &str) -> String {
    crate::lang::outline::strip_bom(line).trim_end().to_string()
}

/// Assembled search results before formatting.
#[derive(Debug)]
pub struct SearchResult {
    pub query: String,
    pub scope: PathBuf,
    pub matches: Vec<Match>,
    pub total_found: usize,
    pub definitions: usize,
    pub usages: usize,
    /// Pre-cap subfacet counts. Computed in `symbol::search` and `content::search`
    /// by faceting the merged set before truncation; used by the renderer to print
    /// `displayed/total` headings and the per-facet hidden-count tail line.
    pub facet_totals: FacetTotals,
}

/// Pre-cap counts per subfacet. `content::search` used to leave this at its all-zero
/// default, which made the renderer print a bare `10` for every capped result and
/// suppress every hidden-count tail — a query with 34290 matches was indistinguishable
/// from one with 10. Both search paths now populate it.
#[derive(Debug, Default, Clone, Copy)]
pub struct FacetTotals {
    pub definitions: usize,
    pub implementations: usize,
    pub tests: usize,
    pub usages_local: usize,
    pub usages_cross: usize,
}

/// A single entry in a code outline.
#[derive(Debug)]
pub struct OutlineEntry {
    pub kind: OutlineKind,
    pub name: String,
    pub start_line: u32,
    pub end_line: u32,
    pub signature: Option<String>,
    pub children: Vec<OutlineEntry>,
    pub doc: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OutlineKind {
    Import,
    Function,
    Class,
    Struct,
    Interface,
    TypeAlias,
    Enum,
    Constant,
    Variable,
    ImmutableVariable,
    Export,
    // Property is constructed in lang/outline.rs (property_declaration nodes).
    Property,
    Module,
    // Reserved/roadmap: no tree-sitter grammar currently emits TestSuite nodes.
    #[allow(dead_code)]
    TestSuite,
    // Reserved/roadmap: no tree-sitter grammar currently emits TestCase nodes.
    #[allow(dead_code)]
    TestCase,
}

/// Detect test files by path patterns.
pub(crate) fn is_test_file(path: &std::path::Path) -> bool {
    let s = path.to_string_lossy();
    s.contains(".test.") || s.contains(".spec.") || s.contains("__tests__/")
}

/// Tokens ≈ bytes / 4. Ceiling division, no float.
#[must_use]
pub fn estimate_tokens(byte_len: u64) -> u64 {
    byte_len.div_ceil(4)
}

/// UTF-8 safe string truncation. Never panics on multi-byte characters.
#[must_use]
pub fn truncate_str(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        &s[..s.floor_char_boundary(max)]
    }
}
