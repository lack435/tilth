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
/// Since #57 the stakes are higher than reordering. `search::retain` bounds retention *before*
/// the display cap, keyed on this same score, so a mis-scored line-1 definition can now fall
/// below the retention bound and be dropped from the result set entirely rather than merely
/// sorted low. A silent wrong answer, not a cosmetic one.
///
/// `trim_end` matches what every construction site already did — all nine — so it changes
/// nothing on its own; it lives here only so the two operations cannot drift apart between
/// backends.
///
/// Scope worth stating precisely: this strips U+FEFF from the start of *any* matched line, not
/// only from a genuine byte-order mark at offset 0 of the file. U+FEFF is also a legal
/// zero-width no-break space, so a line elsewhere in a file that begins with one renders here
/// without it while `tilth_read mode=full` still shows it — the kind of cross-surface
/// disagreement `read::read_file`'s own note argues against. Accepted rather than narrowed:
/// distinguishing the two would mean threading a file-offset into every construction site, and
/// the existing BOM helpers in `search::strip` and `read::imports` already take the same
/// line-prefix view. The practical exposure is a ZWNBSP opening a line, which is vanishingly
/// rare next to a BOM opening a file.
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
///
/// Each marker is matched where it actually means something, which is not what a whole-path
/// `contains` does:
///
///   * `.test.` / `.spec.` are **file name** conventions — `auth.test.ts`, `parser.spec.js`. Nothing
///     about a *directory* called `app.test.stuff` says its contents are tests.
///   * `__tests__` is a **directory** convention, so it is matched as a whole path component.
///
/// The old spelling was `s.contains(".test.") || s.contains(".spec.") || s.contains("__tests__/")`
/// against the whole path, which let the directories *above* the project decide: a checkout at
/// `/home/u/proj.spec.v2/` classified every file in it as a test. Not hypothetical for the callers
/// that get absolute paths, and `search::rank` docks a test file **120 points**.
///
/// Matching `__tests__` by component rather than by `"__tests__/"` also fixes it on Windows, where
/// the embedded forward slash meant it never matched at all — so a `__tests__` tree was classified
/// one way on Linux and another on Windows, the kind of platform split `overview`'s
/// `path_bearing_lines_are_identical_across_platforms` exists to refuse. That is a *widening* on
/// Windows, and the only one here: a file under `__tests__` now takes the ranking penalty and the
/// test grouping there, as it always has on Linux.
///
/// **Pass a project-relative path when you have one.** Matching each marker where it means
/// something removes the ancestor exposure for the two file-name markers, but `__tests__` is a
/// directory name and an ancestor called `__tests__` is indistinguishable from a local one without
/// a root. Four of the five callers have a scope and strip it (`search::rank`, `search::grok`,
/// `search::blast`, `search::deps`); `read::outline` has none and passes the path it was given.
/// An earlier version of this note claimed two callers had no scope, which review disproved —
/// `blast` uses its `scope` two lines below each call.
pub(crate) fn is_test_file(path: &std::path::Path) -> bool {
    let name_is_test = path
        .file_name()
        .map(|n| n.to_string_lossy())
        .is_some_and(|n| n.contains(".test.") || n.contains(".spec."));

    // Case-sensitive, which is what the old `contains("__tests__/")` was on Linux — on Windows it
    // matched nothing at all, so there is no prior behaviour there to preserve or depart from.
    // Matching `__TESTS__` would be a widening, and this change is meant to narrow; the cost is
    // that on a case-insensitive volume `__TESTS__` names the same directory and is classified
    // differently. Left as-is deliberately rather than unnoticed.
    //
    // `file_name()` is `None` for a path ending in `..`, which therefore classifies on components
    // alone — `a.test.ts/..` was `true` and is now `false`. No caller builds such a path.
    name_is_test || path.components().any(|c| c.as_os_str() == "__tests__")
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

#[cfg(test)]
mod is_test_file_tests {
    use super::is_test_file;
    use std::path::PathBuf;

    /// Build a path from components so the test says the same thing on both platforms.
    fn p(parts: &[&str]) -> PathBuf {
        parts.iter().collect()
    }

    /// Which of these are tripwires and which are regression guards, since the two are not the
    /// same and a reader cannot tell by looking:
    ///
    ///   * `the_conventions_are_still_recognised` — a tripwire **only on Windows**, where
    ///     `"__tests__/"` never matched a backslash path. It passes under the old spelling on
    ///     Linux, so on CI it guards nothing.
    ///   * `an_ancestor_directory_does_not_make_a_file_a_test` — the reported defect. Tripwire on
    ///     both platforms.
    ///   * `a_partial_component_is_not_the_directory_convention` — tripwire on both, but only
    ///     because of the `my__tests__` cases: `contains("__tests__/")` needed a *trailing*
    ///     separator, so a suffix partial like `__tests__extra` was already `false` and only a
    ///     prefix partial distinguishes the two spellings. Review found this test was originally
    ///     written with the two non-distinguishing cases alone, and passed in every revert.
    ///   * `the_directory_convention_is_case_sensitive`, `an_ordinary_tests_directory_is_unchanged`
    ///     — regression guards. They pass with the fix reverted, by design.
    ///
    /// That accounting exists because the `__tests__` half of this change is guarded on the Linux
    /// runner by exactly one of these — the partial-component test — and it was not obvious.

    /// The conventions themselves must still be recognised. Without this the narrowing below
    /// passes against a function that returns `false` for everything.
    ///
    /// A tripwire on Windows only; see the module note above.
    #[test]
    fn the_conventions_are_still_recognised() {
        for parts in [
            vec!["repo", "src", "auth.test.ts"],
            vec!["repo", "src", "parser.spec.js"],
            vec!["repo", "src", "__tests__", "auth.ts"],
            vec!["repo", "__tests__", "nested", "deep.ts"],
        ] {
            assert!(
                is_test_file(&p(&parts)),
                "stopped recognising a test path: {parts:?}"
            );
        }
    }

    /// The reported defect: a directory *above* the project decided the answer for everything
    /// inside it, because the markers were matched against the whole path.
    ///
    /// `search::rank` docks a test file 120 points, so under the old spelling a checkout at
    /// `~/proj.spec.v2/` sank every file in it relative to a genuine test file beside them, and
    /// relative to everything in an untainted tree the same search reached.
    #[test]
    fn an_ancestor_directory_does_not_make_a_file_a_test() {
        for parts in [
            vec!["home", "u", "proj.spec.v2", "src", "main.rs"],
            vec!["home", "u", "app.test.stuff", "src", "main.rs"],
        ] {
            assert!(
                !is_test_file(&p(&parts)),
                "a directory above the project still decides this file is a test: {parts:?}"
            );
        }
    }

    /// `__tests__` is matched as a whole component, not as a substring.
    ///
    /// The first two cases are the ones that matter, and are the only guard the `__tests__` half of
    /// this change has on the Linux runner: `contains("__tests__/")` required a trailing separator,
    /// so a directory *ending* in `__tests__` matched it and a directory merely *containing* the
    /// text did not. Reverting the component match makes those two fail on either platform. The
    /// last two are regression guards — already `false` under both spellings — kept because they
    /// are the cases a reader assumes are at issue.
    #[test]
    fn a_partial_component_is_not_the_directory_convention() {
        // Spelled with forward slashes deliberately, and not through `p()`. `Path::components`
        // splits on `/` on both platforms, but `contains("__tests__/")` only ever matched a
        // forward-slash path — so this is the one spelling under which reverting the component
        // match fails on the Linux runner as well as here. Without it the `__tests__` half of this
        // change is guarded on Windows only, which review measured and which is invisible locally.
        assert!(
            !is_test_file(std::path::Path::new("repo/my__tests__/a.ts")),
            "a directory ending in `__tests__` matched the old `contains(\"__tests__/\")` and \
             must not match the component test"
        );
        assert!(!is_test_file(&p(&["repo", "my__tests__", "a.ts"])));
        assert!(!is_test_file(&p(&["repo", "x__tests__", "a.ts"])));
        assert!(!is_test_file(&p(&["repo", "not__tests__here", "a.ts"])));
        assert!(!is_test_file(&p(&["repo", "__tests__extra", "a.ts"])));
    }

    /// Case-sensitive, as the previous spelling was. Pinned so the narrowing is not read as a
    /// licence to widen.
    #[test]
    fn the_directory_convention_is_case_sensitive() {
        assert!(!is_test_file(&p(&["repo", "__TESTS__", "a.ts"])));
    }

    /// A plain `test` or `tests` directory is not one of the three markers, and never was.
    #[test]
    fn an_ordinary_tests_directory_is_unchanged() {
        assert!(!is_test_file(&p(&["repo", "tests", "integration.rs"])));
        assert!(!is_test_file(&p(&["repo", "test", "helper.rs"])));
    }
}
