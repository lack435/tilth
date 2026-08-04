//! Blank out macros the C/C++ grammar cannot parse, before it sees them.
//!
//! Two masks, with the same mechanism and two different targets: `mask_export_macros` for
//! dllexport-style macros *inside* a type head, and `mask_annotation_macros` for Unreal
//! reflection annotations, which sit anywhere and derail recovery from a distance. Everything
//! below about spaces, positions and the scanner applies to both.
//!
//! `class MYLIB_API Widget : public Base { … };` — how essentially every Windows C++
//! library spells dllexport — does not parse. tree-sitter-cpp has no way to know
//! `MYLIB_API` is a macro, so it reads the macro as the type name and the real name as
//! a declarator, and the whole definition collapses into error recovery.
//!
//! tilth used to repair that downstream, matching the shapes recovery happened to
//! produce. That cost three separate defects (#16, #49, #52), each a different repair,
//! and error recovery is not a stable contract — the repair for one class already
//! differed with the *length* of its base-class name. This removes the cause instead:
//! overwrite the macro with spaces before parsing, and the grammar sees an ordinary
//! class.
//!
//! # Why spaces rather than deletion
//!
//! Every byte offset, line and column in the masked text matches the original, so a
//! node's position is a position in the real source. Callers parse the masked text and
//! read every piece of text from the original — no offset mapping, and nothing
//! downstream needs to know masking happened. C++ identifiers are ASCII, so replacing
//! one with N spaces is always length-preserving.
//!
//! # What is masked
//!
//! An identifier between a `class` / `struct` / `union` / `enum` keyword and the type
//! name — together with its `(…)` arguments, if it takes any, since blanking
//! `ALIGNAS(16)`'s name alone would strand `(16)` exactly where the type name belongs.
//!
//! Four gates, all of which must pass. The first two are structural; the last two are
//! here because structure alone is provably insufficient.
//!
//! **A brace before the semicolon.** `class Foo bar;` is an ordinary variable
//! declaration of type `Foo` — the same token shape as a macro-prefixed head — and
//! masking it would delete the type and leave `class bar;`. Only a definition has a
//! body.
//!
//! **The type's own name takes no arguments.** `struct RECT GetBounds() { … }` is a
//! function returning an elaborated type: a brace before the semicolon, two
//! identifiers, and an upper-case first one. A type name is never followed by `(`,
//! so that is what separates them.
//!
//! **The braces hold declarations, not an initialiser.** `struct RECT r{0,0,1,1};`
//! declares a brace-initialised *variable*, and is otherwise identical to a definition
//! — see `body_is_type_definition`.
//!
//! **Every masked identifier is macro-shaped**, meaning upper-case, digits and
//! underscores only. A naming convention rather than a structural fact, and the one
//! gate that catches `struct FVector ALIGN16 P{0,0,0};`, whose type name is
//! mixed-case. Every dllexport spelling in the wild is upper-case (`MYLIB_API`,
//! `Q_DECL_EXPORT`, `DLLEXPORT`); an upper-case C++ *type* name is common enough in
//! Win32 (`RECT`, `POINT`, `MSG`) that this gate alone is not sufficient, which is why
//! the two above exist.
//!
//! Masking is refused for the whole head unless every candidate passes, rather than
//! per identifier: a head where only some names look like macros is one this scanner
//! has misread, and guessing which half is right is how a declaration gets corrupted.
//!
//! # Known residuals
//!
//! `struct COLOR c{RED};` — a brace-initialised variable whose initialiser starts with
//! an identifier — is indistinguishable from a definition by the rules above, and is
//! masked. Narrower than what the gates reject, and the pre-masking recovery path
//! mis-read the same shape.
//!
//! A raw string literal with an embedded `"` (`R"(he said "hi". class API X {)"`) can
//! desynchronise the scanner into string content. Harmless: every masked span is an
//! identifier, so no token boundary outside the literal can move, and all text is read
//! from the original. Worth knowing before adding a rule that blanks anything wider.

/// Rewrite `content` with C/C++ type-head export macros replaced by spaces.
///
/// Returns `None` when there is nothing to mask, which is the overwhelmingly common
/// case — callers keep using the original and pay only the scan.
pub fn mask_export_macros(content: &str) -> Option<String> {
    let bytes = content.as_bytes();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut scan = Scanner { bytes, pos: 0 };

    while let Some(start) = scan.next_type_keyword() {
        if let Some(idents) = scan.read_head() {
            if let Some((name, candidates)) = idents.split_last() {
                // A name carrying an argument list is a function returning an
                // elaborated type — `struct RECT GetBounds() { … }`. Valid C, and
                // token-for-token a macro-prefixed head; masking it deletes the return
                // type. A type's own name is never followed by `(`.
                let name_is_a_type = name.span_end == name.name_end;
                // All candidates must look like macros, or none are masked: a head
                // where only some names do is one this scanner has misread, and
                // guessing which half is right is how a declaration gets corrupted.
                let all_macros = !candidates.is_empty()
                    && candidates
                        .iter()
                        .all(|i| is_macro_shaped(&bytes[i.start..i.name_end]));
                if name_is_a_type && all_macros {
                    spans.extend(candidates.iter().map(|i| (i.start, i.span_end)));
                }
            }
        }
        debug_assert!(scan.pos > start, "scanner must always advance");
    }

    if spans.is_empty() {
        return None;
    }
    blank_spans(content.as_bytes(), &spans)
}

/// Overwrite `spans` with spaces, **leaving control bytes where they are**.
///
/// Preserving newlines is not a detail — it is the whole contract. Callers parse the masked
/// text and then read every piece of text from the *original* via `lines[row]`, so a mask that
/// removes a line makes every row below it index the wrong source line. The failure is silent
/// and it is not confined to the masked construct: a `UMETA(X = "a",\n Y = "b")` wrapped across
/// two lines shifted everything after it, so the `struct` two lines below was reported at the
/// wrong line, under the *next* struct's name, and the last one vanished.
///
/// A span can legitimately contain a newline — a wrapped annotation or a
/// `UE_DEPRECATED(5.4,\n "use Foo")` in a type head — so refusing multi-line spans would give
/// up the masking exactly where it is needed. Keeping the control bytes gives both: the tokens
/// are gone and the line structure is untouched.
///
/// `\r` is preserved for the same reason on CRLF files, where dropping it would shorten a row
/// that tree-sitter still counts — the mechanism behind `clamp_col`'s existence.
fn blank_spans(bytes: &[u8], spans: &[(usize, usize)]) -> Option<String> {
    // Byte-wise, then revalidated: every replacement is an ASCII space, so this cannot split a
    // char boundary — but an `unsafe` block to save one linear validation of a source file is a
    // poor trade. ASCII bytes never occur inside a multi-byte UTF-8 sequence, so the scanner
    // cannot have picked a span that starts mid-character either.
    let mut out = bytes.to_vec();
    for &(start, end) in spans {
        for b in &mut out[start..end] {
            if !b.is_ascii_control() {
                *b = b' ';
            }
        }
    }
    String::from_utf8(out).ok()
}

/// Annotation macros blanked wherever they appear, arguments included.
///
/// `UMETA` is here because of what it does to a `UENUM`, and the interaction is worth
/// stating precisely because neither half is broken alone:
///
/// ```text
/// UENUM(BlueprintType)                  // alone: parses, macro becomes an expression_statement
/// enum class EMode : uint8
/// {
///     A UMETA(DisplayName = "Alpha"),   // alone: parses, enum_specifier intact
///     B UMETA(DisplayName = "Beta")
/// };
/// ```
///
/// Together, tree-sitter-cpp absorbs the `enum` keyword into an `ERROR` node — and that node
/// does not stop at the enum. How far it reaches depends on what follows, and at the extreme
/// it is the whole file: `walk_top_level` reads the root's children, so a root whose only
/// child is the `ERROR` yields an empty outline and nothing findable by name.
///
/// Measured over the 262 headers in an Unreal Engine `Engine/Source` tree that contain a
/// `UMETA`, masking took the total outline entry count from 15 325 to 18 755. The
/// distribution matters more than the total: most files gained a handful, and the worst had
/// lost nearly everything — one component header went from 11 entries to 521, another from 1
/// to 143. Six files *lost* entries, all of them artifacts of the misparse being removed:
/// `fn UMETA`, `class <anonymous>` and `prop BlueprintType` placeholders, replaced by the real
/// enums and structs that had been hiding behind them.
///
/// Blanking `UMETA` removes the interaction, and costs nothing: it is pure editor metadata —
/// display names and tooltips — so no symbol anybody searches for lives inside one. It also
/// removes the artifact `read::outline::code` deduplicates, where each annotated enumerator
/// parsed as a `function_declarator` and the outline gained one `fn UMETA` per enumerator.
///
/// The **deprecation attributes** are here for a different failure, on the same principle. A
/// deprecation attribute before a class member —
///
/// ```text
/// namespace NetworkingPrivate
/// {
/// struct FRepPropertyDescriptor
/// {
///     UE_DEPRECATED(5.2, "No longer used")   // ← attribute before the constructor
///     FRepPropertyDescriptor(const FProperty* Property) { … }
/// };
/// }
/// ```
///
/// derails the enclosing type: tree-sitter-cpp reads `UE_DEPRECATED(…)` as a member
/// declaration, and inside a namespace the recovery is *not* local — the `struct` keyword and
/// tag are absorbed into an `ERROR`, so the `struct_specifier` disappears and only the
/// surviving constructor is findable. `read`/`search`/`grok` then report the type's own line as
/// a *usage* and resolve the type to a member (#130). Like `UMETA`, `UE_DEPRECATED` is pure
/// metadata — a version and a message — with no symbol inside, so blanking every invocation is
/// length-preserving and loses nothing. It composes with `mask_export_macros`, which already
/// blanks the same macro in a *type head* (`class UE_DEPRECATED(…) API Widget`); the two never
/// fight, since both overwrite the identical span with spaces.
///
/// **Why `UE_DEPRECATED` and not "every deprecation macro".** The misparse is triggered by the
/// *shape of the arguments*, not by the macro being a deprecation macro: a **numeric first
/// argument** — `MACRO(5.2, …)` — is what tree-sitter-cpp cannot place in member-declarator
/// position. Isolated on synthetic input: `M(x, "s")`, `M("a", "b")` and `M("s")` all parse; only
/// `M(1, …)` / `M(1.0, …)` collapse the enclosing type. `UE_DEPRECATED(Version, Message)` is the
/// one deprecation attribute spelled with a leading numeric version, and a scan of
/// `Engine/Source/Runtime` finds no other `*DEPRECATED*(<number>, …)` invocation. The
/// single-*message* attributes of other libraries — `BOOST_DEPRECATED("m")`,
/// `ABSL_DEPRECATED("m")`, `GTEST_INTERNAL_DEPRECATED("m")` — take a lone string and do **not**
/// misparse, so masking them recovers nothing (measured: zero change over `GoogleTest` and Boost
/// header trees) and is deliberately not done.
///
/// Two `DEPRECATED`-named kinds must never be added, because blanking them would be wrong rather
/// than merely useless: **declaration-generating** macros — above all Slate's
/// `SLATE_ARGUMENT_DEPRECATED` / `SLATE_ATTRIBUTE_DEPRECATED`, where
/// `SLATE_ATTRIBUTE_DEPRECATED(Type, Name, …)` expands to a member variable and function named
/// `Name`, so blanking it deletes a real symbol — and **file-scope pragma emitters**
/// (`UE_DEPRECATED_HEADER`, `UE_DEPRECATED_MACRO`), which sit at file scope, not before a member.
///
/// The plain `UCLASS`/`USTRUCT`/`UPROPERTY`/`UFUNCTION` macros are likewise absent: each leaves an
/// `ERROR` or `expression_statement` behind, but recovery is local — the declaration after them
/// still parses — so masking them would change tested output for no correctness gain.
const ANNOTATION_MACROS: [&[u8]; 2] = [
    // Enumerator editor metadata.
    b"UMETA",
    // The one deprecation attribute whose `(numeric version, message)` shape triggers the
    // member misparse (see above); other libraries' single-message ones do not, so are absent.
    b"UE_DEPRECATED",
];

/// Newlines an annotation's argument list may cross before the span is refused.
///
/// Three, because a wrapped `UMETA(DisplayName = "…",\n ToolTip = "…")` is ordinary and a
/// clang-formatted one can take a third line, while an unterminated `UMETA(` reaching a stray
/// `)` in unrelated code is unbounded. See the call site for why an over-long span is dropped
/// rather than clipped.
const MAX_ANNOTATION_LINES: usize = 3;

/// Newlines in `span`. Named rather than inlined so the bound above reads as a line count.
fn bytecount_newlines(span: &[u8]) -> usize {
    memchr::memchr_iter(b'\n', span).count()
}

/// Rewrite `content` with `ANNOTATION_MACROS` invocations replaced by spaces.
///
/// `None` when there is nothing to mask, which is every file that is not UE-flavoured C++.
/// Length- and position-preserving in the same way as `mask_export_macros`, and composable
/// with it for that reason: both only ever overwrite ASCII spans with ASCII spaces, so
/// neither can move a byte offset the other recorded.
pub fn mask_annotation_macros(content: &str) -> Option<String> {
    let bytes = content.as_bytes();
    // The overwhelmingly common case is a file with no annotation at all, and this scan runs
    // on every C/C++ file parsed. One SIMD pass rejects those without touching the scanner.
    if !ANNOTATION_MACROS
        .iter()
        .any(|m| memchr::memmem::find(bytes, m).is_some())
    {
        return None;
    }

    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut scan = Scanner { bytes, pos: 0 };

    while scan.pos < bytes.len() {
        if scan.skip_trivia() {
            continue;
        }
        let b = bytes[scan.pos];
        if !is_ident_start(b) {
            scan.pos += 1;
            continue;
        }
        let start = scan.pos;
        let name_end = scan.ident_end(start);
        scan.pos = name_end;
        if !ANNOTATION_MACROS.contains(&&bytes[start..name_end]) {
            continue;
        }
        // Only an *invocation* is masked. A bare `UMETA` identifier — in a comment the
        // scanner already skipped, or as somebody's variable — is left alone, because
        // blanking a name that is not a macro would delete real code.
        let after_name = scan.pos;
        while scan.pos < bytes.len() && scan.skip_trivia() {}
        if bytes.get(scan.pos) != Some(&b'(') {
            scan.pos = after_name;
            continue;
        }
        if scan.skip_balanced(b'(', b')').is_some() {
            // Through the closing paren, so the arguments go too — including the nested
            // parens of `UMETA(ToolTip = "Fire (primary)")`, which `skip_balanced` counts
            // and `skip_trivia` keeps out of, since the inner ones are inside a literal.
            //
            // Bounded by line count, because an *unterminated* `UMETA(` does not fail — it
            // runs to whatever stray `)` appears next, which can be many declarations later,
            // and every identifier in between would be blanked. A real annotation is a
            // display name or a tooltip; it wraps onto a second or third line and no further.
            // Past the bound the span is dropped rather than truncated: a half-blanked
            // argument list is a worse input to the grammar than an unmasked one.
            let lines_spanned = bytecount_newlines(&bytes[start..scan.pos]);
            if lines_spanned <= MAX_ANNOTATION_LINES {
                spans.push((start, scan.pos));
            }
        } else {
            scan.pos = after_name;
        }
    }

    if spans.is_empty() {
        return None;
    }
    blank_spans(bytes, &spans)
}

struct Scanner<'a> {
    bytes: &'a [u8],
    pos: usize,
}

/// One identifier in a type head.
///
/// `span_end` reaches past `name_end` only for an argument-taking macro, where the
/// parenthesised arguments have to be blanked along with the name. The shape gate
/// reads `start..name_end`; the mask covers `start..span_end`.
#[derive(Clone, Copy)]
struct HeadIdent {
    start: usize,
    name_end: usize,
    span_end: usize,
}

/// Keywords that can introduce a type definition. `enum` is included for
/// `enum class API Mode : uint8_t { … }`, whose macro sits in the same place.
const TYPE_KEYWORDS: [&[u8]; 4] = [b"class", b"struct", b"union", b"enum"];

/// Contextual keywords that end the name list without being part of it. Without this,
/// `class API Widget final : public Base` reads `final` as the type name and masks
/// `Widget`.
const HEAD_TERMINATORS: [&[u8]; 3] = [b"final", b"sealed", b"abstract"];

impl Scanner<'_> {
    /// Advance to just past the next type keyword outside a comment or literal.
    /// Returns the keyword's start offset.
    fn next_type_keyword(&mut self) -> Option<usize> {
        while self.pos < self.bytes.len() {
            if self.skip_trivia() {
                continue;
            }
            let b = self.bytes[self.pos];
            if is_ident_start(b) {
                let start = self.pos;
                let end = self.ident_end(start);
                let word = &self.bytes[start..end];
                self.pos = end;
                if TYPE_KEYWORDS.contains(&word) {
                    return Some(start);
                }
                continue;
            }
            self.pos += 1;
        }
        None
    }

    /// Read the identifiers between the keyword just consumed and the type's body.
    ///
    /// Returns `None` unless the head is a definition — a `{` reached before any `;`.
    fn read_head(&mut self) -> Option<Vec<HeadIdent>> {
        let mut idents: Vec<HeadIdent> = Vec::new();
        loop {
            if self.pos >= self.bytes.len() {
                return None;
            }
            if self.skip_trivia() {
                continue;
            }
            let b = self.bytes[self.pos];
            match b {
                // The body: this head defines a type, if the braces hold declarations
                // rather than an initialiser.
                b'{' => return self.body_is_type_definition().then_some(idents),
                // A declaration, not a definition — `class Foo bar;`, `class Fwd;`.
                // `,` means a declarator list (`class Foo a, b;`); a `(` with no
                // identifier before it is a shape this scanner does not understand.
                b';' | b',' | b')' | b'=' | b'(' => return None,
                // A base-class clause. Nothing after it can be the type's own name, so
                // stop collecting, but keep scanning for the brace that proves this is
                // a definition.
                b':' => {
                    return self.skip_to_body().then_some(idents);
                }
                // `[[nodiscard]]` and friends parse natively; step over them.
                b'[' => {
                    self.skip_balanced(b'[', b']')?;
                }
                _ if is_ident_start(b) => {
                    let start = self.pos;
                    let name_end = self.ident_end(start);
                    let word = &self.bytes[start..name_end];
                    self.pos = name_end;
                    if HEAD_TERMINATORS.contains(&word) {
                        return self.skip_to_body().then_some(idents);
                    }
                    // `enum class API Mode` — step over the second keyword rather than
                    // counting it as a name.
                    if TYPE_KEYWORDS.contains(&word) {
                        continue;
                    }
                    // An argument-taking macro — `ALIGNAS(16)`,
                    // `UE_DEPRECATED(5.4, "gone")` — must be blanked *with* its
                    // arguments. Blanking the name alone strands the argument list
                    // exactly where a type name belongs, which is worse than not
                    // masking: the head then resembles nothing the recovery fallback
                    // recognises either, so it falls between the two paths.
                    let span_end = self.span_end_after(name_end)?;
                    idents.push(HeadIdent {
                        start,
                        name_end,
                        span_end,
                    });
                }
                // Anything else in a type head (`<`, `*`, `&`, …) means this is not the
                // simple shape an export macro produces. Give up rather than guess.
                _ => return None,
            }
        }
    }

    /// End of an identifier's span, extended over a following `(…)` argument list.
    ///
    /// Leaves `self.pos` just past whatever the span covers.
    fn span_end_after(&mut self, name_end: usize) -> Option<usize> {
        let before_trivia = self.pos;
        self.skip_trivia();
        if self.bytes.get(self.pos) == Some(&b'(') {
            self.skip_balanced(b'(', b')')?;
            return Some(self.pos);
        }
        self.pos = before_trivia;
        Some(name_end)
    }

    /// True when the `{` at the cursor opens a type body rather than an initialiser.
    ///
    /// `struct RECT r{0,0,1,1};` is a brace-initialised *variable* whose head is
    /// token-for-token a macro-prefixed definition, and Win32 spells every struct tag
    /// in upper case (`RECT`, `POINT`, `MSG`) so the macro-shape gate does not catch
    /// it. Masking it deletes the type and promotes the variable to one. A member
    /// declaration always starts with a name or a keyword, so a literal or an
    /// immediate `}` means an initialiser.
    ///
    /// Does not move the cursor.
    fn body_is_type_definition(&mut self) -> bool {
        let save = self.pos;
        self.pos += 1; // step over `{`
        self.skip_trivia();
        let verdict = match self.bytes.get(self.pos) {
            // `{}` is genuinely ambiguous — an empty definition and an empty
            // initialiser are spelled identically. Declining leaves an empty exported
            // type to the recovery fallback, which is where it was handled before.
            None | Some(b'}') => false,
            Some(&b) => {
                !(b.is_ascii_digit()
                    || matches!(b, b'"' | b'\'' | b'-' | b'+' | b'{' | b'&' | b'*'))
            }
        };
        self.pos = save;
        verdict
    }

    /// Scan forward for the `{` that opens a type body. False if a `;` comes first,
    /// which means a declaration rather than a definition.
    fn skip_to_body(&mut self) -> bool {
        while self.pos < self.bytes.len() {
            if self.skip_trivia() {
                continue;
            }
            match self.bytes[self.pos] {
                b'{' => return true,
                b';' => return false,
                b'(' => {
                    if self.skip_balanced(b'(', b')').is_none() {
                        return false;
                    }
                }
                _ => self.pos += 1,
            }
        }
        false
    }

    /// Step over a bracketed run, honouring nesting. `None` if it never closes.
    fn skip_balanced(&mut self, open: u8, close: u8) -> Option<()> {
        let mut depth = 0usize;
        while self.pos < self.bytes.len() {
            if self.skip_trivia() {
                continue;
            }
            let b = self.bytes[self.pos];
            self.pos += 1;
            if b == open {
                depth += 1;
            } else if b == close {
                // Unreachable via the two call sites, which both check for the opening
                // byte first — but `outline::generate` is a fuzz target, and a wrapping
                // decrement here would spin to the end of the file.
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(());
                }
            }
        }
        None
    }

    /// Step over whitespace, comments and literals. True if anything was consumed.
    ///
    /// Literals are skipped so a `class` keyword inside a string or comment cannot
    /// start a head. Masking inside one would not change the parse — the text keeps
    /// its length either way — but it would put spaces in a position the caller may
    /// later read text from.
    fn skip_trivia(&mut self) -> bool {
        let start = self.pos;
        while let Some(&b) = self.bytes.get(self.pos) {
            match b {
                b' ' | b'\t' | b'\r' | b'\n' | b'\\' => self.pos += 1,
                b'/' if self.bytes.get(self.pos + 1) == Some(&b'/') => {
                    while self.pos < self.bytes.len() && self.bytes[self.pos] != b'\n' {
                        self.pos += 1;
                    }
                }
                b'/' if self.bytes.get(self.pos + 1) == Some(&b'*') => {
                    self.pos += 2;
                    while self.pos < self.bytes.len() {
                        if self.bytes[self.pos] == b'*'
                            && self.bytes.get(self.pos + 1) == Some(&b'/')
                        {
                            self.pos += 2;
                            break;
                        }
                        self.pos += 1;
                    }
                }
                b'"' => self.skip_literal(b),
                // A `'` after an identifier byte is a digit separator (`1'000`), not a
                // character literal. Reading it as one swallowed the rest of the line,
                // so any head after it on that line was never seen.
                b'\'' if !self.prev_is_ident_byte() => self.skip_literal(b),
                // A preprocessor directive can hold anything, including a bare `class`
                // in a macro definition. Skip the whole logical line.
                b'#' => {
                    self.pos += 1;
                    while self.pos < self.bytes.len() {
                        match self.bytes[self.pos] {
                            // A line continuation. `\r\n` has to be consumed whole, or
                            // the `\n` ends the directive scan one line early — and
                            // CRLF is the norm on the platform this feature targets.
                            b'\\' => {
                                self.pos += 1;
                                if self.bytes.get(self.pos) == Some(&b'\r') {
                                    self.pos += 1;
                                }
                                if self.bytes.get(self.pos) == Some(&b'\n') {
                                    self.pos += 1;
                                }
                            }
                            b'\n' => break,
                            _ => self.pos += 1,
                        }
                    }
                }
                _ => break,
            }
        }
        self.pos > start
    }

    /// True when the byte before the cursor could end an identifier or number.
    fn prev_is_ident_byte(&self) -> bool {
        self.pos
            .checked_sub(1)
            .and_then(|i| self.bytes.get(i))
            .is_some_and(|&b| is_ident_continue(b))
    }

    /// Step over a `"…"` or `'…'` literal, honouring backslash escapes.
    fn skip_literal(&mut self, quote: u8) {
        self.pos += 1;
        while self.pos < self.bytes.len() {
            match self.bytes[self.pos] {
                b'\\' => self.pos += 2,
                b if b == quote => {
                    self.pos += 1;
                    return;
                }
                b'\n' => return,
                _ => self.pos += 1,
            }
        }
    }

    fn ident_end(&self, start: usize) -> usize {
        let mut end = start;
        while end < self.bytes.len() && is_ident_continue(self.bytes[end]) {
            end += 1;
        }
        end
    }
}

/// True for an identifier spelled the way export macros are: upper-case letters,
/// digits and underscores, with at least one letter. `MYLIB_API`, `Q_DECL_EXPORT`,
/// `DLLEXPORT` — but not `FVector`, `Widget` or `_`.
fn is_macro_shaped(ident: &[u8]) -> bool {
    ident.iter().any(u8::is_ascii_uppercase)
        && ident
            .iter()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || *b == b'_')
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[cfg(test)]
mod tests {
    use super::{mask_annotation_macros, mask_export_macros};

    /// What masking must never do: change a byte offset. Every downstream position —
    /// outline ranges, search line numbers, edit anchors — is read straight off the
    /// tree built from the masked text and used against the original.
    ///
    /// **Line count is checked here as well as length, and equal length does not imply it.**
    /// Blanking a newline keeps the byte count identical while deleting a row, so a mask can
    /// satisfy every other assertion in this file and still make the tree disagree with the
    /// source about which line anything is on. That is exactly what the annotation mask did to
    /// a wrapped `UMETA(…,\n …)`: two structs below it were reported at the wrong lines under
    /// each other's names, and a third disappeared. Length alone was the assertion that let it
    /// through.
    fn assert_positions_preserved(src: &str, masked: &str) {
        assert_eq!(
            src.len(),
            masked.len(),
            "masking changed the length:\n{src:?}\n{masked:?}"
        );
        assert_eq!(
            src.lines().count(),
            masked.lines().count(),
            "masking changed the line count — every row below the mask now indexes the wrong \
             source line:\n{src:?}\n{masked:?}"
        );
        for (i, (a, b)) in src.bytes().zip(masked.bytes()).enumerate() {
            assert!(
                a == b || b == b' ',
                "byte {i} changed to something other than a space: {a:?} -> {b:?}"
            );
        }
    }

    /// Every byte the mask changed must have been part of an identifier or its
    /// argument list — never a newline, which would move every line below it.
    fn assert_only_head_bytes_blanked(src: &str, masked: &str) {
        for (i, (a, b)) in src.bytes().zip(masked.bytes()).enumerate() {
            if a != b {
                assert_eq!(b, b' ', "byte {i} became {b:?}, not a space");
                assert!(
                    !a.is_ascii_control(),
                    "byte {i} was control byte {a:?}; blanking it moves every line below"
                );
            }
        }
    }

    #[track_caller]
    fn masks_to(src: &str, expected: &str) {
        let masked = mask_export_macros(src).unwrap_or_else(|| panic!("nothing masked in {src:?}"));
        assert_positions_preserved(src, &masked);
        assert_only_head_bytes_blanked(src, &masked);
        assert_eq!(masked, expected, "for {src:?}");
    }

    #[track_caller]
    fn unchanged(src: &str) {
        assert!(
            mask_export_macros(src).is_none(),
            "must not mask {src:?}, got {:?}",
            mask_export_macros(src)
        );
    }

    #[test]
    fn masks_export_macros_in_type_definitions() {
        masks_to(
            "class API Widget { int X; };",
            "class     Widget { int X; };",
        );
        masks_to(
            "class MYLIB_API Widget : public Base { void W(); };",
            "class           Widget : public Base { void W(); };",
        );
        masks_to(
            "class API Widget final : public Base { void W(); };",
            "class     Widget final : public Base { void W(); };",
        );
        // Multiple inheritance — issue #49's shape.
        masks_to(
            "class API W : public P1, public P2 { void W(); };",
            "class     W : public P1, public P2 { void W(); };",
        );
        masks_to(
            "struct API Point { int X; };",
            "struct     Point { int X; };",
        );
        masks_to("union API U { int A; };", "union     U { int A; };");
        masks_to(
            "enum class API Mode : uint8_t { On, Off };",
            "enum class     Mode : uint8_t { On, Off };",
        );
        // Two macros in one head.
        masks_to(
            "class API DEPRECATED Widget { int X; };",
            "class                Widget { int X; };",
        );
    }

    /// The gate that matters most. Each of these is valid C++ that a token-shape rule
    /// alone would rewrite into something with a different meaning.
    #[test]
    fn never_masks_a_declaration() {
        // An elaborated type specifier declaring a variable. Masking would leave
        // `class     bar;` — the type deleted and `bar` promoted to a type name.
        unchanged("class Foo bar;");
        unchanged("struct Foo bar;");
        // Forward declarations and type references have no body.
        unchanged("class Fwd;");
        unchanged("class Fwd* Global;");
        unchanged("class API Fwd;");
        // An attribute macro between the type and the *variable* name, with a brace
        // initialiser: a body, and two candidate identifiers. Only the macro-shape gate
        // stops this one, since `FVector` would otherwise be masked as a macro.
        unchanged("struct FVector ALIGN16 P{0,0,0};");
        unchanged("struct FVector P{0,0,0};");
        unchanged("enum Color c{RED};");
        // Ordinary definitions have nothing to mask.
        unchanged("class Widget { int X; };");
        unchanged("enum class Mode { On, Off };");
        // An initialiser after `=` is a declaration however it is spelled.
        unchanged("struct Foo f = {1,2};");
    }

    /// A `class` keyword that is not code must not start a head — and a macro
    /// *definition* naming a class is exactly where one appears.
    #[test]
    fn ignores_keywords_outside_code() {
        unchanged("// class API Widget {\nint x;\n");
        unchanged("/* class API Widget { */\nint x;\n");
        unchanged("const char* s = \"class API Widget {\";\n");
        unchanged("#define DECLARE(n) class API n {\nint x;\n");
        // A directive continued across lines is still one directive.
        unchanged("#define DECLARE(n) \\\n    class API n {\nint x;\n");
    }

    /// Masking is all-or-nothing per head: a head where only some candidates look like
    /// macros is one the scanner has misread.
    #[test]
    fn refuses_a_head_with_a_non_macro_candidate() {
        unchanged("struct FVector ALIGN16 Position { int X; };");
        masks_to(
            "struct ALIGN16 Position { int X; };",
            "struct         Position { int X; };",
        );
    }

    /// An argument-taking macro must be blanked *with* its arguments. Blanking the
    /// name alone strands the argument list where a type name belongs — worse than not
    /// masking, because the head no longer resembles anything the recovery fallback
    /// recognises either, so it falls between the two paths.
    #[test]
    fn masks_argument_taking_macros_with_their_arguments() {
        // `ALIGNAS(16)` is 11 bytes, plus the space either side of it.
        masks_to(
            "class ALIGNAS(16) Widget { int X; };",
            &format!("class{}Widget {{ int X; }};", " ".repeat(13)),
        );
        // The real-world shape: a deprecation macro carrying a string, next to an
        // export macro. A `)` inside the string must not end the argument list.
        masks_to(
            "class DEPRECATED(5.4, \"gone)\") MYLIB_API Widget : public Base { void W(); };",
            &format!(
                "class{}Widget : public Base {{ void W(); }};",
                " ".repeat(36)
            ),
        );
        // A lower-case one is not macro-shaped, so the whole head declines.
        unchanged("class __declspec(dllexport) Widget { int X; };");
    }

    /// An all-caps identifier is not always a macro. Win32 spells every struct tag that
    /// way (`RECT`, `POINT`, `MSG`), so an elaborated type specifier naming one has the
    /// exact token shape of a macro-prefixed head — and masking it deletes the type.
    #[test]
    fn never_masks_an_upper_case_type_tag() {
        // A function returning an elaborated type. `GetBounds` is followed by `(`,
        // which no type name ever is.
        unchanged("struct RECT GetBounds() { return r; }");
        unchanged("struct POINT Origin() { return p; }");
        // Brace-initialised variables of an upper-case tag type.
        unchanged("struct RECT r{0,0,1,1};");
        unchanged("struct RECT Bounds{};");
        unchanged("void f() { struct RECT bounds{0,0,1,1}; }");
    }

    /// The `#define` guard has to hold on the line endings this platform actually uses.
    #[test]
    fn ignores_keywords_in_directives_with_crlf() {
        unchanged("#define DECLARE(n) \\\r\n    class API n {\r\nint x;\r\n");
        unchanged("#define DECLARE(n) class API n {\r\nint x;\r\n");
    }

    /// A digit separator is not a character literal. Treating it as one swallowed the
    /// rest of the line, so a head after it on the same line was never seen.
    #[test]
    fn digit_separators_do_not_swallow_the_line() {
        masks_to(
            "constexpr int K = 1'000; class API Widget { int X; };",
            "constexpr int K = 1'000; class     Widget { int X; };",
        );
    }

    /// The invariant everything downstream rests on, asserted over adversarial input
    /// rather than argued: the scan terminates, the length never changes, and no byte
    /// that carries position information is ever blanked.
    ///
    /// Deterministic rather than randomised — a seeded LCG over a C++-flavoured
    /// alphabet, so a failure is reproducible and CI cannot go intermittently red.
    #[test]
    fn masking_is_total_and_position_preserving() {
        const ALPHABET: &[u8] = b"class struct union enum API_X ab(){};:,*&<>\"'\\\n\r\t/#=1";
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..20_000 {
            let len = (next() % 120) as usize;
            let src: String = (0..len)
                .map(|_| ALPHABET[(next() % ALPHABET.len() as u64) as usize] as char)
                .collect();
            // Only well-formed UTF-8 reaches this — the alphabet is ASCII.
            if let Some(masked) = mask_export_macros(&src) {
                assert_positions_preserved(&src, &masked);
                assert_only_head_bytes_blanked(&src, &masked);
            }
        }
    }

    #[test]
    fn leaves_files_without_macros_untouched() {
        unchanged("int main() { return 0; }\n");
        unchanged("#include <vector>\nnamespace n { class A { int x; }; }\n");
        unchanged("");
    }

    /// Several heads in one file, including one that must not be masked.
    #[test]
    fn masks_every_head_independently() {
        let src = "\
class API A { int X; };
class B bar;
struct OTHER_API C : public A { int Y; };
";
        let masked = mask_export_macros(src).expect("two heads mask");
        assert_positions_preserved(src, &masked);
        assert!(masked.contains("class     A { int X; };"), "{masked}");
        assert!(masked.contains("class B bar;"), "{masked}");
        assert!(masked.contains("struct           C : public A"), "{masked}");
    }

    // -- annotation macros -------------------------------------------------------------

    #[track_caller]
    fn annotation_masks_to(src: &str, expected: &str) {
        let masked =
            mask_annotation_macros(src).unwrap_or_else(|| panic!("nothing masked in {src:?}"));
        assert_positions_preserved(src, &masked);
        // The check this helper originally skipped, which is how the newline bug shipped.
        assert_only_head_bytes_blanked(src, &masked);
        assert_eq!(masked, expected, "for {src:?}");
    }

    /// A wrapped annotation — the shape that broke. The mask must remove the tokens and leave
    /// the line structure exactly as it found it, so the declarations below keep their lines.
    #[test]
    fn masks_a_wrapped_umeta_without_moving_any_line() {
        let src = "\
enum class EMode : uint8
{
    A UMETA(X = \"a\",
            Y = \"b\"),
    B
};
struct FAlpha { int A; };
struct FBravo { int B; };
";
        let masked = mask_annotation_macros(src).expect("wrapped annotation masks");
        assert_positions_preserved(src, &masked);
        assert_only_head_bytes_blanked(src, &masked);
        assert_eq!(masked.matches("UMETA").count(), 0, "{masked}");
        // The load-bearing consequence, stated as line identity rather than as a count.
        for (i, (a, b)) in src.lines().zip(masked.lines()).enumerate() {
            if !a.contains("UMETA") && !a.trim_start().starts_with("Y =") {
                assert_eq!(a, b, "line {} moved or changed", i + 1);
            }
        }
    }

    /// The export mask has the same newline hazard, and no test covered it.
    ///
    /// `span_end_after` can produce a span containing a `\n` — a `UE_DEPRECATED(5.4,\n "use
    /// Foo")` in a type head is the ordinary spelling — and the unconditional fill this used
    /// before `blank_spans` blanked it. Measured on the shape below: `Widget` resolved to no
    /// definition at all, and a usage was labelled with a scope name synthesised from the
    /// desynced row (a fragment of the deprecation string). The 20 000-iteration fuzz never
    /// produced a maskable head with a newline inside its parens, which is why this is written
    /// as a fixture rather than left to it.
    #[test]
    fn masks_a_wrapped_export_macro_without_moving_any_line() {
        let src = "\
class UE_DEPRECATED(5.4,
    \"use Foo\") MYLIB_API Widget : public Base { void W(); };
struct FTrailing { int X; };
";
        let masked = mask_export_macros(src).expect("wrapped head masks");
        assert_positions_preserved(src, &masked);
        assert_only_head_bytes_blanked(src, &masked);
        assert!(masked.contains("Widget"), "type name blanked:\n{masked}");
        assert!(
            masked.contains("struct FTrailing"),
            "the declaration below moved or was blanked:\n{masked}"
        );
    }

    /// An unterminated invocation runs to whatever `)` comes next, which is unbounded. The
    /// span is refused past `MAX_ANNOTATION_LINES` rather than blanking unrelated code.
    #[test]
    fn refuses_an_unterminated_annotation_that_would_run_away() {
        let src = "\
enum class EMode : uint8
{
    A UMETA(DisplayName = \"a\",
    B
};
struct FAlpha { int A; };
void Helper(int x);
struct FBravo { int B; };
int Tail(void) { return 0; }
";
        // Either nothing masks, or whatever did masks without touching the code below.
        if let Some(masked) = mask_annotation_macros(src) {
            assert_positions_preserved(src, &masked);
            assert_only_head_bytes_blanked(src, &masked);
            for name in ["FAlpha", "Helper", "FBravo", "Tail"] {
                assert!(
                    masked.contains(name),
                    "runaway span blanked {name}:\n{masked}"
                );
            }
        }
    }

    /// Blank exactly the invocation — name through closing paren — and nothing either side.
    /// The expected string is built rather than typed so it states the rule instead of a
    /// space count nobody can verify by eye.
    #[track_caller]
    fn blanks_only(prefix: &str, invocation: &str, suffix: &str) {
        let src = format!("{prefix}{invocation}{suffix}");
        let expected = format!("{prefix}{}{suffix}", " ".repeat(invocation.len()));
        annotation_masks_to(&src, &expected);
    }

    #[test]
    fn masks_a_umeta_invocation_with_its_arguments() {
        blanks_only("    EPistol ", "UMETA(DisplayName = \"Pistol\")", ",\n");
    }

    /// The argument list is matched by balanced parens, not by "up to the next `)`", so a
    /// tooltip containing punctuation does not cut the mask short and leave a stray `)`.
    #[test]
    fn masks_a_umeta_whose_argument_contains_parentheses() {
        blanks_only("    A ", "UMETA(ToolTip = \"Fire (primary)\")", ",\n");
    }

    /// Every enumerator in a list, not just the first — the shape that actually occurs.
    #[test]
    fn masks_every_umeta_in_an_enumerator_list() {
        let src = "\
UENUM(BlueprintType)
enum class EAmmo : uint8
{
    EPistol UMETA(DisplayName = \"Pistol\"),
    ERifle  UMETA(DisplayName = \"Rifle\")
};
";
        let masked = mask_annotation_macros(src).expect("two annotations mask");
        assert_positions_preserved(src, &masked);
        assert_eq!(masked.matches("UMETA").count(), 0, "{masked}");
        // The enumerators themselves, the enum head and every newline survive.
        assert!(masked.contains("    EPistol "), "{masked}");
        assert!(masked.contains("    ERifle  "), "{masked}");
        assert!(masked.contains("enum class EAmmo : uint8"), "{masked}");
        assert_eq!(masked.lines().count(), src.lines().count());
    }

    /// Only invocations. A bare identifier is somebody's code, and blanking it would delete
    /// a real name — the mask must be able to tell the two apart.
    #[test]
    fn leaves_a_bare_annotation_identifier_alone() {
        assert!(mask_annotation_macros("int UMETA = 1;\n").is_none());
        assert!(mask_annotation_macros("return UMETA;\n").is_none());
        // In a comment or a string it is not code at all, and the scanner skips both.
        assert!(mask_annotation_macros("// UMETA(DisplayName = \"x\")\n").is_none());
        assert!(mask_annotation_macros("const char* s = \"UMETA(x)\";\n").is_none());
    }

    #[test]
    fn leaves_files_without_annotations_untouched() {
        assert!(mask_annotation_macros("class A { int x; };\n").is_none());
        assert!(mask_annotation_macros("").is_none());
        // The other UE macros are deliberately not masked — see `ANNOTATION_MACROS`.
        assert!(mask_annotation_macros("UCLASS(Blueprintable)\nclass A {};\n").is_none());
        assert!(mask_annotation_macros("UPROPERTY(EditAnywhere)\nint32 H;\n").is_none());
    }

    /// The reason this mask exists, asserted at the level that motivated it.
    ///
    /// A `UENUM` whose enumerators carry `UMETA` collapsed **the whole rest of the header**
    /// into one root `ERROR` node. `walk_top_level` reads the root's children, so the outline
    /// of a 936-line UE header came back empty and nothing declared below the enum was
    /// findable by name.
    ///
    /// The fixture is a real UE header shape rather than a reduced one, and that is
    /// deliberate: a simplified version — bare `struct FData { int Count; };` after the enum —
    /// recovers on its own and does *not* reproduce the cascade. The unmasked half is asserted
    /// first so this cannot silently stop testing anything if a grammar update improves
    /// recovery.
    ///
    /// The masked half goes through `parse_masked`, the production path, so it covers the
    /// composition with `mask_export_macros` (`PROBEMODULE_API`) rather than this mask alone.
    #[test]
    fn a_uenum_with_umeta_no_longer_swallows_the_declarations_after_it() {
        let src = "\
UENUM(BlueprintType)
enum class EProbeMode : uint8
{
    ModeAlpha UMETA(DisplayName = \"Alpha\"),
    ModeBeta  UMETA(DisplayName = \"Beta\")
};

USTRUCT(BlueprintType)
struct PROBEMODULE_API FProbeData
{
    GENERATED_BODY()
    UPROPERTY(EditAnywhere)
    int32 ProbeCount = 0;
};

UCLASS(Blueprintable)
class PROBEMODULE_API AProbeActor : public AActor
{
    GENERATED_BODY()
public:
    AProbeActor();
};
";
        let lang = crate::types::Lang::Cpp;
        let ts = crate::lang::outline::outline_language(lang).expect("grammar");
        let root_kinds = |tree: &tree_sitter::Tree| -> Vec<String> {
            let mut c = tree.root_node().walk();
            tree.root_node()
                .children(&mut c)
                .map(|n| n.kind().to_string())
                .collect()
        };

        // The "before" is the *export-masked* text, not the raw file, because that is what
        // production fed the parser before this mask existed — and the distinction matters:
        // the raw file recovers a `class_specifier`, and blanking `PROBEMODULE_API` is what
        // tips recovery over into swallowing everything. Reproducing the bug needs both.
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&ts).expect("grammar loads");
        let exports_only = mask_export_macros(src).expect("the API macros mask");
        let before = root_kinds(&parser.parse(&exports_only, None).expect("parse succeeds"));
        assert_eq!(
            before,
            vec!["ERROR"],
            "fixture no longer reproduces the cascade it guards"
        );

        // Masked: every declaration is back at the top level, where `walk_top_level` sees it.
        let fixed = root_kinds(&crate::lang::parse_masked(src, Some(lang), &ts).expect("parses"));
        for want in ["enum_specifier", "struct_specifier", "class_specifier"] {
            assert!(
                fixed.contains(&want.to_string()),
                "missing {want} in {fixed:?}"
            );
        }
    }

    /// `UE_DEPRECATED` is masked wherever it appears, arguments and all — the same treatment as
    /// `UMETA`, for the member-position misparse it causes (#130).
    #[test]
    fn masks_a_ue_deprecated_invocation() {
        blanks_only(
            "    ",
            "UE_DEPRECATED(5.2, \"No longer used\")",
            "\n    void Old();\n",
        );
        // The message's own parens are inside a string literal, so balanced-paren matching does
        // not end the span early on them.
        blanks_only(
            "",
            "UE_DEPRECATED(5.4, \"use Bar() instead\")",
            "\nint x;\n",
        );
    }

    /// The list is curated by the misparse trigger, not by the name containing `DEPRECATED`. Two
    /// kinds must stay untouched: a declaration-*generating* macro (Slate's, which declares the
    /// member `Width`), and any other `DEPRECATED`-named macro. Exact-identifier matching is what
    /// guarantees it — a substring hit in the prefilter must not widen the mask.
    #[test]
    fn leaves_unlisted_and_declaration_generating_deprecated_macros_alone() {
        // Declares a real member — blanking it would delete `Width`.
        assert!(
            mask_annotation_macros("SLATE_ARGUMENT_DEPRECATED(int, Width, 5.0, \"x\")\n").is_none()
        );
        // A longer identifier that shares the `UE_DEPRECATED` prefix must not be caught by the
        // prefilter's substring hit; nor a single-message attribute that never misparses.
        assert!(mask_annotation_macros("UE_DEPRECATED_FORGAME(5.0, \"x\")\nvoid f();\n").is_none());
        assert!(mask_annotation_macros("BOOST_DEPRECATED(\"x\")\nvoid f();\n").is_none());
    }

    /// The failure `UE_DEPRECATED` masking exists to fix (#130): a deprecation attribute before a
    /// class member, inside a namespace, absorbs the `struct` keyword into an `ERROR`, so the
    /// type's own `struct_specifier` disappears and only the surviving constructor is findable.
    /// Masking restores the struct.
    ///
    /// The namespace wrapper is load-bearing — the same struct at file scope recovers a
    /// `struct_specifier` on its own — so the unmasked half is asserted first to keep the fixture
    /// honest if a grammar update changes recovery. Goes through `parse_masked`, the production
    /// path.
    #[test]
    fn a_member_ue_deprecated_no_longer_hides_its_enclosing_struct() {
        let src = "\
namespace NP
{
struct FDescriptor
{
    UE_DEPRECATED(5.2, \"No longer used\")
    FDescriptor(const int* P) : Name(P) {}
    const int* Name;
};
}
";
        let lang = crate::types::Lang::Cpp;
        let ts = crate::lang::outline::outline_language(lang).expect("grammar");
        let has_named_struct = |tree: &tree_sitter::Tree| -> bool {
            let mut stack = vec![tree.root_node()];
            while let Some(n) = stack.pop() {
                if n.kind() == "struct_specifier"
                    && n.child_by_field_name("name")
                        .is_some_and(|x| &src[x.byte_range()] == "FDescriptor")
                {
                    return true;
                }
                let mut c = n.walk();
                stack.extend(n.children(&mut c));
            }
            false
        };

        // Guard the fixture: unmasked, the struct_specifier is gone — the misparse this fixes.
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&ts).expect("grammar loads");
        let raw = parser.parse(src, None).expect("parse");
        assert!(
            !has_named_struct(&raw),
            "fixture no longer reproduces the misparse it guards"
        );

        // Masked (production path): the struct is back.
        let fixed = crate::lang::parse_masked(src, Some(lang), &ts).expect("parses");
        assert!(
            has_named_struct(&fixed),
            "UE_DEPRECATED masking did not restore the struct_specifier"
        );
    }
}
