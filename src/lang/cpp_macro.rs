//! Blank out export macros in C/C++ type heads before the grammar sees them.
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
    // Byte-wise, then revalidated: every span is an ASCII identifier and every
    // replacement an ASCII space, so this cannot split a char boundary — but an
    // `unsafe` block to save one linear validation of a source file is a poor trade.
    // ASCII bytes never occur inside a multi-byte UTF-8 sequence, so the scanner
    // cannot have picked a span that starts mid-character either.
    let mut out = content.as_bytes().to_vec();
    for (start, end) in spans {
        out[start..end].fill(b' ');
    }
    String::from_utf8(out).ok()
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
    use super::mask_export_macros;

    /// What masking must never do: change a byte offset. Every downstream position —
    /// outline ranges, search line numbers, edit anchors — is read straight off the
    /// tree built from the masked text and used against the original.
    fn assert_positions_preserved(src: &str, masked: &str) {
        assert_eq!(
            src.len(),
            masked.len(),
            "masking changed the length:\n{src:?}\n{masked:?}"
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
}
