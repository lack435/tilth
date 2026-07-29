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
//! name, subject to two gates that both have to pass.
//!
//! **A brace before the semicolon.** `class Foo bar;` is an ordinary variable
//! declaration of type `Foo` — the same token shape as a macro-prefixed head — and
//! masking it would delete the type and leave `class bar;`. Only a definition has a
//! body.
//!
//! **Every masked identifier is macro-shaped**, meaning upper-case, digits and
//! underscores only. This one is a naming convention rather than a structural fact,
//! and it is here because structure alone is not sufficient:
//! `struct FVector ALIGN16 P{0,0,0};` declares a brace-initialised *variable* and has
//! a body, a type name, an attribute macro and a declarator — masking everything but
//! the last name would rewrite it into a definition of `struct P`. Since `FVector` is
//! not macro-shaped, the whole head is left alone. Every dllexport spelling in the
//! wild is upper-case (`MYLIB_API`, `Q_DECL_EXPORT`, `DLLEXPORT`), and a C++ *type*
//! named in upper-case is rare enough to be worth the trade — the failure mode
//! without the gate is corrupting a declaration, which is far worse than declining to
//! mask one.
//!
//! Masking is refused for the whole head unless every candidate passes, rather than
//! per identifier: a head where only some names look like macros is one this scanner
//! has misread, and guessing which half is right is how a declaration gets corrupted.

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
            // The last identifier is the type's own name; the ones before it are the
            // candidates. All of them must look like macros or none are masked.
            let macros = idents.len().saturating_sub(1);
            let candidates = &idents[..macros];
            if !candidates.is_empty()
                && candidates
                    .iter()
                    .all(|&(s, e)| is_macro_shaped(&bytes[s..e]))
            {
                spans.extend_from_slice(candidates);
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
    fn read_head(&mut self) -> Option<Vec<(usize, usize)>> {
        let mut idents: Vec<(usize, usize)> = Vec::new();
        loop {
            if self.pos >= self.bytes.len() {
                return None;
            }
            if self.skip_trivia() {
                continue;
            }
            let b = self.bytes[self.pos];
            match b {
                // The body: this head defines a type.
                b'{' => return Some(idents),
                // A declaration, not a definition — `class Foo bar;`, `class Fwd;`.
                // Also `,`, which means a declarator list (`class Foo a, b;`).
                b';' | b',' | b')' | b'=' => return None,
                // A base-class clause. Nothing after it can be the type's own name, so
                // stop collecting, but keep scanning for the brace that proves this is
                // a definition.
                b':' => {
                    return self.skip_to_body().then_some(idents);
                }
                // `__declspec(dllexport)`, `alignas(16)` — a macro that takes
                // arguments. It is not an identifier we can blank (the parens would be
                // left behind), and tree-sitter reads it as an attribute; skip it.
                b'(' => {
                    self.skip_balanced(b'(', b')')?;
                }
                // `[[nodiscard]]` and friends parse natively; step over them.
                b'[' => {
                    self.skip_balanced(b'[', b']')?;
                }
                _ if is_ident_start(b) => {
                    let start = self.pos;
                    let end = self.ident_end(start);
                    let word = &self.bytes[start..end];
                    self.pos = end;
                    if HEAD_TERMINATORS.contains(&word) {
                        return self.skip_to_body().then_some(idents);
                    }
                    // `enum class API Mode` — step over the second keyword rather than
                    // counting it as a name.
                    if TYPE_KEYWORDS.contains(&word) {
                        continue;
                    }
                    idents.push((start, end));
                }
                // Anything else in a type head (`<`, `*`, `&`, …) means this is not the
                // simple shape an export macro produces. Give up rather than guess.
                _ => return None,
            }
        }
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
                b'"' | b'\'' => self.skip_literal(b),
                // A preprocessor directive can hold anything, including a bare `class`
                // in a macro definition. Skip the whole logical line.
                b'#' => {
                    self.pos += 1;
                    while self.pos < self.bytes.len() {
                        match self.bytes[self.pos] {
                            b'\\' => self.pos += 2,
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

    #[track_caller]
    fn masks_to(src: &str, expected: &str) {
        let masked = mask_export_macros(src).unwrap_or_else(|| panic!("nothing masked in {src:?}"));
        assert_positions_preserved(src, &masked);
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
