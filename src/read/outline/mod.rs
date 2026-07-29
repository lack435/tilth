pub mod code;
pub mod fallback;
pub mod markdown;
pub mod structured;
pub mod tabular;
pub mod test_file;

use std::path::Path;

use crate::types::FileType;

const OUTLINE_CAP: usize = 100; // max outline lines for huge files

/// Generate a smart view based on file type.
///
/// A leading UTF-8 BOM is removed here, once, for every backend. `structured` already did
/// this one level down and that fixed the JSON parse failure *and* removed the same stray
/// character from YAML and the key-value passthrough for free (#41); this is the same move
/// at the funnel, which is what stops the next backend having to remember. Without it a
/// BOM'd CSV outlined as `columns: ﻿name,age` — Excel's "CSV UTF-8" export writes a BOM, so
/// that is the most frequently hit of the three leaks in #42 even though it is the mildest.
///
/// Both spellings of the content are stripped, because the backends do not agree on which
/// they take: `markdown` parses `buf` while everything else takes `content`.
///
/// The two buy different things, and it is worth being precise because the obvious guess is
/// wrong. Stripping `content` fixes the stray glyph, in the CSV header and in a code
/// outline's first entry. Stripping `buf` fixes nothing for a *single* BOM — tree-sitter-md
/// skips one by itself, so a BOM'd markdown file already outlined correctly — but a
/// **doubled** BOM makes it parse the *first* heading as a paragraph, so that heading drops
/// out of the outline. Measured on a two-heading file: 0 and 1 BOM outline both headings,
/// 2 BOMs outline only the second. A single-heading file therefore outlines as `""` under
/// two BOMs, which is where the loss is total — but the mechanism is "first heading lost",
/// not "whole outline emptied". Either way the `buf` strip is not cosmetic; it recovers a
/// heading the parser would otherwise swallow.
///
/// This is an *outline* funnel, so nothing here feeds `tilth_write`'s hash verification —
/// see the note in `read::full_view` for why the full-content path is deliberately left
/// carrying the BOM. But it does have to agree with `resolve_heading` and
/// `suggest_headings`, which parse the same bytes for the section reader: while only this
/// side stripped, the outline advertised a doubled-BOM heading that the resolver denied.
/// The search-side markdown parsers (`search::symbol`, `search::mod`) still parse unstripped
/// and so still disagree about a doubled-BOM file's first heading — tracked in #51.
pub fn generate(
    path: &Path,
    file_type: FileType,
    content: &str,
    buf: &[u8],
    capped: bool,
) -> String {
    let content = crate::lang::outline::strip_bom(content);
    let buf = crate::lang::outline::strip_bom_bytes(buf);
    let max_lines = if capped { OUTLINE_CAP } else { usize::MAX };

    // Test files get special treatment regardless of language
    if crate::types::is_test_file(path) {
        if let FileType::Code(lang) = file_type {
            if let Some(outline) = test_file::outline(content, lang, max_lines) {
                return with_omission_note(outline, max_lines);
            }
        }
    }

    let outline = match file_type {
        FileType::Code(lang) => code::outline(content, lang, max_lines),
        FileType::Markdown => markdown::outline(buf, max_lines),
        FileType::StructuredData => structured::outline(path, content, max_lines),
        FileType::Tabular => tabular::outline(content, max_lines),
        FileType::Log => fallback::log_view(content),
        FileType::Other => fallback::head_tail(content),
    };
    with_omission_note(outline, max_lines)
}

/// Append a note when the outline likely hit `max_lines` and more symbols
/// exist below. Without this note, agents read the outline as exhaustive
/// and miss symbols below the cap.
///
/// Note: `max_lines` is an entry cap inside `format_entries` (one
/// `out.push(...)` per entry, joined with `\n`). For the code-outline path
/// `outline.lines().count() == entry_count` exactly. For other backends
/// (markdown, structured, tabular) the same identity holds because each
/// pushes single-line entries. So the heuristic compares like-for-like.
/// We avoid claiming a specific count in the user-facing message — we
/// only state that more symbols exist, which is the actionable signal.
fn with_omission_note(outline: String, max_lines: usize) -> String {
    if max_lines == usize::MAX {
        return outline;
    }
    if outline.lines().count() < max_lines {
        return outline;
    }
    format!(
        "{outline}\n\n> outline truncated — more symbols exist below the cap. \
         Use section=\"<start>-<end>\" with the line numbers shown in [...] \
         brackets above, or tilth_search \"<name>\" for a specific symbol."
    )
}

#[cfg(test)]
mod tests {
    use super::{generate, with_omission_note};
    use crate::types::FileType;
    use std::path::Path;

    /// A UTF-8 BOM, written as bytes. Per the convention #35 and #41 established: a `&str`
    /// literal cannot express this, which is how the class of bug kept coming back.
    const UTF8_BOM: &[u8] = &[0xEF, 0xBB, 0xBF];

    /// `body` behind `n` BOMs, as bytes.
    fn with_boms(n: usize, body: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        for _ in 0..n {
            bytes.extend_from_slice(UTF8_BOM);
        }
        bytes.extend_from_slice(body.as_bytes());
        bytes
    }

    /// A BOM must not change the outline, whichever backend produced it.
    ///
    /// The CSV row is the acceptance case in #42 and the one most often hit in practice —
    /// Excel's "CSV UTF-8" export writes a BOM — because `tabular` takes line 0 verbatim for
    /// its `columns:` header. That one is the cosmetic fix: the BOM rendered as a stray
    /// glyph in front of the first column name.
    ///
    /// Markdown is here for a different and stronger reason. It parses `buf`, and
    /// tree-sitter-md already skips a *single* leading BOM — so one BOM never leaked. Two
    /// make it parse the **first** heading as a paragraph, dropping it. The fixture carries
    /// two headings on purpose: a single-heading file cannot tell "first heading lost" from
    /// "whole outline emptied", and asserting the wrong one of those is how the claim first
    /// went in overstated. Removing the `buf` strip fails this on `n == 2` because `Alpha`
    /// goes missing while `Beta` survives — not because the outline is empty.
    #[test]
    fn a_bom_does_not_reach_the_rendered_outline() {
        // (path, file type, body, substrings that must appear in the outline)
        let cases: &[(&str, FileType, &str, &[&str])] = &[
            (
                "data.csv",
                FileType::Tabular,
                "name,age\nalice,30\nbob,25\n",
                &["columns: name,age"],
            ),
            (
                "readme.md",
                FileType::Markdown,
                "# Alpha\n\nbody text\n\n# Beta\n\nmore text\n",
                &["Alpha", "Beta"],
            ),
        ];

        for (name, file_type, body, needles) in cases {
            let path = Path::new(name);
            let plain = generate(path, *file_type, body, body.as_bytes(), false);

            for n in 1..=2 {
                let bytes = with_boms(n, body);
                let content = String::from_utf8(bytes.clone()).unwrap();
                let out = generate(path, *file_type, &content, &bytes, false);

                assert!(
                    !out.contains('\u{feff}'),
                    "{name}: {n} BOM(s) reached the outline: {out:?}"
                );
                for needle in *needles {
                    assert!(
                        out.contains(needle),
                        "{name}: with {n} BOM(s), expected {needle:?} in the outline, got {out:?}"
                    );
                }
                assert_eq!(out, plain, "{name}: {n} BOM(s) changed the outline");
            }
        }
    }

    #[test]
    fn note_appended_when_at_cap() {
        let outline = (0..100)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let result = with_omission_note(outline, 100);
        assert!(result.contains("outline truncated"));
    }

    #[test]
    fn no_note_when_under_cap() {
        let outline = (0..50)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let result = with_omission_note(outline.clone(), 100);
        assert_eq!(result, outline);
    }

    #[test]
    fn no_note_when_uncapped() {
        let outline = (0..200)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let result = with_omission_note(outline.clone(), usize::MAX);
        assert_eq!(result, outline);
    }

    /// Integration test: drive the full `generate()` pipeline with a
    /// real Rust source containing more than OUTLINE_CAP top-level
    /// functions. Verifies the cap actually fires and that
    /// `with_omission_note` is wired into the pipeline correctly —
    /// not just exercised in isolation.
    #[test]
    fn integration_note_on_capped_code_file() {
        let src: String = (0..150)
            .map(|i| format!("pub fn func_{i}() {{}}\n"))
            .collect();
        let path = std::path::Path::new("fake.rs");
        let file_type = crate::types::FileType::Code(crate::types::Lang::Rust);
        let result = super::generate(path, file_type, &src, src.as_bytes(), true);
        assert!(
            result.contains("outline truncated"),
            "expected truncation note for 150 funcs over OUTLINE_CAP=100, got:\n{result}"
        );
    }

    /// Integration test: a small file (5 functions) must NOT produce
    /// the truncation note even when `capped=true` is passed, because
    /// the actual entry count is well below the cap.
    #[test]
    fn integration_no_note_on_small_code_file() {
        let src: String = (0..5)
            .map(|i| format!("pub fn func_{i}() {{}}\n"))
            .collect();
        let path = std::path::Path::new("fake.rs");
        let file_type = crate::types::FileType::Code(crate::types::Lang::Rust);
        let result = super::generate(path, file_type, &src, src.as_bytes(), true);
        assert!(
            !result.contains("outline truncated"),
            "spurious truncation note for 5 funcs (under OUTLINE_CAP=100):\n{result}"
        );
    }
}
