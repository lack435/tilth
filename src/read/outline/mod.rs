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
/// they take: `markdown` parses `buf` while everything else takes `content`. Stripping only
/// one leaves a leak that moves rather than closes.
///
/// This is an *outline* funnel, so nothing here feeds `tilth_write`'s hash verification —
/// see the note in `read::full_view` for why the full-content path is deliberately left
/// carrying the BOM.
pub fn generate(
    path: &Path,
    file_type: FileType,
    content: &str,
    buf: &[u8],
    capped: bool,
) -> String {
    let content = crate::lang::outline::strip_bom(content);
    let buf = strip_bom_bytes(buf);
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

/// The byte counterpart to `lang::outline::strip_bom`, for the one backend that parses raw
/// bytes rather than a `&str`.
///
/// Repeats are stripped for the same reason both `&str` helpers strip them: a tool that
/// prepends a BOM without checking for an existing one leaves two, and stopping after the
/// first leaves the second rendering exactly as the first did.
fn strip_bom_bytes(buf: &[u8]) -> &[u8] {
    const BOM: &[u8] = &[0xEF, 0xBB, 0xBF];
    let mut rest = buf;
    while let Some(stripped) = rest.strip_prefix(BOM) {
        rest = stripped;
    }
    rest
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

    fn with_bom(body: &str) -> Vec<u8> {
        let mut bytes = UTF8_BOM.to_vec();
        bytes.extend_from_slice(body.as_bytes());
        bytes
    }

    /// A BOM must not reach the rendered outline, whichever backend produced it.
    ///
    /// The CSV row is the acceptance case in #42 and the one most often hit in practice —
    /// Excel's "CSV UTF-8" export writes a BOM — because `tabular` takes line 0 verbatim for
    /// its `columns:` header.
    ///
    /// Markdown is asserted alongside it because it is the one backend that parses `buf`
    /// rather than `content`, so stripping only the `&str` would have moved the leak instead
    /// of closing it. Doubled BOMs are covered for the reason both `&str` helpers already
    /// strip repeats: a tool that prepends one without checking leaves two.
    #[test]
    fn a_bom_does_not_reach_the_rendered_outline() {
        let doubled = [UTF8_BOM, UTF8_BOM].concat();

        // (path, file type, body, the text that must open the outline)
        let cases: &[(&str, FileType, &str, &str)] = &[
            (
                "data.csv",
                FileType::Tabular,
                "name,age\nalice,30\nbob,25\n",
                "columns: name,age",
            ),
            (
                "readme.md",
                FileType::Markdown,
                "# Title\n\nbody text\n",
                "Title",
            ),
        ];

        for (name, file_type, body, needle) in cases {
            let path = Path::new(name);
            let plain = generate(path, *file_type, body, body.as_bytes(), false);

            for prefix in [UTF8_BOM.to_vec(), doubled.clone()] {
                let mut bytes = prefix;
                bytes.extend_from_slice(body.as_bytes());
                let content = String::from_utf8(bytes.clone()).unwrap();
                let out = generate(path, *file_type, &content, &bytes, false);

                assert!(
                    !out.contains('\u{feff}'),
                    "{name}: a BOM reached the outline: {out:?}"
                );
                assert!(
                    out.contains(needle),
                    "{name}: expected {needle:?} in the outline, got {out:?}"
                );
                assert_eq!(out, plain, "{name}: a BOM changed the outline");
            }
        }
    }

    /// The fixture has to be able to fail, or the assertions above prove nothing: this pins
    /// that a BOM'd body really does differ from a clean one before `generate` sees it.
    #[test]
    fn the_bom_fixture_is_not_vacuous() {
        let body = "name,age\nalice,30\n";
        assert_ne!(
            with_bom(body),
            body.as_bytes(),
            "the fixture must actually carry a BOM"
        );
        assert!(
            String::from_utf8(with_bom(body))
                .unwrap()
                .contains('\u{feff}'),
            "the BOM must survive the round trip into a String"
        );
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
