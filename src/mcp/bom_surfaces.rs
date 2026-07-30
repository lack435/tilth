//! One table naming every MCP output surface and which side of the BOM rule it sits on (#64).
//!
//! Five issues have each fixed the sites that were visible at the time — #35 (import detection),
//! #41 (six parses), #43 (manifest reads), #42 (outline funnel, heading resolver, strip pass) and
//! #51 (eleven sites across search, grok, callers, diff). Within #51 alone the count went 4 → 6 → 7
//! → 9 across three passes plus a review, and the review found a leak in `diff/` that none of the
//! sweeps had touched. Every round the fix was correct and the *enumeration* was incomplete. That
//! is a missing invariant, not a reviewer-attention problem, and this module is the invariant.
//!
//! **There is no single funnel and there cannot be one, because the correct behaviour differs by
//! surface.** Surfaces that emit `{line}:{hash}|` anchors must *keep* the BOM: `edit::apply_batch`
//! verifies a hash against the raw bytes on disk, so stripping desynchronises hashing from the file
//! and every anchored edit fails. Every other surface must *strip*, because a BOM rendered into
//! match text or an outline entry is a visible glyph the reader did not ask for and a prefix that
//! `starts_with` comparisons silently fail on. So the rule is real but per-surface, and nothing
//! stopped a *new* surface from picking neither side.
//!
//! **The exhaustiveness check is the point, not the assertions.** `SURFACES` is compared against
//! the schema the server actually advertises — the tool names in `tool_definitions`, plus the
//! `mode` and `kind` enums inside them — so adding a tool or a read mode fails
//! `every_advertised_surface_is_named` until someone writes down which side it is on. Assertions
//! that a surface behaves correctly are worth having; a test that notices a surface *nobody
//! classified* is what the five previous rounds each lacked.

use super::{dispatch_tool, tool_definitions, Services};
use serde_json::{json, Value};
use std::fmt::Write as _;
use std::path::Path;

const BOM: &str = "\u{FEFF}";

/// Which side of the rule a surface sits on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bom {
    /// Output must be **byte-identical** to the same tree written without the BOM.
    ///
    /// Stronger than "contains no U+FEFF", deliberately. A BOM that survives into a line-length
    /// calculation, a column, a token estimate or a match count leaves no U+FEFF anywhere in the
    /// output and is still a bug — #51's ranking leak was exactly that shape.
    Strips,
    /// Output must carry the BOM, because it emits hashline anchors that `edit::apply_batch`
    /// verifies against the raw file.
    Keeps,
    /// Renders no file-derived text, so neither column applies. Carries its reason so the third
    /// option cannot become a place to put a surface nobody wanted to think about.
    NoFileText(&'static str),
}

/// A surface: a label, the tool it dispatches to, the arguments that select it, and its side.
struct Surface {
    label: &'static str,
    tool: &'static str,
    /// Edit mode changes `read`'s output (hashlines), so it is part of the surface's identity.
    edit_mode: bool,
    /// For read surfaces, the `[marker]` the header must carry — `[full]`, `[outline]`,
    /// `[stripped]`, `[signature]`.
    ///
    /// **This is load-bearing, not decoration.** A read `mode` is a *request*; what decides the
    /// BOM rule is the view it resolves to, and two modes resolve conditionally. `auto` picks full
    /// content or an outline depending on the OGATE, and `stripped` falls back to a full read for
    /// any file type with no strip pass. Both pairs straddle the rule, so a row that does not pin
    /// its resolved view can silently drift onto the other side of the table and keep passing —
    /// which is the too-weak-fixture trap #51 hit three times.
    view: Option<&'static str>,
    expect: Bom,
}

/// Every surface, with the arguments built per fixture root by `args_for`.
///
/// **Two read modes appear twice, because they are not one surface each.** #64's proposed table had
/// a single `("read:auto", StripsBom)` row and a single `("read:stripped", StripsBom)`. Both
/// resolve conditionally, and each pair straddles the rule:
///
/// * `auto` gives full content or an outline. The OGATE in `read::smart_view` rejects the outline
///   unless it is meaningfully *smaller* than the file, so this is a compression question, not a
///   size one. Measured while writing this: 400 and then 6000 one-line functions — 53k tokens —
///   both resolved to **full**, because a signature list for one-line functions is nearly the file
///   itself. Only functions with bodies select the outline.
/// * `stripped` gives a strip pass or, for a file type that has none, a **full read**. Measured:
///   `.rs` strips the BOM and reports `[stripped]`; `.md` reports `[full]` and keeps it, single or
///   doubled. So `mode: "stripped"` on markdown is `read:full` wearing another name, and belongs in
///   the keep column.
///
/// Neither would have been visible from the mode name, and a table keyed on the requested mode
/// rather than the resolved view records the wrong answer for half of each pair. Every read row
/// therefore pins its `view` marker.
const SURFACES: &[Surface] = &[
    Surface {
        label: "read:auto -> full content",
        tool: "tilth_read",
        edit_mode: true,
        view: Some("[full]"),
        expect: Bom::Keeps,
    },
    Surface {
        label: "read:auto -> outline",
        tool: "tilth_read",
        edit_mode: true,
        view: Some("[outline]"),
        expect: Bom::Strips,
    },
    Surface {
        label: "read:full",
        tool: "tilth_read",
        edit_mode: true,
        view: Some("[full]"),
        expect: Bom::Keeps,
    },
    Surface {
        label: "read:full non-edit",
        tool: "tilth_read",
        edit_mode: false,
        view: Some("[full]"),
        expect: Bom::Keeps,
    },
    Surface {
        label: "read:signature",
        tool: "tilth_read",
        edit_mode: true,
        view: Some("[signature]"),
        expect: Bom::Keeps,
    },
    Surface {
        label: "read:stripped -> strip pass",
        tool: "tilth_read",
        edit_mode: true,
        view: Some("[stripped]"),
        expect: Bom::Strips,
    },
    // Markdown has no strip pass, so `mode: "stripped"` degrades to a full read and inherits the
    // keep column with it. Measured, not assumed — see the note on `SURFACES`.
    Surface {
        label: "read:stripped -> full fallback",
        tool: "tilth_read",
        edit_mode: true,
        view: Some("[full]"),
        expect: Bom::Keeps,
    },
    Surface {
        label: "read:section",
        tool: "tilth_read",
        edit_mode: true,
        view: Some("[section]"),
        expect: Bom::Keeps,
    },
    Surface {
        label: "read:sections",
        tool: "tilth_read",
        edit_mode: true,
        view: Some("[section]"),
        expect: Bom::Keeps,
    },
    Surface {
        label: "search:symbol",
        tool: "tilth_search",
        edit_mode: false,
        view: None,
        expect: Bom::Strips,
    },
    // Expansion and no-expansion are mutually exclusive render paths, and #51 learned this the
    // hard way: `fence_will_follow` suppresses the `-> [line]` preview whenever an expansion
    // reprints the line, so an expand-only fixture never exercises the preview at all.
    Surface {
        label: "search:symbol expand=0",
        tool: "tilth_search",
        edit_mode: false,
        view: None,
        expect: Bom::Strips,
    },
    Surface {
        label: "search:content",
        tool: "tilth_search",
        edit_mode: false,
        view: None,
        expect: Bom::Strips,
    },
    Surface {
        label: "search:regex",
        tool: "tilth_search",
        edit_mode: false,
        view: None,
        expect: Bom::Strips,
    },
    Surface {
        label: "search:callers",
        tool: "tilth_search",
        edit_mode: false,
        view: None,
        expect: Bom::Strips,
    },
    Surface {
        label: "grok",
        tool: "tilth_grok",
        edit_mode: false,
        view: None,
        expect: Bom::Strips,
    },
    Surface {
        label: "deps",
        tool: "tilth_deps",
        edit_mode: false,
        view: None,
        expect: Bom::Strips,
    },
    Surface {
        label: "files",
        tool: "tilth_files",
        edit_mode: false,
        view: None,
        expect: Bom::Strips,
    },
    Surface {
        label: "diff:patch",
        tool: "tilth_diff",
        edit_mode: false,
        view: None,
        expect: Bom::Strips,
    },
    Surface {
        label: "savings",
        tool: "tilth_savings",
        edit_mode: false,
        view: None,
        expect: Bom::NoFileText("a token-savings ratio over counters; no file text reaches it"),
    },
    Surface {
        label: "session",
        tool: "tilth_session",
        edit_mode: false,
        view: None,
        expect: Bom::NoFileText(
            "summary/reset of dedup and savings state; no file text reaches it",
        ),
    },
    Surface {
        label: "write",
        tool: "tilth_write",
        edit_mode: true,
        view: None,
        expect: Bom::NoFileText(
            "a per-file outcome report. Its BOM contract is the hashline round-trip, \
             asserted by `a_kept_bom_round_trips_through_a_hash_anchored_write`",
        ),
    },
];

/// Arguments for a surface, against a fixture rooted at `root`.
///
/// Keyed on the label so the table above stays readable; an unrecognised label is a hard failure
/// rather than a skip, because a silently-skipped surface is the thing this module exists to stop.
fn args_for(label: &str, root: &Path) -> Value {
    let p = |name: &str| root.join(name).to_string_lossy().to_string();
    let scope = root.to_string_lossy().to_string();
    match label {
        "read:auto -> full content" => json!({"path": p("lib.rs"), "mode": "auto"}),
        "read:auto -> outline" => json!({"path": p("big.rs"), "mode": "auto"}),
        "read:full" | "read:full non-edit" => json!({"path": p("lib.rs"), "mode": "full"}),
        "read:signature" => json!({"path": p("lib.rs"), "mode": "signature"}),
        "read:stripped -> strip pass" => json!({"path": p("lib.rs"), "mode": "stripped"}),
        "read:stripped -> full fallback" => json!({"path": p("notes.md"), "mode": "stripped"}),
        "read:section" => json!({"path": p("lib.rs"), "section": "1-3"}),
        "read:sections" => json!({"path": p("lib.rs"), "sections": ["1-2", "5-6"]}),
        "search:symbol" => json!({"query": "bom_target", "kind": "symbol", "scope": scope}),
        "search:symbol expand=0" => {
            json!({"query": "bom_target", "kind": "symbol", "scope": scope, "expand": 0})
        }
        "search:content" => json!({"query": "bom_target", "kind": "content", "scope": scope}),
        "search:regex" => json!({"query": "bom_tar[g]et", "kind": "regex", "scope": scope}),
        "search:callers" => json!({"query": "bom_target", "kind": "callers", "scope": scope}),
        "grok" => json!({"target": "bom_target", "scope": scope}),
        "deps" => json!({"path": p("lib.rs")}),
        "files" => json!({"pattern": "*.rs", "scope": scope}),
        "diff:patch" => json!({"patch": p("change.patch")}),
        "savings" => json!({}),
        "session" => json!({"action": "summary"}),
        // Append to a file the other surfaces do not read, so the write cannot perturb them.
        "write" => {
            json!({"files": [{"path": p("scratch.rs"), "mode": "append", "content": "// x\n"}]})
        }
        other => panic!("no arguments defined for surface `{other}`"),
    }
}

/// Write the fixture. `bom` selects whether the BOM bytes are present, so the same tree can be
/// produced both ways and the two outputs compared.
///
/// Two requirements #64 records, both learned in #51 where three fixtures were initially too weak
/// and passed while a fix was reverted:
///
/// * **The BOM must land on a line the surface actually renders.** A definition on line 3 leaves
///   `Match.text` clean; a call on line 1 that is not inside a function leaves `caller_range` `None`
///   and skips the expansion read entirely. So `bom_target` is defined on line 1 of `lib.rs`, and
///   called from *inside a function* in `other.rs`, whose line 1 also carries a BOM.
/// * **Mutually exclusive render paths both need coverage.** Handled by the table, which carries
///   both `search:symbol` and `search:symbol expand=0`.
fn write_fixture(root: &Path, bom: bool) {
    let b = if bom { BOM } else { "" };

    // Definition on line 1, so search match text, grok's body, the outline funnel and every read
    // mode all have to render the BOM'd line.
    std::fs::write(
        root.join("lib.rs"),
        format!(
            "{b}pub fn bom_target() -> u32 {{\n    7\n}}\n\npub fn second() -> u32 {{\n    8\n}}\n"
        ),
    )
    .unwrap();

    // A caller *inside a function*, so `callers` has a `caller_range` and takes the expansion path.
    std::fs::write(
        root.join("other.rs"),
        format!(
            "{b}use crate::lib::bom_target;\n\npub fn use_it() -> u32 {{\n    bom_target()\n}}\n"
        ),
    )
    .unwrap();

    // Bodies, not one-liners: see the note on `SURFACES` for why the OGATE needs compression
    // rather than size to select the outline.
    let mut big = format!("{b}pub fn bom_target() -> u32 {{\n");
    for i in 0..30 {
        let _ = writeln!(big, "    let x{i} = {i};");
    }
    big.push_str("    0\n}\n");
    for i in 0..300 {
        let _ = writeln!(big, "pub fn filler{i}() -> u32 {{");
        for j in 0..30 {
            let _ = writeln!(big, "    let y{j} = {j};");
        }
        let _ = writeln!(big, "    {i}\n}}");
    }
    std::fs::write(root.join("big.rs"), &big).unwrap();

    // Markdown, for the `stripped`-has-no-strip-pass row. The BOM sits on the heading, which is the
    // line the read renders.
    std::fs::write(
        root.join("notes.md"),
        format!("{b}# bom_target\n\nProse about it.\n"),
    )
    .unwrap();

    std::fs::write(root.join("scratch.rs"), "// scratch\n").unwrap();
    std::fs::write(
        root.join("Cargo.toml"),
        "[package]\nname = \"p\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();

    // The patch's BOM'd line names `patched_only`, **not** `bom_target`, and that is deliberate.
    //
    // A unified diff's content lines each begin with a marker — space, `+`, `-` — so a BOM'd source
    // line inside a patch necessarily sits *after* that marker, which makes it a **mid-line** BOM.
    // #64's non-goals exclude those explicitly: `types::match_text` strips U+FEFF from the *start* of
    // a matched line and deliberately leaves it elsewhere. A first version used `bom_target` here, so
    // the search surfaces matched the patch file too and rendered
    // `-> [4]  <BOM>pub fn bom_target…` — failing as a leak when it is documented behaviour, and
    // failing on the wrong surface at that. Naming a symbol nothing else searches for keeps the
    // patch as diff's input without contaminating rows that scan the tree. Putting it in a
    // dot-directory does *not* work: the walker still visits it.
    std::fs::write(
        root.join("change.patch"),
        format!(
            "--- a/lib.rs\n+++ b/lib.rs\n@@ -1,3 +1,4 @@\n {b}pub fn patched_only() -> u32 {{\n+    let added = 1;\n     7\n }}\n"
        ),
    )
    .unwrap();
}

/// Run a surface and return its output with absolute paths replaced, so two fixture roots compare.
fn render(surface: &Surface, root: &Path) -> String {
    let services = Services::new(surface.edit_mode);
    let args = args_for(surface.label, root);
    let out = dispatch_tool(surface.tool, &args, &services)
        .unwrap_or_else(|e| panic!("surface `{}` failed: {e}", surface.label));
    // Both the plain and the verbatim (`\\?\`) spellings appear on Windows.
    let s = root.to_string_lossy().to_string();
    let out = out
        .replace(&format!("\\\\?\\{s}"), "<ROOT>")
        .replace(&s, "<ROOT>");
    normalise_token_estimate(&out)
}

/// Replace `~N tokens` with a placeholder, and nothing else.
///
/// The one legitimate difference between a BOM'd tree and its BOM-free twin. The header's estimate
/// describes **the file's bytes**, and a BOM is three of them, so `~19 tokens` against `~18` is the
/// header telling the truth rather than leaking — measured on `read:stripped`, whose rendered body is
/// byte-identical and BOM-free either way.
///
/// Deliberately narrow. The **line count** in the same header stays under the strong comparison,
/// because that is where a BOM genuinely does misreport: a file of only BOM bytes has
/// `"".lines().count() == 0` stripped against `1` unstripped, which #64 names as a live wrinkle. Path,
/// view marker and the `stripped N of M` line all stay exact too.
fn normalise_token_estimate(out: &str) -> String {
    let mut result = String::with_capacity(out.len());
    let mut rest = out;
    while let Some(i) = rest.find('~') {
        result.push_str(&rest[..i]);
        let after = &rest[i + 1..];
        let digits: String = after.chars().take_while(char::is_ascii_digit).collect();
        let tail = &after[digits.len()..];
        // `~3.3k tokens` also occurs, so accept a decimal-and-unit spelling as well as bare digits.
        let unit = ["k tokens", " tokens"]
            .into_iter()
            .find(|u| tail.starts_with(u))
            .or_else(|| {
                tail.starts_with('.')
                    .then(|| {
                        let frac: String =
                            tail[1..].chars().take_while(char::is_ascii_digit).collect();
                        ["k tokens", " tokens"]
                            .into_iter()
                            .find(|u| tail[1 + frac.len()..].starts_with(u))
                    })
                    .flatten()
            });
        match unit {
            Some(_) if !digits.is_empty() => {
                result.push_str("~N tokens");
                // Skip past the whole estimate, however it was spelled.
                let consumed = tail
                    .find("tokens")
                    .map_or(tail.len(), |p| p + "tokens".len());
                rest = &tail[consumed..];
            }
            _ => {
                result.push('~');
                rest = after;
            }
        }
    }
    result.push_str(rest);
    result
}

/// The guard: every surface is byte-identical without the BOM, or explicitly anchor-bearing.
#[test]
fn every_surface_strips_the_bom_or_is_declared_to_keep_it() {
    let bom_dir = tempfile::tempdir().unwrap();
    let plain_dir = tempfile::tempdir().unwrap();
    write_fixture(bom_dir.path(), true);
    write_fixture(plain_dir.path(), false);

    for surface in SURFACES {
        let with = render(surface, bom_dir.path());

        // The resolved view first: a row whose view drifted is testing a different surface than it
        // claims, and every assertion below would still pass while doing so.
        if let Some(view) = surface.view {
            let header = with.lines().next().unwrap_or_default();
            assert!(
                header.contains(view),
                "`{}` no longer resolves to {view} — it is now testing a different surface, and                  whichever side of the rule that one sits on, this row's answer is stale:
{header}",
                surface.label
            );
        }

        match surface.expect {
            Bom::Strips => {
                let without = render(surface, plain_dir.path());
                assert_eq!(
                    with, without,
                    "`{}` is declared to strip the BOM, but its output differs from the BOM-free \
                     spelling of the same tree",
                    surface.label
                );
                assert!(!with.contains(BOM), "`{}` renders a U+FEFF", surface.label);
            }
            Bom::Keeps => {
                assert!(
                    with.contains(BOM),
                    "`{}` is declared to keep the BOM for the hashline contract, but stripped it — \
                     `edit::apply_batch` verifies hashes against the raw file, so every anchored \
                     edit to line 1 of a BOM'd file now fails:\n{with}",
                    surface.label
                );
            }
            Bom::NoFileText(reason) => {
                assert!(
                    !with.contains(BOM),
                    "`{}` is declared to render no file text ({reason}), yet a BOM reached its \
                     output — the declaration is wrong:\n{with}",
                    surface.label
                );
            }
        }
    }
}

/// Every tool, read mode and search kind the server advertises must be named by the table.
///
/// This is the exhaustiveness half, and the reason the module exists. It reads the schema
/// `tool_definitions` actually sends over the wire rather than a hand-maintained count, so a new
/// tool, a new `mode` or a new `kind` fails here until someone decides which side of the rule it is
/// on. `edit_mode: true` because that is the superset — `tilth_write` only exists there.
#[test]
fn every_advertised_surface_is_named() {
    let defs = tool_definitions(true);
    let labels: Vec<&str> = SURFACES.iter().map(|s| s.label).collect();
    let named = |needle: &str| labels.iter().any(|l| l.contains(needle));

    let mut tools: Vec<String> = Vec::new();
    for def in &defs {
        let name = def["name"]
            .as_str()
            .expect("every tool definition has a name");
        tools.push(name.to_string());

        // `tilth_read`'s `mode` and `tilth_search`'s `kind` are the two enums that multiply a tool
        // into several surfaces. Pulled from the schema so a new variant cannot slip past.
        let props = &def["inputSchema"]["properties"];
        for enum_field in ["mode", "kind"] {
            if let Some(values) = props[enum_field]["enum"].as_array() {
                for v in values {
                    let variant = v.as_str().expect("enum variants are strings");
                    assert!(
                        named(variant),
                        "`{name}` advertises {enum_field} `{variant}`, which no row in SURFACES \
                         names. Add a row saying whether it strips the BOM or keeps it for the \
                         hashline contract — that decision is what five rounds of this bug each \
                         skipped."
                    );
                }
            }
        }
    }

    for tool in &tools {
        // Rows carry short labels (`grok`, not `tilth_grok`), so compare on the tool field.
        assert!(
            SURFACES.iter().any(|s| s.tool == tool),
            "tool `{tool}` is advertised but no row in SURFACES exercises it. Add one, with \
             `Bom::NoFileText` and a reason if it genuinely renders no file-derived text."
        );
    }

    // And the reverse, so a row for a tool that no longer exists is caught rather than passing
    // forever. Asserted against **dispatchability**, not advertisement, because the two sets are
    // not equal: `tilth_session` has an arm in `dispatch_tool` and no entry in
    // `tool_definitions` — the server answers a tool it never announces. Found by an earlier version
    // of this check, which compared against the advertised list and failed on that row. Worth an
    // issue of its own; not this module's business to decide, but very much its business not to
    // paper over.
    let services = Services::new(true);
    for s in SURFACES {
        let err = dispatch_tool(s.tool, &json!({}), &services).err();
        assert!(
            err.as_deref() != Some(&*format!("unknown tool: {}", s.tool)),
            "SURFACES names tool `{}`, which `dispatch_tool` no longer handles",
            s.tool
        );
    }
}

/// The keep column's actual contract: a hash from a BOM'd file's line 1 must still apply.
///
/// Asserting the BOM is present is necessary but not sufficient — what makes keeping it correct is
/// that `edit::apply_batch` hashes the raw bytes on disk, so a read that stripped it would produce
/// an anchor that cannot match. Driven through `tilth_write` rather than `apply_batch` directly, so
/// it exercises the path an agent takes.
#[test]
fn a_kept_bom_round_trips_through_a_hash_anchored_write() {
    let dir = tempfile::tempdir().unwrap();
    write_fixture(dir.path(), true);
    let path = dir.path().join("lib.rs").to_string_lossy().to_string();
    let services = Services::new(true);

    let read = dispatch_tool(
        "tilth_read",
        &json!({"path": path, "mode": "full"}),
        &services,
    )
    .expect("full read in edit mode");

    // Hashlines are `{line}:{hash}|`; take line 1's, which is the BOM'd line.
    let anchor = read
        .lines()
        .find_map(|l| {
            let (head, _) = l.split_once('|')?;
            let (line_no, hash) = head.trim().split_once(':')?;
            (line_no == "1").then(|| format!("1:{hash}"))
        })
        .unwrap_or_else(|| panic!("no hashline for line 1 in edit-mode full read:\n{read}"));

    let out = dispatch_tool(
        "tilth_write",
        &json!({"files": [{"path": path, "mode": "hash", "edits": [
            {"start": anchor, "content": "pub fn bom_target() -> u32 {"}
        ]}]}),
        &services,
    )
    .unwrap_or_else(|e| {
        panic!("hash-anchored write to line 1 of a BOM'd file failed, so the read stripped the BOM the hash is computed over: {e}")
    });
    assert!(
        !out.to_lowercase().contains("hash mismatch"),
        "hash anchor from a BOM'd file did not verify:\n{out}"
    );
}

/// Degenerate inputs, which is where #64 records the remaining known wrinkles.
///
/// A file that is *only* BOM bytes: `"".lines().count()` is 0 against 1 for the unstripped spelling,
/// so a stripping view and a keeping view disagree about the line count. And a doubled BOM, because
/// tree-sitter-md absorbs one on its own — a single-BOM fixture cannot detect a markdown regression
/// at all, which is why #51 needed the doubled spelling.
///
/// **Classified by resolved view, not by requested mode**, which is the same lesson `SURFACES`
/// learned. A first version of this test asserted "`mode: stripped` must not carry a BOM" and failed
/// on `doubled.md` — correctly, but for the wrong reason: markdown has no strip pass, so that read
/// resolves to `[full]` and is in the *keep* column. Keying on the mode name reported a leak that was
/// not one. The marker is the discriminator, so read it rather than assuming.
#[test]
fn degenerate_bom_inputs_do_not_panic_and_stay_classified() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    std::fs::write(root.join("only_bom.rs"), BOM).unwrap();
    std::fs::write(
        root.join("doubled.md"),
        format!("{BOM}{BOM}# bom_target\n\nProse.\n"),
    )
    .unwrap();

    let services = Services::new(true);
    for (file, mode) in [
        ("only_bom.rs", "auto"),
        ("only_bom.rs", "full"),
        ("only_bom.rs", "stripped"),
        ("doubled.md", "auto"),
        ("doubled.md", "stripped"),
    ] {
        let path = root.join(file).to_string_lossy().to_string();
        let out = dispatch_tool(
            "tilth_read",
            &json!({"path": path, "mode": mode}),
            &services,
        )
        .unwrap_or_else(|e| panic!("read {file} mode={mode}: {e}"));
        let header = out.lines().next().unwrap_or_default().to_string();

        // Only `[stripped]` is in the strip column. `[full]` keeps, whatever mode was asked for.
        if header.contains("[stripped]") {
            assert!(
                !out.contains(BOM),
                "a `[stripped]` view leaked a BOM from {file} (mode={mode}):\n{out}"
            );
        }
    }
}
