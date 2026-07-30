//! One table naming every MCP output surface and which side of the BOM rule it sits on (#64).
//!
//! Five issues have each fixed the sites visible at the time — #35 (import detection), #41 (six
//! parses), #43 (manifest reads), #42 (outline funnel, heading resolver, strip pass) and #51 (eleven
//! sites across search, grok, callers, diff). Within #51 alone the count went 4 → 6 → 7 → 9 across
//! three passes plus a review, and the review found a leak in `diff/` none of the sweeps had touched.
//! Every round the fix was correct and the *enumeration* was incomplete. That is a missing
//! invariant, and this module is the invariant.
//!
//! **There is no single funnel and there cannot be one, because the correct behaviour differs by
//! surface.** Surfaces emitting `{line}:{hash}|` anchors must *keep* the BOM: `edit::apply_batch`
//! verifies a hash against the raw bytes on disk, so stripping desynchronises hashing from the file
//! and every anchored edit fails. Some surfaces pass a file through verbatim, which keeps it for a
//! different reason. Everything else must *strip*, because a BOM in match text or an outline entry is
//! a glyph the reader did not ask for and a prefix that `starts_with` silently fails on.
//!
//! # What "exhaustive" has to mean here
//!
//! A first version of this module derived the surface list from `tool_definitions` alone and claimed
//! that as the guard. Review defeated it five ways, each verified by patching a fake surface in: a
//! tool added to `dispatch_tool` but never advertised; a variant whose name is a *substring* of an
//! existing row's label; an enum field other than `mode`/`kind`; `tilth_write`'s mode enum, nested a
//! level deeper than the walk looked; and every sub-mode expressed as something other than an enum,
//! including five of `tilth_diff`'s six sources. `overview` was absent altogether and, being derived
//! from the tool list, structurally unreachable — while being the largest block of file-derived text
//! the server emits.
//!
//! So exhaustiveness rests on **four independent anchors**, and a surface has to escape all four:
//!
//! 1. [`ANCHOR: dispatchable`] every name in `mcp::DISPATCHABLE_TOOLS` is exercised by a row.
//!    `dispatch_tool` gates on that const, so an unreachable tool cannot hide there.
//! 2. [`ANCHOR: advertised`] every tool in `tool_definitions` is dispatchable and named.
//! 3. [`ANCHOR: enums`] every `enum` in every schema, **at any depth**, has each variant named by a
//!    row — matched exactly, never by substring.
//! 4. [`ANCHOR: params`] every property name in every schema, at any depth, is either covered by a
//!    row or listed in [`PARAMS_WITHOUT_THEIR_OWN_SURFACE`] with a reason. A new parameter fails
//!    until someone decides whether it opens a new output shape.
//!
//! Plus `overview`, named explicitly because it is not a tool at all.
//!
//! # Every row must be able to fail
//!
//! The other half of the review's verdict: four of twenty rows could not fail under *any* BOM change
//! — a patch fixture the diff parser never accepted, a `deps` fixture with no dependents, a `files`
//! row whose only BOM-sensitive datum the normaliser erased, and a `write` row refused by the scope
//! guard before it did anything. A row that cannot fail is worse than a missing row, because it reads
//! as coverage. [`SITES_CAUGHT`] records which production strip site each row actually protects,
//! established by mutation rather than by reading, and names what is still uncovered.

use super::{dispatch_tool, tool_definitions, Services, DISPATCHABLE_TOOLS};
use serde_json::{json, Value};
use std::fmt::Write as _;
use std::path::Path;

const BOM: &str = "\u{FEFF}";

/// Which side of the rule a surface sits on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bom {
    /// Output must be **byte-identical** to the same tree written without the BOM.
    ///
    /// Stronger than "contains no U+FEFF", deliberately, and it is what found the seventh leak: a
    /// BOM surviving into a scope lookup, a rank, a column or a count leaves no U+FEFF anywhere and
    /// is still a wrong answer. #51's ranking leak was exactly that shape.
    Strips,
    /// Emits `{line}:{hash}|` anchors, so the BOM must survive for `edit::apply_batch` to verify
    /// them. Round-tripped by `every_anchor_bearing_surface_round_trips_through_a_write`.
    KeepsForAnchors,
    /// Passes the file through verbatim without anchors, so the BOM comes along.
    ///
    /// Split out from `KeepsForAnchors` because review caught the conflation: `read:full` outside
    /// edit mode and `mode: "stripped"` on markdown both emit **no hashlines at all**, so calling
    /// their behaviour a hash contract was a description masquerading as one. Nothing verifies these
    /// beyond "the BOM is still there", and that is the honest claim.
    PassesThrough,
    /// Renders no file-derived text, so neither column applies. Carries its reason, and the reason
    /// has to be true — the first version declared `tilth_write` this way and was wrong twice over.
    NoFileText(&'static str),
    /// Renders file *names* and byte-derived *sizes*, but no file content.
    ///
    /// A BOM is three bytes, so every size estimate legitimately differs — `files` renders nine of
    /// them and eight change. That is not a leak, and `Strips` is the wrong claim for it. The first
    /// version hid the difference by normalising `~N tokens` everywhere, which review showed made the
    /// row unfalsifiable; declaring it is the honest version. Asserted as "no U+FEFF anywhere",
    /// because a *name* or a size can never legitimately contain one.
    SizesMayDifferNoContent(&'static str),
}

/// A surface: a label, the tool it dispatches to, the arguments that select it, and its side.
struct Surface {
    label: &'static str,
    tool: &'static str,
    /// Edit mode changes `read`'s output (hashlines), so it is part of the surface's identity.
    edit_mode: bool,
    /// For read surfaces, the `[marker]` the header must carry.
    ///
    /// **Load-bearing.** A read `mode` is a *request*; the BOM rule is decided by the view it
    /// resolves to, and two modes resolve conditionally — `auto` picks full content or an outline via
    /// the OGATE, and `stripped` falls back to a full read for any file type with no strip pass. Both
    /// pairs straddle the rule, so a row that does not pin its resolved view can drift to the other
    /// side and keep passing.
    view: Option<&'static str>,
    expect: Bom,
}

/// Every surface. See the module docs for the four exhaustiveness anchors that keep this honest.
const SURFACES: &[Surface] = &[
    // ---- read: eight surfaces from four advertised modes, because two modes resolve conditionally.
    //
    // `auto` gives full content or an outline. The OGATE rejects an outline that is not meaningfully
    // *smaller* than the file, so it is a compression question, not a size one — 400 and then 6000
    // one-line functions (53k tokens) both resolved to full, because a signature list for one-line
    // functions is nearly the file itself. Only bodies select the outline.
    //
    // `stripped` gives a strip pass or, for a file type with none, a full read. Measured: `.rs`
    // reports `[stripped]` and strips; `.md` reports `[full]` and keeps, single BOM or doubled.
    Surface {
        label: "read:auto/full",
        tool: "tilth_read",
        edit_mode: true,
        view: Some("[full]"),
        expect: Bom::KeepsForAnchors,
    },
    Surface {
        label: "read:auto/outline",
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
        expect: Bom::KeepsForAnchors,
    },
    Surface {
        label: "read:full/no-edit-mode",
        tool: "tilth_read",
        edit_mode: false,
        view: Some("[full]"),
        expect: Bom::PassesThrough,
    },
    Surface {
        label: "read:signature",
        tool: "tilth_read",
        edit_mode: true,
        view: Some("[signature]"),
        expect: Bom::KeepsForAnchors,
    },
    Surface {
        label: "read:stripped",
        tool: "tilth_read",
        edit_mode: true,
        view: Some("[stripped]"),
        expect: Bom::Strips,
    },
    Surface {
        label: "read:stripped/no-strip-pass",
        tool: "tilth_read",
        edit_mode: true,
        view: Some("[full]"),
        expect: Bom::PassesThrough,
    },
    Surface {
        label: "read:section",
        tool: "tilth_read",
        edit_mode: true,
        view: Some("[section]"),
        expect: Bom::KeepsForAnchors,
    },
    Surface {
        label: "read:sections",
        tool: "tilth_read",
        edit_mode: true,
        view: Some("[section]"),
        expect: Bom::KeepsForAnchors,
    },
    // ---- search: one row per advertised `kind`, plus the expand/no-expand split.
    //
    // Expansion and no-expansion are mutually exclusive render paths, and #51 learned it the hard
    // way: `fence_will_follow` suppresses the `-> [line]` preview whenever an expansion reprints the
    // line, so an expand-only fixture never exercises the preview.
    Surface {
        label: "search:symbol",
        tool: "tilth_search",
        edit_mode: false,
        view: None,
        expect: Bom::Strips,
    },
    Surface {
        label: "search:symbol/expand=0",
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
        expect: Bom::SizesMayDifferNoContent(
            "a glob listing: file names plus a per-file byte-derived token estimate, no content",
        ),
    },
    // ---- diff: one row per source `resolve_source` can return. Five were unnamed in the first
    // version. `patch` and `log` render no file-derived content; the file pair does — but only in its
    // **file-detail** render, which reprints raw source lines. Its *overview* render (symbol names +
    // counts) cannot carry a BOM, because a name is a tree-sitter identifier node and the file's BOM
    // leads the line before it. So `diff:files` is scoped (see `args_for`) to reach the detail path,
    // which is the falsifiable one; a second review caught that the unscoped row was fake coverage.
    Surface {
        label: "diff:files",
        tool: "tilth_diff",
        edit_mode: false,
        view: None,
        expect: Bom::Strips,
    },
    Surface {
        label: "diff:patch",
        tool: "tilth_diff",
        edit_mode: false,
        view: None,
        // NOT `Strips`, though a patch clearly carries source lines. The Patch source reads no
        // working tree — `get_old_content`/`get_new_content` both return an empty string for it
        // (`diff/overlay.rs`), so `compute_overlay` parses **zero symbols** and the render is
        // metadata only (`## lib.rs (0 symbols)`); the patch's BOM'd content line never reaches the
        // output. A `Strips` byte-identical check would therefore be `X == X` — the fake-coverage
        // class this module exists to kill. And even in principle it could not strip a line-start
        // BOM: a unified-diff content line is prefixed by a `+`/`-`/space marker, so any BOM in it is
        // mid-line, which #64's non-goals exclude. Declared `NoFileText`, like `diff:log`.
        expect: Bom::NoFileText(
            "the Patch source reads no working tree (empty old/new content in overlay.rs), so the \
             structural diff renders 0 symbols and no file-derived line; a patch's BOM is mid-line \
             anyway, excluded by #64",
        ),
    },
    Surface {
        label: "diff:log",
        tool: "tilth_diff",
        edit_mode: false,
        view: None,
        expect: Bom::NoFileText(
            "renders git commit metadata; the working tree is never read. Named so the source is \
             classified, not because it can leak",
        ),
    },
    Surface {
        label: "diff:source=uncommitted",
        tool: "tilth_diff",
        edit_mode: false,
        view: None,
        expect: Bom::NoFileText(
            "diffs the process repo (cwd), never the fixture tree, so a fixture BOM cannot reach it; \
             the working tree holds no BOMs of its own",
        ),
    },
    Surface {
        label: "diff:source=staged",
        tool: "tilth_diff",
        edit_mode: false,
        view: None,
        expect: Bom::NoFileText("as `diff:source=uncommitted`"),
    },
    Surface {
        label: "diff:source=ref",
        tool: "tilth_diff",
        edit_mode: false,
        view: None,
        expect: Bom::NoFileText("as `diff:source=uncommitted`"),
    },
    // ---- write: two surfaces, because it straddles the rule.
    //
    // The first version declared this one `NoFileText` and was wrong twice: `diff: true` renders
    // `render_text_diff` over the file's own bytes, and append's hashline echo includes the
    // pre-existing last line when the file has no trailing newline. So the anchor echo keeps, and
    // the human-facing diff block strips.
    Surface {
        label: "write:append",
        tool: "tilth_write",
        edit_mode: true,
        view: None,
        expect: Bom::KeepsForAnchors,
    },
    Surface {
        label: "write:overwrite/diff",
        tool: "tilth_write",
        edit_mode: true,
        view: None,
        expect: Bom::Strips,
    },
    // `hash` mode's own response echoes the post-edit hashlines, so it is anchor-bearing too. The
    // contract itself — that an anchor over a BOM'd line verifies — is
    // `every_anchor_bearing_surface_round_trips_through_a_write`.
    Surface {
        label: "write:hash",
        tool: "tilth_write",
        edit_mode: true,
        view: None,
        expect: Bom::KeepsForAnchors,
    },
    // ---- the remaining tools, which genuinely render no file text.
    Surface {
        label: "savings",
        tool: "tilth_savings",
        edit_mode: false,
        view: None,
        expect: Bom::NoFileText("a token ratio over counters; no file text reaches it"),
    },
    Surface {
        label: "session",
        tool: "tilth_session",
        edit_mode: false,
        view: None,
        expect: Bom::NoFileText("summary/reset of dedup and savings counters; no file text"),
    },
];

/// Schema properties that do **not** open an output shape of their own, each with its reason.
///
/// The fourth exhaustiveness anchor. Every property name in every schema, at any depth, must be
/// covered by a row or listed here — so a new parameter fails the check until someone decides which
/// it is. This is where the first version leaked: `section`, `sections`, `expand`, the legacy `full`
/// boolean and `tilth_write`'s `diff` boolean were all reachable and none was named.
const PARAMS_WITHOUT_THEIR_OWN_SURFACE: &[(&str, &str)] = &[
    ("path", "selects the target, not the rendering"),
    ("paths", "batch of `path`; same rendering per file"),
    ("scope", "selects the tree, not the rendering"),
    ("root", "anchors relative paths; no rendering of its own"),
    ("query", "the needle; `kind` selects the rendering"),
    ("target", "grok's symbol; no rendering of its own"),
    ("pattern", "glob for `files`; no rendering of its own"),
    ("patterns", "batch of `pattern`"),
    ("glob", "filters which files a search visits"),
    ("context", "biases ranking; does not add a render path"),
    ("budget", "truncates an already-rendered surface"),
    (
        "blast",
        "adds callers of touched definitions, rendered as search results",
    ),
    (
        "search",
        "diff sub-filter; renders through the diff surfaces",
    ),
    (
        "content",
        "write input, never rendered back except via `diff`",
    ),
    ("overwrite", "write flag; does not change what is rendered"),
    (
        "edits",
        "write input; its rendering is `write:hashline-echo`",
    ),
    ("start", "an edit's anchor; input only"),
    ("end", "an edit's anchor; input only"),
    (
        "files",
        "write's per-file array; rendering is the two write rows",
    ),
    ("a", "diff file pair; rendering is `diff:files`"),
    ("b", "diff file pair; rendering is `diff:files`"),
];

/// Short spellings of an enum variant that name the same surface as their long form.
///
/// `tilth_write` advertises `hash`/`h`, `overwrite`/`w`, `append`/`a`. An alias is not a new output
/// shape, so it needs no row — but it does need saying so, otherwise anchor 3 either fails forever or
/// gets loosened back to substring matching, which is what let a genuinely new variant through the
/// first time.
const ENUM_VARIANT_ALIASES: &[(&str, &str, &str)] = &[
    ("mode", "h", "hash"),
    ("mode", "w", "overwrite"),
    ("mode", "a", "append"),
    // `resolve_source` also accepts `working` as a spelling of `uncommitted`, but the schema does not
    // advertise it, so it is not in scope for anchor 3 and listing it here would rot immediately.
];

/// Which production `strip_bom` call each row actually protects, established by **mutation**.
///
/// Review's sharpest finding was that the first version caught 2 of 14 sites while its docstring
/// implied it guarded the class. Recording the map keeps that honest, and recording the gaps keeps
/// them from being rediscovered as new bugs. Re-derive by replacing a call with a no-op and running
/// only this module.
const SITES_CAUGHT: &[(&str, &str)] = &[
    (
        "src/search/scope.rs",
        "search:content — the seventh leak, fixed in this commit",
    ),
    ("src/types.rs (match_text)", "search:symbol and friends"),
    (
        "src/search/callers.rs",
        "search:callers, once the BOM'd line is inside the expansion window",
    ),
    ("src/read/outline/mod.rs", "read:auto/outline"),
    (
        "src/diff/format.rs (write_diff_lines)",
        "diff:files, scoped to file detail — the raw `  1| …` source lines it reprints. The overview \
         path renders only tree-sitter identifier names, which never carry the file's leading BOM, so \
         the row must be scoped to reach a strip site at all",
    ),
    (
        "src/mcp/tools/write.rs (render_text_diff)",
        "write:overwrite/diff — the human-facing diff block, found the moment the diff-source fix let \
         the guard reach the write rows",
    ),
];

/// Sites this module still does not reach, named so they are not mistaken for covered.
const SITES_NOT_CAUGHT: &[(&str, &str)] = &[
    (
        "src/read/outline/structured.rs",
        "the `[keys]` view. No anchor requires a row — it is file-type-conditional, not an advertised \
         enum variant — and no JSON fixture I tried selects it: 400 keys, then two 2000-element \
         arrays, both resolved to `[full]`, because a keys outline lists every key and so does not \
         compress. The row was dropped rather than shipped with a `view` assertion showing it tested \
         full content while claiming the outline. Its own follow-up.",
    ),
    (
        "src/search/rank.rs",
        "#51's ranking leak. Needs a fixture where the BOM changes the *order* of two matches, not \
         just their text — and review found nothing in the whole suite catches it either. Its own \
         issue.",
    ),
    (
        "src/search/grok.rs",
        "grok renders the definition body through a path this fixture does not reach",
    ),
    (
        "src/search/symbol.rs / src/search/mod.rs (markdown heading defs)",
        "reached only when a heading def wins ranking over the code def of the same name",
    ),
    (
        "src/search/deps.rs",
        "the `deps` row renders real symbol text, so it is a `Strips` row, not `NoFileText` like \
         `diff:patch` — but the current fixture routes no BOM through a line whose extraction the \
         render reproduces. It targets `consumer.rs`, whose edges derive from `use lib::bom_target` \
         on the BOM-free line 2 and from callee resolution, not from the BOM'd line 1 (`mod lib;`). \
         Confirmed by mutation: no-op'ing `trim_start_bom_aware` (the import-detection strip) leaves \
         the row green. Falsifying it would mean targeting a file whose exported symbol is *defined* \
         on the BOM'd line — but `analyze_deps` hands unstripped content to `get_outline_entries`, so \
         that would test tree-sitter's BOM tolerance, not a strip site. Its own follow-up, like grok.",
    ),
];

/// Arguments for a surface, against a fixture rooted at `root`.
///
/// An unrecognised label panics rather than skipping: a silently-skipped surface is the thing this
/// module exists to stop.
fn args_for(label: &str, root: &Path) -> Value {
    let p = |name: &str| root.join(name).to_string_lossy().to_string();
    let scope = root.to_string_lossy().to_string();
    match label {
        "read:auto/full" => json!({"path": p("lib.rs"), "mode": "auto"}),
        "read:auto/outline" => json!({"path": p("big.rs"), "mode": "auto"}),
        "read:full" => json!({"path": p("lib.rs"), "mode": "full"}),
        "read:full/no-edit-mode" => json!({"path": p("lib.rs"), "mode": "full"}),
        "read:signature" => json!({"path": p("lib.rs"), "mode": "signature"}),
        "read:stripped" => json!({"path": p("lib.rs"), "mode": "stripped"}),
        "read:stripped/no-strip-pass" => json!({"path": p("notes.md"), "mode": "stripped"}),
        "read:section" => json!({"path": p("lib.rs"), "section": "1-3"}),
        "read:sections" => json!({"path": p("lib.rs"), "sections": ["1-2", "5-6"]}),
        "search:symbol" => json!({"query": "bom_target", "kind": "symbol", "scope": scope}),
        "search:symbol/expand=0" => {
            json!({"query": "bom_target", "kind": "symbol", "scope": scope, "expand": 0})
        }
        "search:content" => json!({"query": "bom_target", "kind": "content", "scope": scope}),
        "search:regex" => json!({"query": "bom_tar[g]et", "kind": "regex", "scope": scope}),
        // `expand: 1` keeps the window tight around line 1, which is where the BOM is — the first
        // version rendered lines 3-5 and never showed the BOM'd line at all.
        "search:callers" => {
            json!({"query": "helper", "kind": "callers", "scope": scope, "expand": 1})
        }
        "grok" => json!({"target": "bom_target", "scope": scope}),
        "deps" => json!({"path": p("consumer.rs"), "scope": scope}),
        "files" => json!({"pattern": "*.rs", "scope": scope}),
        // `scope` renders the **file detail**, not the overview. Without it the file-pair diff goes
        // through `format_overview`, which prints only symbol names and counts — and a symbol name is
        // a tree-sitter identifier node, so the file's leading BOM (which sits before `pub`) is never
        // part of it. That overview render is byte-identical BOM-vs-plain no matter what any strip
        // does: fake coverage, the same trap the first review pulled out of `diff:patch`. The scoped
        // detail path (`format_file_detail` -> `write_diff_lines`) is the one diff path that reprints
        // raw source lines (`  1| pub fn …`), so it is the only one a leading BOM can actually leak
        // through — and it did, until the strip added alongside this. `after.rs` matches the overlay
        // by `ends_with`, so no absolute-path spelling is involved.
        //
        // Forward-slash `a`/`b`: `git diff --no-index` mangles a backslash absolute path on Windows —
        // it fails to strip its own `a/`/`b/` prefixes and emits one garbled 0-symbol overlay, which
        // both hides the detail and defeats the `after.rs` scope match. Forward slashes parse cleanly
        // there and are already native on Linux. `render()` knows the forward-slash spelling of the
        // root so the header still normalises to `<ROOT>`.
        "diff:files" => json!({
            "a": p("before.rs").replace('\\', "/"),
            "b": p("after.rs").replace('\\', "/"),
            "scope": "after.rs"
        }),
        "diff:patch" => json!({"patch": p("change.patch")}),
        // `-1` (= `--max-count=1`) is the one bounded, depth-agnostic spelling. It must be bounded:
        // a bare `HEAD` enumerates every ancestor, and `diff_log` shells a `git diff` + tree-sitter
        // overlay per commit — that was the whole-history 135s. But it must also reference no parent:
        // CI checks out a shallow clone (`fetch-depth: 1`), where `HEAD~1..HEAD` fails with "unknown
        // revision" because `HEAD~1` was never fetched. `git log -1` lists exactly the tip on a deep
        // *or* shallow clone; the per-commit `hash^..hash` diff then finds no parent on a shallow tip
        // and `run_git_diff` returns empty for it (no hard error), so the row renders the tip's
        // metadata (BOM-free) either way. No `scope`: it is a suffix filter against the ambient repo's
        // paths, so the fixture path only ever filters every commit out — the `diff:source=*` misuse.
        "diff:log" => json!({"log": "-1"}),
        // No `scope`: for a git source, `scope` is a post-hoc file *filter*, not a sandbox — it never
        // redirects git away from the process cwd. Passing the fixture path as scope is wrong on every
        // platform: the git diff is taken against the tilth repo (cwd), so no overlay ever matches the
        // fixture path and the format step returns `file '<path>' not found in diff` — as `C` on
        // Windows, where the drive-letter colon also makes it read as a `file:function` scope. Dropped
        // entirely; these rows only need each source to render its BOM-free ambient diff.
        "diff:source=uncommitted" => json!({"source": "uncommitted"}),
        "diff:source=staged" => json!({"source": "staged"}),
        // `HEAD`, not `HEAD~1`: against a clean tree this is empty, so the row is `No changes.` and does
        // not depend on the content of any particular commit. A dirty tree renders the real diff, which
        // is still BOM-free — either way the `NoFileText` claim holds.
        "diff:source=ref" => json!({"source": "HEAD"}),
        // `scope` is mandatory: without it the scope guard resolves to the process cwd and refuses
        // every write to a tempdir — which is how the first version's round-trip test passed while
        // never writing anything at all, even with a deliberately wrong hash.
        "write:append" => json!({
            "scope": scope,
            "files": [{"path": p("appendee.rs"), "mode": "append", "content": "// appended\n"}]
        }),
        "write:overwrite/diff" => json!({
            "scope": scope,
            "diff": true,
            "files": [{"path": p("overwritee.rs"), "mode": "overwrite", "overwrite": true,
                       "content": "pub fn replaced() {}\n"}]
        }),
        // The anchor has to be read back per tree, since the two fixtures hash differently.
        //
        // The replacement **keeps** the leading BOM (`{BOM}pub fn ...`), which is what an agent
        // editing line 1 of a BOM'd file does — and it is what makes this a `KeepsForAnchors` row.
        // Without it the edit would replace the file's only BOM'd line with BOM-free content, leaving
        // the post-edit echo carrying no BOM to keep: the row would then be testing a stripped file
        // and pass its `contains(BOM)` check only by luck of a wrong fixture. With the BOM preserved,
        // the hashlined echo over the post-edit bytes must reproduce it — so a regression where
        // `format::hashlines` (the anchor path) began stripping is caught here.
        "write:hash" => json!({
            "scope": scope,
            "files": [{"path": p("hashee.rs"), "mode": "hash", "edits": [
                {"start": hash_anchor_for(&p("hashee.rs")),
                 "content": format!("{BOM}pub fn hashee() -> u32 {{ 2 }}")}
            ]}]
        }),
        "savings" => json!({}),
        "session" => json!({"action": "summary"}),
        other => panic!("no arguments defined for surface `{other}`"),
    }
}

/// The live `1:<hash>` anchor for a file, read back through the surface an agent would use.
///
/// A hash-mode write needs an anchor matching the file *as it is now*, and the BOM'd and BOM-free
/// fixtures hash differently, so it cannot be a constant.
fn hash_anchor_for(path: &str) -> String {
    let services = Services::new(true);
    let read = dispatch_tool(
        "tilth_read",
        &json!({"path": path, "mode": "full"}),
        &services,
    )
    .unwrap_or_else(|e| panic!("read for anchor: {e}"));
    read.lines()
        .find_map(|l| {
            let (head, _) = l.split_once('|')?;
            let (n, hash) = head.trim().split_once(':')?;
            (n == "1").then(|| format!("1:{hash}"))
        })
        .unwrap_or_else(|| panic!("no line-1 hashline in:\n{read}"))
}

/// Write the fixture. `bom` selects whether the BOM bytes are present, so the same tree can be
/// produced both ways and the two renderings compared.
///
/// Three fixture requirements, all learned from failures rather than foresight:
///
/// * **The BOM must land on a line the surface renders.** #64's requirement, and the first version
///   missed it for `callers`: the expansion window rendered lines 3-5 while the BOM sat on line 1.
/// * **Mutually exclusive render paths both need coverage** — handled by the expand/no-expand rows.
/// * **A fixture file must not contaminate rows that scan the whole tree.** The patch's BOM'd line
///   names `patched_only`, because a unified diff's content lines each begin with a marker, so a
///   BOM'd source line inside a patch is a *mid-line* BOM — which #64's non-goals exclude, since
///   `match_text` strips only from the start of a line. Named `bom_target`, it made `search:symbol`
///   fail as a leak that is documented behaviour.
fn write_fixture(root: &Path, bom: bool) {
    let b = if bom { BOM } else { "" };

    // Definition on line 1, so match text, grok bodies, the outline funnel and every read mode all
    // have to render the BOM'd line.
    std::fs::write(
        root.join("lib.rs"),
        format!(
            "{b}pub fn bom_target() -> u32 {{\n    7\n}}\n\npub fn second() -> u32 {{\n    8\n}}\n"
        ),
    )
    .unwrap();

    // `helper` is *defined* on the BOM'd line 1 and called on line 4, so a callers query with a
    // tight expansion window renders the BOM'd line rather than the body below it.
    std::fs::write(
        root.join("caller_site.rs"),
        format!(
            "{b}pub fn helper() -> u32 {{ 1 }}\n\npub fn calls_it() -> u32 {{\n    helper()\n}}\n"
        ),
    )
    .unwrap();

    // A real import of a real module, so `deps` has a dependent to report instead of `0 dependents`.
    std::fs::write(
        root.join("consumer.rs"),
        format!("{b}mod lib;\nuse lib::bom_target;\n\npub fn use_it() -> u32 {{\n    bom_target()\n}}\n"),
    )
    .unwrap();

    // A structured file with the BOM on line 1. No row reads it: the `[keys]` view could never be
    // made to select — a keys outline lists every key, so it never compresses past the OGATE, no
    // matter how the file is padded (400 keys and two 2000-element arrays both resolved to `[full]`).
    // That gap is recorded in SITES_NOT_CAUGHT, and the row was dropped rather than shipped asserting
    // full content while claiming the outline. So nothing here depends on this file's size — it is
    // only ambient tree content: a BOM'd structured file that whole-tree scans (search, files) must
    // still handle. Kept minimal; the old 4000 elements were bulk grown while fighting that same gate.
    std::fs::write(
        root.join("data.json"),
        format!("{b}{{\n  \"bom_target\": [1, 2, 3],\n  \"second\": [4, 5, 6]\n}}\n"),
    )
    .unwrap();

    // Bodies, not one-liners: the OGATE needs *compression*, not size, to select an outline (a
    // signature list for one-line functions is nearly the file itself, so it resolves to `[full]`).
    //
    // Two floors set the minimum, no more. (1) The file must clear the small-file gate to reach the
    // OGATE at all: `estimate_tokens = bytes/4 > TOKEN_THRESHOLD` (6000), i.e. **> 24_000 bytes**;
    // below it, full content is returned before the OGATE is consulted. (2) Given bodies, the outline
    // is a handful of signature lines and compresses far past the 80% bar, so nothing else is needed.
    // 60 filler functions (~33 KB) clears the byte floor with margin while staying a quarter of the
    // old 300 — the size chased while first fighting the OGATE. `read:auto/outline`'s `view` marker is
    // what proves this shrink did not tip the branch back to `[full]`.
    let mut big = format!("{b}pub fn bom_target() -> u32 {{\n");
    for i in 0..30 {
        let _ = writeln!(big, "    let x{i} = {i};");
    }
    big.push_str("    0\n}\n");
    for i in 0..60 {
        let _ = writeln!(big, "pub fn filler{i}() -> u32 {{");
        for j in 0..30 {
            let _ = writeln!(big, "    let y{j} = {j};");
        }
        let _ = writeln!(big, "    {i}\n}}");
    }
    std::fs::write(root.join("big.rs"), &big).unwrap();

    // Markdown, for the `stripped`-has-no-strip-pass row.
    std::fs::write(
        root.join("notes.md"),
        format!("{b}# bom_target\n\nProse about it.\n"),
    )
    .unwrap();

    // A file pair for `diff:files`. The BOM'd line carries the *symbol name*, which is the
    // file-derived text the structural diff renders — `patch` and `log` render none at all.
    std::fs::write(
        root.join("before.rs"),
        format!("{b}pub fn diffed_symbol() -> u32 {{\n    1\n}}\n"),
    )
    .unwrap();
    std::fs::write(
        root.join("after.rs"),
        format!("{b}pub fn diffed_symbol() -> u32 {{\n    2\n}}\n"),
    )
    .unwrap();

    // Edited by `write:hash` at its BOM'd line 1.
    std::fs::write(
        root.join("hashee.rs"),
        format!("{b}pub fn hashee() -> u32 {{ 1 }}\n"),
    )
    .unwrap();

    // No trailing newline, so append's hashline echo reprints this BOM'd line — which is the
    // anchor-bearing half of `tilth_write`.
    std::fs::write(
        root.join("appendee.rs"),
        format!("{b}pub fn appendee() {{}}"),
    )
    .unwrap();
    // Overwritten with `diff: true`, so `render_text_diff` renders the old BOM'd content.
    std::fs::write(
        root.join("overwritee.rs"),
        format!("{b}pub fn overwritee() {{}}\n"),
    )
    .unwrap();

    // `diff --git` is required: `diff::parse` sets a path only from that line, so without it no file
    // is ever flushed and the surface renders `No changes.` for both trees — which is how the first
    // version's `diff` row passed while asserting nothing.
    std::fs::write(
        root.join("change.patch"),
        format!(
            "diff --git a/lib.rs b/lib.rs\n--- a/lib.rs\n+++ b/lib.rs\n@@ -1,3 +1,4 @@\n {b}pub fn patched_only() -> u32 {{\n+    let added = 1;\n     7\n }}\n"
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
    let s = root.to_string_lossy().to_string();
    let fwd = s.replace('\\', "/");
    // The diff surfaces render a path in a spelling that is neither the OS-native root nor stable
    // across platforms, so `<ROOT>` substitution has to cover every spelling or two tempdir names
    // leak into `diff:files` and it fails as a false leak. Native tools (read, write, search) print
    // the OS path (`&s` — backslashes on Windows, forward slashes on Linux). `git` prints forward
    // slashes on both, and on Linux its `a/`/`b/` convention also strips the leading `/`, so
    // `/tmp/.tmpXXX` comes back as `tmp/.tmpXXX` (that one passed on Windows but failed CI). The
    // verbatim `\\?\` and doubled-backslash forms are older Windows spellings, kept as no-ops.
    let out = out
        .replace(&format!("\\\\?\\{s}"), "<ROOT>")
        .replace(&s.replace('\\', "\\\\"), "<ROOT>")
        .replace(&s, "<ROOT>")
        .replace(&fwd, "<ROOT>")
        .replace(fwd.trim_start_matches('/'), "<ROOT>");
    normalise_header_token_estimate(&out)
}

/// Blank the `~N tokens` estimate **on the first line only**.
///
/// A read header's estimate is derived from the file's byte length, and a BOM is three of those, so
/// `~19 tokens` against `~18` is the header telling the truth rather than leaking. The first version
/// normalised every `~N tokens` occurrence *anywhere*, and review showed that made the `files` row
/// unfalsifiable: its per-file estimate — a body line, not the header — was the only BOM-sensitive
/// datum it had. Search's `(~N tokens)` footer counts *rendered output* rather than file bytes, so the
/// justification did not hold there either. Restricting the exemption to line 1 keeps the honest part
/// and returns both of those to the strong comparison.
///
/// Everything else in the header stays exact, including the **line count** — which is where a BOM
/// genuinely does misreport, since a BOM-only file has `lines().count()` 0 stripped against 1
/// unstripped, the wrinkle #64 names.
fn normalise_header_token_estimate(out: &str) -> String {
    match out.split_once('\n') {
        Some((first, rest)) => format!("{}\n{rest}", blank_estimate(first)),
        None => blank_estimate(out),
    }
}

fn blank_estimate(line: &str) -> String {
    let Some(i) = line.find('~') else {
        return line.to_string();
    };
    let after = &line[i + 1..];
    let Some(end) = after.find(" tokens") else {
        return line.to_string();
    };
    // Only a bare number or a `3.3k`-style spelling is an estimate.
    if !after[..end].is_empty()
        && after[..end]
            .chars()
            .all(|c| c.is_ascii_digit() || c == '.' || c == 'k')
    {
        format!("{}~N tokens{}", &line[..i], &after[end + " tokens".len()..])
    } else {
        line.to_string()
    }
}

/// The main guard: every surface is byte-identical without the BOM, or declared to keep it.
///
/// The only exemption is the header's byte-derived token estimate; see
/// `normalise_header_token_estimate` for why it is narrow and what the first version got wrong.
#[test]
fn every_surface_strips_the_bom_or_is_declared_to_keep_it() {
    let bom_dir = tempfile::tempdir().unwrap();
    let plain_dir = tempfile::tempdir().unwrap();
    write_fixture(bom_dir.path(), true);
    write_fixture(plain_dir.path(), false);

    for surface in SURFACES {
        let with = render(surface, bom_dir.path());

        // A tool failure is reported inside `Ok` for `tilth_write`, which is how the first version's
        // write rows stayed invisible while being refused by the scope guard.
        assert!(
            !with.contains("\nerror: "),
            "`{}` reported an error inside a successful response, so it rendered nothing to \
             classify:\n{with}",
            surface.label
        );

        // The resolved view first: a row whose view drifted is testing a different surface than it
        // claims, and every assertion below would still pass while doing so.
        if let Some(view) = surface.view {
            let header = with.lines().next().unwrap_or_default();
            assert!(
                header.contains(view),
                "`{}` no longer resolves to {view}, so it is testing a different surface and this \
                 row's answer is stale:\n{header}",
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
            }
            Bom::KeepsForAnchors | Bom::PassesThrough => {
                assert!(
                    with.contains(BOM),
                    "`{}` is declared to keep the BOM, but stripped it:\n{with}",
                    surface.label
                );
            }
            Bom::SizesMayDifferNoContent(reason) => {
                assert!(
                    !with.contains(BOM),
                    "`{}` renders only names and sizes ({reason}), so a U+FEFF cannot be legitimate:
{with}",
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

/// ANCHOR 1 + 2: every dispatchable tool is named, and every advertised tool is dispatchable.
///
/// The dispatch direction is the one that matters and the one the first version lacked: a tool added
/// to `dispatch_tool` and never advertised was invisible, which is exactly the state `tilth_session`
/// is in. `dispatch_tool` now gates on `DISPATCHABLE_TOOLS`, so that const is authoritative.
#[test]
fn every_dispatchable_and_advertised_tool_is_named() {
    for tool in DISPATCHABLE_TOOLS {
        assert!(
            SURFACES.iter().any(|s| &s.tool == tool),
            "`{tool}` is dispatchable but no row in SURFACES exercises it. Add one — with \
             `Bom::NoFileText` and a true reason if it genuinely renders no file-derived text."
        );
    }
    for def in tool_definitions(true) {
        let name = def["name"].as_str().expect("every definition has a name");
        assert!(
            DISPATCHABLE_TOOLS.contains(&name),
            "`{name}` is advertised but not in DISPATCHABLE_TOOLS, so calling it fails"
        );
    }
    // `tilth_session` is dispatchable and unadvertised. Pinned rather than tolerated silently: if the
    // asymmetry is ever closed, this fails and the note comes out.
    let advertised: Vec<String> = tool_definitions(true)
        .iter()
        .map(|d| d["name"].as_str().unwrap_or_default().to_string())
        .collect();
    assert!(
        !advertised.iter().any(|a| a == "tilth_session"),
        "`tilth_session` is now advertised — good, but remove this assertion and the note beside it"
    );
}

/// Recursively collect `(property name, enum variants)` from a schema, at any depth.
///
/// Depth matters: `tilth_write`'s `mode` enum lives at `files.items.properties.mode`, and the first
/// version read only top-level `properties`, so it never saw it.
fn walk_schema(v: &Value, props: &mut Vec<String>, enums: &mut Vec<(String, String)>) {
    match v {
        Value::Object(map) => {
            if let Some(Value::Object(p)) = map.get("properties") {
                for (name, sub) in p {
                    props.push(name.clone());
                    if let Some(Value::Array(variants)) = sub.get("enum") {
                        for variant in variants {
                            if let Some(s) = variant.as_str() {
                                enums.push((name.clone(), s.to_string()));
                            }
                        }
                    }
                    walk_schema(sub, props, enums);
                }
            }
            for (k, sub) in map {
                if k != "properties" {
                    walk_schema(sub, props, enums);
                }
            }
        }
        Value::Array(items) => {
            for sub in items {
                walk_schema(sub, props, enums);
            }
        }
        _ => {}
    }
}

/// ANCHOR 3: every enum variant in every schema, at any depth, is named by a row — **exactly**.
///
/// Exact matching, not `contains`. The first version used substring matching, so a new `mode:
/// "outline"` or `kind: "call"` passed vacuously on the strength of the labels
/// `read:auto -> outline` and `search:callers`. Both are plausible next features.
#[test]
fn every_enum_variant_in_every_schema_is_named_by_a_row() {
    let mut props = Vec::new();
    let mut enums = Vec::new();
    for def in tool_definitions(true) {
        walk_schema(&def["inputSchema"], &mut props, &mut enums);
    }
    assert!(!enums.is_empty(), "walked no enums; the walk is broken");

    for (field, variant) in &enums {
        // A row names a variant by carrying it as a `/`-separated or `:`-separated segment of its
        // label, or as the whole label. Segment equality, never substring.
        // Resolve an alias to its long form first, so `h` is satisfied by the `hash` row.
        let canonical = ENUM_VARIANT_ALIASES
            .iter()
            .find(|(f, short, _)| f == field && short == variant)
            .map_or(variant.as_str(), |(_, _, long)| *long);
        let named = SURFACES.iter().any(|s| {
            s.label
                .split(['/', ':'])
                .any(|seg| seg == canonical || seg == format!("{field}={canonical}"))
        });
        assert!(
            named,
            "{field} `{variant}` is advertised but no row in SURFACES names it as a label segment. \
             Add a row saying whether it strips the BOM or keeps it — that decision is what five \
             rounds of this bug each skipped."
        );
    }
}

/// ANCHOR 4: every schema property is covered by a row or explicitly declared surface-less.
///
/// The anchor that catches a sub-mode expressed as anything other than an enum — a boolean, a string
/// range, a list. `section`, `sections`, `expand`, the legacy `full` boolean and `tilth_write`'s
/// `diff` were all reachable and unnamed in the first version.
#[test]
fn every_schema_property_opens_a_named_surface_or_is_declared_surface_less() {
    let mut props = Vec::new();
    let mut enums = Vec::new();
    for def in tool_definitions(true) {
        walk_schema(&def["inputSchema"], &mut props, &mut enums);
    }
    props.sort();
    props.dedup();
    assert!(props.len() > 20, "walked only {} properties", props.len());

    // A field that *is* an enum is covered by anchor 3, which names every one of its variants —
    // `kind` and `mode` are not surfaces themselves, their variants are.
    let enum_fields: Vec<&String> = enums.iter().map(|(f, _)| f).collect();

    for prop in &props {
        let covered_by_row = enum_fields.contains(&prop)
            || SURFACES
                .iter()
                .any(|s| s.label.split(['/', ':']).any(|seg| seg.starts_with(prop)));
        let declared = PARAMS_WITHOUT_THEIR_OWN_SURFACE
            .iter()
            .any(|(name, _)| name == prop);
        assert!(
            covered_by_row || declared,
            "parameter `{prop}` is advertised but neither opens a named surface nor appears in \
             PARAMS_WITHOUT_THEIR_OWN_SURFACE. Decide which it is: a new render path needs a row, \
             and anything else needs a line there saying why not."
        );
    }

    // An alias must actually be advertised, or the list is quietly absorbing future names.
    for (field, short, long) in ENUM_VARIANT_ALIASES {
        assert!(
            enums.iter().any(|(f, v)| f == field && v == short),
            "ENUM_VARIANT_ALIASES claims {field} `{short}` is a spelling of `{long}`, but no schema              advertises it"
        );
    }

    // And the reverse, so the declared list cannot rot into a list of parameters that no longer
    // exist and quietly absorb a future name.
    for (name, _) in PARAMS_WITHOUT_THEIR_OWN_SURFACE {
        assert!(
            props.iter().any(|p| p == name),
            "PARAMS_WITHOUT_THEIR_OWN_SURFACE lists `{name}`, which no schema advertises"
        );
    }
}

/// `overview` is a surface too, and it is not a tool — so no tool-derived anchor can reach it.
///
/// It is injected into the `initialize` response and is the largest single block of file-derived text
/// the server emits. #64's proposed table listed it; the first version of this module omitted it and,
/// deriving everything from `tool_definitions`, could never have noticed.
#[test]
fn the_initialize_fingerprint_strips_the_bom() {
    let bom_dir = tempfile::tempdir().unwrap();
    let plain_dir = tempfile::tempdir().unwrap();
    write_fixture(bom_dir.path(), true);
    write_fixture(plain_dir.path(), false);

    let with = crate::overview::fingerprint(bom_dir.path());
    let without = crate::overview::fingerprint(plain_dir.path());
    let norm = |s: String, root: &Path| s.replace(&root.to_string_lossy().to_string(), "<ROOT>");

    assert!(!with.is_empty(), "fingerprint rendered nothing to classify");
    assert_eq!(
        norm(with, bom_dir.path()),
        norm(without, plain_dir.path()),
        "the initialize fingerprint differs between a BOM'd tree and its BOM-free twin"
    );
}

/// The anchor contract: a hash from a BOM'd file's line 1 must still apply.
///
/// The first version asserted this and **never wrote anything** — no `scope` argument, so
/// `tool_write`'s guard resolved the scope root to the process cwd and refused every write to a
/// tempdir. The assertion was `!contains("hash mismatch")`, which a refusal also satisfies: a
/// deliberately wrong anchor `1:000` produced byte-identical output and the test passed. Now the
/// write is scoped, and a wrong anchor is asserted to fail — so the test can tell a verified hash
/// from a refused write.
#[test]
fn every_anchor_bearing_surface_round_trips_through_a_write() {
    let dir = tempfile::tempdir().unwrap();
    write_fixture(dir.path(), true);
    let root = dir.path().to_string_lossy().to_string();
    let path = dir.path().join("lib.rs").to_string_lossy().to_string();
    let services = Services::new(true);

    // Every anchor-bearing read surface must produce a line-1 anchor that verifies.
    for label in ["read:full", "read:section", "read:signature"] {
        let surface = SURFACES
            .iter()
            .find(|s| s.label == label)
            .expect("row exists");
        let read = render(surface, dir.path());
        let anchor = read
            .lines()
            .find_map(|l| {
                let (head, _) = l.split_once('|')?;
                let (n, hash) = head.trim().split_once(':')?;
                (n == "1").then(|| format!("1:{hash}"))
            })
            .unwrap_or_else(|| panic!("`{label}` emits no line-1 hashline:\n{read}"));

        let ok = dispatch_tool(
            "tilth_write",
            &json!({"scope": root, "files": [{"path": path, "mode": "hash", "edits": [
                {"start": anchor, "content": "pub fn bom_target() -> u32 {"}
            ]}]}),
            &services,
        )
        .unwrap_or_else(|e| panic!("`{label}`: scoped write failed: {e}"));
        assert!(
            !ok.contains("error:") && !ok.to_lowercase().contains("mismatch"),
            "an anchor from `{label}` over a BOM'd file did not verify, so the read stripped the \
             BOM the hash is computed over:\n{ok}"
        );
    }

    // And the control: a wrong hash must be rejected. Without this the assertions above pass on any
    // response that merely fails to say "mismatch" — including a refusal.
    let bad = dispatch_tool(
        "tilth_write",
        &json!({"scope": root, "files": [{"path": path, "mode": "hash", "edits": [
            {"start": "1:000", "content": "nope"}
        ]}]}),
        &services,
    );
    let bad_out = bad.unwrap_or_else(|e| e);
    assert!(
        bad_out.to_lowercase().contains("mismatch") || bad_out.contains("error:"),
        "a deliberately wrong anchor was accepted, so this test cannot tell a verified hash from a \
         write that never happened:\n{bad_out}"
    );
}

/// Degenerate inputs, where #64 records the remaining known wrinkles.
///
/// A file that is *only* BOM bytes: `"".lines().count()` is 0 against 1 unstripped, so a stripping
/// view and a keeping view disagree about the line count. And a doubled BOM, because tree-sitter-md
/// absorbs one on its own — a single-BOM fixture cannot detect a markdown regression, which is why
/// #51 needed the doubled spelling.
///
/// **Classified by resolved view, not requested mode**, the same lesson `SURFACES` learned. A first
/// version asserted "`mode: stripped` must not carry a BOM" and failed on `doubled.md` — correctly,
/// but for the wrong reason: markdown has no strip pass, so that read resolves to `[full]` and is in
/// the keep column. Keying on the mode name reported a leak that was not one.
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
        if out
            .lines()
            .next()
            .unwrap_or_default()
            .contains("[stripped]")
        {
            assert!(
                !out.contains(BOM),
                "a `[stripped]` view leaked a BOM from {file} (mode={mode}):\n{out}"
            );
        }
    }
}

/// The mutation map is a claim about coverage, so it has to name real files.
///
/// Cheap insurance on the honesty of `SITES_CAUGHT` / `SITES_NOT_CAUGHT`: a path that no longer
/// exists means the map is stale and its coverage claim is worthless.
#[test]
fn the_recorded_mutation_map_names_files_that_exist() {
    for (path, _) in SITES_CAUGHT.iter().chain(SITES_NOT_CAUGHT) {
        // Entries may name a function or a pair of files in parentheses; take the leading path.
        let first = path.split_whitespace().next().unwrap_or(path);
        for part in first.split('/') {
            assert!(
                !part.is_empty(),
                "malformed path in the mutation map: {path}"
            );
        }
        if first.starts_with("src/") {
            // Absolute, via `CARGO_MANIFEST_DIR`. A relative path here passed alone and failed in the
            // full suite, because another test changes the process cwd — a test that only holds when
            // run in isolation is not a test.
            let abs = Path::new(env!("CARGO_MANIFEST_DIR")).join(first);
            assert!(
                abs.exists(),
                "the mutation map names `{first}`, which no longer exists — the coverage claim is \
                 stale"
            );
        }
    }
}
