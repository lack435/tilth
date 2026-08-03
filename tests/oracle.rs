//! Differential oracle: tilth against the tools it replaces.
//!
//! The in-source unit tests check that each mechanism does what its author meant. They cannot
//! check the claim the README actually makes — that `tilth` is a safe substitute for `grep`, `cat`
//! and `find`. That claim is about agreement with a *different* implementation, so it needs a
//! second implementation to disagree with. This file carries one: a deliberately dumb, serial,
//! `std::fs`-only re-derivation of the same answers, written against no tilth code at all.
//!
//! Everything here drives the real CLI binary (`CARGO_BIN_EXE_tilth`) rather than the library, for
//! the same reason: the substitution claim is about the surface an agent touches, and going through
//! `main.rs` keeps argument parsing, scope resolution and formatting inside the thing under test.
//!
//! # The two classes of property, and why they are separated
//!
//! **Precision** — every result tilth reports is real. A reported `path:line` really does contain
//! the term; a reported definition range really does contain the symbol. This is unconditional: it
//! depends on nothing about the tree, so it holds on *any* corpus, including trees this repository
//! must never describe. `TILTH_ORACLE_ROOTS` (below) runs exactly these.
//!
//! **Recall** — nothing real is missed. This one is conditional, because tilth is allowed to omit
//! things: `SKIP_DIRS`, files over `MAX_SEARCH_FILE_SIZE`, minified bundles, and rank-truncated
//! tails it declares in the output. Asserting recall therefore means re-deriving the same exclusion
//! set, which is only honest on a tree whose shape this file controls. Recall runs on the generated
//! fixture and on this repository's own checkout.
//!
//! The split is not bookkeeping. Precision failures are the ones that actually hurt an agent: a
//! wrong line number sends it to read the wrong code and it has no way to notice. A recall failure
//! at least leaves it empty-handed.
//!
//! # Running against a private tree
//!
//! ```text
//! TILTH_ORACLE_ROOTS="/path/one;/path/two" cargo test --release --test oracle
//! ```
//!
//! Semicolon-separated (`;` not `:` — these paths have drive letters). Nothing about the roots is
//! written to the repo: failures name paths, which is fine for a local run and is why this is
//! opt-in rather than a checked-in path list.
//!
//! **With the variable unset those tests pass without checking anything**, and there is no way
//! around that worth having: `#[ignore]` would also skip them when the variable *is* set unless
//! `--ignored` is passed too, which is a worse trap. They print a skip line, which `cargo test`
//! swallows without `--nocapture`. So a green `cargo test` says nothing about any external tree —
//! read the `TILTH_ORACLE_ROOTS` run's own output, not the suite's exit code. (An earlier version
//! of this comment claimed they "report as ignored-with-a-reason"; they do not, and believing that
//! is exactly the mistake it warned about.)

// `[lints.clippy]` in Cargo.toml reaches this crate root too, so pedantic applies here. Two of its
// lints are wrong for this file specifically:
//
// * `format_push_string` — the same fixture-builder argument the lib makes under `cfg(test)`. This
//   is a test crate end to end, so the allow is unconditional rather than `cfg_attr`.
// * `naive_bytecount` — its fix is to depend on the `bytecount` crate. This file exists to be a
//   *deliberately dumb* re-derivation written against as little as possible; taking a dependency to
//   count newlines faster would trade away the only property that makes it a useful oracle.
#![allow(clippy::format_push_string, clippy::naive_bytecount)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

// ---------------------------------------------------------------------------
// Running the binary
// ---------------------------------------------------------------------------

/// Run the real CLI and return stdout, failing loudly on a non-zero exit.
fn tilth(scope: &Path, args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_tilth"))
        .args(args)
        .arg("--scope")
        .arg(scope)
        // The walk consults these; a developer's shell must not change what the suite measures.
        .env_remove("TILTH_THREADS")
        .env_remove("TILTH_FULL_SIZE_CAP")
        .output()
        .expect("failed to spawn tilth binary");

    assert!(
        out.status.success(),
        "tilth {args:?} in {} exited {:?}\nstderr:\n{}",
        scope.display(),
        out.status.code(),
        String::from_utf8_lossy(&out.stderr),
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Read one file through the MCP server and return the rendered text.
///
/// **The CLI cannot be used to test outlines.** `main.rs` sets `full = cli.full || !is_tty`, and
/// a test harness always captures stdout through a pipe, so every CLI invocation here is promoted
/// to full file content. A test that asserted "the outline contains `Foo`" against CLI output was
/// really asserting that the *source* contains `Foo` — it passed with the fix reverted.
///
/// So the outline properties go through the surface that actually produces one, which is also the
/// surface agents use. Spawns the server, does the initialize handshake, issues one `tilth_read`,
/// and closes stdin.
fn tilth_mcp_read(path: &Path) -> String {
    use std::io::Write;

    let mut child = Command::new(env!("CARGO_BIN_EXE_tilth"))
        .arg("--mcp")
        .env_remove("TILTH_THREADS")
        .env_remove("TILTH_FULL_SIZE_CAP")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("failed to spawn tilth --mcp");

    let requests = format!(
        concat!(
            r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"2024-11-05","#,
            r#""capabilities":{{}},"clientInfo":{{"name":"oracle","version":"1"}}}}}}"#,
            "\n",
            r#"{{"jsonrpc":"2.0","method":"notifications/initialized"}}"#,
            "\n",
            r#"{{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{{"name":"tilth_read","#,
            r#""arguments":{{"path":{path}}}}}}}"#,
            "\n",
        ),
        // JSON-escaped, which on Windows matters — the separators are backslashes.
        path = serde_json_string(&path.display().to_string()),
    );
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(requests.as_bytes())
        .expect("write requests");

    let out = child.wait_with_output().expect("mcp server run");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();

    // The response to id 2, pulled out without a JSON dependency: find the line carrying the
    // result, then the `"text":"…"` inside it, and unescape.
    let line = stdout
        .lines()
        .find(|l| l.contains("\"id\":2"))
        .unwrap_or_else(|| panic!("no response to tools/call in:\n{stdout}"));
    let start = line
        .find(r#""text":""#)
        .unwrap_or_else(|| panic!("no text field in:\n{line}"))
        + r#""text":""#.len();
    let mut text = String::new();
    let mut chars = line[start..].chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => break,
            // serde_json's short-escape table is exactly `" \ n t r b f` plus `\uXXXX`; a
            // decoder missing an arm turns the escape into its letter, so `\f` followed by
            // `oo` decodes to `foo`. That is a silent wrong answer in the helper whose whole
            // job is to stop this test passing for the wrong reason, which is why `b` and `f`
            // are spelled out rather than left to the catch-all.
            '\\' => match chars.next() {
                Some('n') => text.push('\n'),
                Some('t') => text.push('\t'),
                Some('r') => text.push('\r'),
                Some('b') => text.push('\u{8}'),
                Some('f') => text.push('\u{c}'),
                Some('u') => {
                    let hex: String = chars.by_ref().take(4).collect();
                    if let Some(ch) = u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                        text.push(ch);
                    }
                }
                Some(other) => text.push(other),
                None => break,
            },
            other => text.push(other),
        }
    }
    text
}

/// Minimal JSON string encoder — enough for a filesystem path, which is all this passes.
fn serde_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// ---------------------------------------------------------------------------
// The oracle: an independent re-derivation of grep / find / cat
// ---------------------------------------------------------------------------

/// Directories tilth prunes. Mirrors `search::SKIP_DIRS`.
///
/// Copied rather than imported on purpose. An oracle that imports the constant it is checking
/// cannot catch the case where the constant itself is wrong, and importing would also mean this
/// file agrees with tilth by construction on the one question — "what is off-limits?" — that the
/// recall properties exist to ask.
const SKIP_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "dist",
    "build",
    "__pycache__",
    ".pycache",
    "vendor",
    ".next",
    ".nuxt",
    "coverage",
    ".cache",
    ".tox",
    ".venv",
    ".eggs",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".turbo",
    ".parcel-cache",
    ".svelte-kit",
    "out",
    ".output",
    ".vercel",
    ".netlify",
    ".gradle",
    ".idea",
    ".scala-build",
    ".bloop",
    ".metals",
];

/// Mirrors `search::content::MAX_SEARCH_FILE_SIZE`.
const MAX_SEARCH_FILE_SIZE: u64 = 500_000;

/// Mirrors `lang::detection::MINIFIED_CHECK_THRESHOLD`.
const MINIFIED_CHECK_THRESHOLD: u64 = 100_000;

/// Every file under `root` that a content search is allowed to look at.
///
/// Deliberately serial and allocation-happy — this is the reference implementation, and being
/// obviously correct matters more here than being fast.
fn oracle_walk(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            // `symlink_metadata`, so a symlinked directory is not descended into. tilth follows
            // links; the fixture and this repo contain none, and the roots that might are only
            // ever run through precision properties, which never consult this walk.
            let Ok(meta) = std::fs::symlink_metadata(&path) else {
                continue;
            };
            if meta.file_type().is_symlink() {
                continue;
            }
            if meta.is_dir() {
                let skip = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| SKIP_DIRS.contains(&n));
                if !skip {
                    stack.push(path);
                }
            } else if meta.is_file() {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// Mirrors `lang::detection::is_minified_by_name`: `.min.<ext>` or `-min.<ext>`.
fn minified_by_name(name: &str) -> bool {
    let Some(stem_end) = name.rfind('.') else {
        return false;
    };
    let stem = &name[..stem_end];
    if let Some(secondary) = stem.rfind('.') {
        if secondary > 0 && stem[secondary + 1..].eq_ignore_ascii_case("min") {
            return true;
        }
    }
    stem.char_indices()
        .nth_back(3)
        .is_some_and(|(i, _)| stem[i..].eq_ignore_ascii_case("-min"))
}

/// Mirrors `lang::detection::is_minified_by_content`: fewer than two newlines in the first 2 KB.
///
/// The threshold and the absence of a length gate both matter, and an earlier version of this
/// had neither right — `< 5` with a `len() >= 2048` guard. Both errors made the oracle exclude
/// *more* than production, which produces false failures rather than hidden bugs, but a
/// reference implementation that has to be lucky to agree is not a reference implementation.
fn minified_by_content(bytes: &[u8]) -> bool {
    let window = &bytes[..bytes.len().min(2048)];
    window.iter().filter(|b| **b == b'\n').count() < 2
}

/// Would a content search read this file at all?
fn searchable(path: &Path) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if meta.len() > MAX_SEARCH_FILE_SIZE {
        return false;
    }
    if path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(minified_by_name)
    {
        return false;
    }
    if meta.len() >= MINIFIED_CHECK_THRESHOLD {
        let Ok(bytes) = std::fs::read(path) else {
            return false;
        };
        if minified_by_content(&bytes) {
            return false;
        }
    }
    true
}

/// `grep -rn --fixed-strings <needle>`, restricted to what tilth is allowed to read.
///
/// Returns `path -> sorted line numbers`, 1-based, paths relative to `root`.
fn oracle_grep(root: &Path, needle: &str) -> BTreeMap<PathBuf, Vec<u32>> {
    let mut hits = BTreeMap::new();
    for path in oracle_walk(root) {
        if !searchable(&path) {
            continue;
        }
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        // Lossy, matching the `UTF8` sink: a line that is not valid UTF-8 cannot match a UTF-8
        // needle either way, so the two implementations agree on it regardless.
        let text = String::from_utf8_lossy(&bytes);
        let lines: Vec<u32> = text
            .lines()
            .enumerate()
            .filter(|(_, l)| l.contains(needle))
            .map(|(i, _)| i as u32 + 1)
            .collect();
        if !lines.is_empty() {
            let rel = path.strip_prefix(root).unwrap_or(&path).to_path_buf();
            hits.insert(rel, lines);
        }
    }
    hits
}

// ---------------------------------------------------------------------------
// Parsing tilth's output
// ---------------------------------------------------------------------------

/// One `### path:lines [label]` result block.
#[derive(Debug)]
struct Hit {
    path: PathBuf,
    /// Every line the header claims, expanded: `12-20` becomes 12..=20, `12,15` becomes [12, 15].
    lines: Vec<u32>,
    /// True for `[definition]` / `[N definitions ...]` blocks.
    is_definition: bool,
    /// The raw header, so a failure message can quote what tilth actually said.
    raw: String,
}

/// Pull the `### …` result headers out of a search rendering.
///
/// Only the header line is parsed. The body is context tilth chose to show and is not a claim about
/// where the match is, so holding it to the same standard would be checking the wrong thing.
fn parse_hits(output: &str) -> Vec<Hit> {
    let mut hits = Vec::new();
    for line in output.lines() {
        let Some(rest) = line.strip_prefix("### ") else {
            continue;
        };
        // `path:lines [label]` — split the label off first so a `[` inside a path cannot confuse
        // the colon search, then take the *last* colon so Windows drive letters survive.
        let (locus, label) = match rest.rfind(" [") {
            Some(i) => (&rest[..i], &rest[i + 2..rest.len().saturating_sub(1)]),
            None => (rest, ""),
        };
        let Some(colon) = locus.rfind(':') else {
            continue;
        };
        let (path, spans) = (&locus[..colon], &locus[colon + 1..]);

        let mut lines = Vec::new();
        for span in spans.split(',') {
            let span = span.trim();
            if let Some((a, b)) = span.split_once('-') {
                // `12-` is an open range tilth prints for an unterminated block; take the anchor.
                match (a.trim().parse::<u32>(), b.trim().parse::<u32>()) {
                    (Ok(a), Ok(b)) if a <= b => lines.extend(a..=b),
                    (Ok(a), _) => lines.push(a),
                    _ => {}
                }
            } else if let Ok(n) = span.parse::<u32>() {
                lines.push(n);
            }
        }
        if lines.is_empty() {
            continue;
        }
        hits.push(Hit {
            path: PathBuf::from(path),
            lines,
            is_definition: label.contains("definition"),
            raw: line.to_string(),
        });
    }
    hits
}

/// The `— N matches` / `— N files` count from a header line.
fn header_count(output: &str) -> Option<u64> {
    let first = output.lines().next()?;
    let tail = first.rsplit_once('—')?.1;
    tail.split_whitespace().next()?.parse().ok()
}

/// Does the output declare that it withheld results? Any of these means "not everything is here",
/// and a recall assertion has to stand down when one is present.
fn declares_truncation(output: &str) -> bool {
    output.contains("... and ")
        || output.contains("more files")
        || output.contains("truncated")
        || output.contains("Narrow with scope")
        // `## Usages — same package (4/17)` — a shown/total ratio is a truncation declaration.
        || output
            .lines()
            .filter(|l| l.starts_with("##"))
            .any(|l| l.contains('/') && l.contains('(') && l.contains(')'))
}

// ---------------------------------------------------------------------------
// The fixture
// ---------------------------------------------------------------------------

/// A tree with one known plant (`ZorbulSentinel`) and one instance of every reason tilth is
/// permitted not to find it.
///
/// The token is deliberately absurd: recall assertions compare against a full-tree grep, so a term
/// that could occur incidentally — in a lockfile, a build artifact, a `.git` object — would make
/// the oracle and tilth disagree for reasons that have nothing to do with either being wrong.
struct Fixture {
    dir: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();

        let write = |rel: &str, body: &str| {
            let p = root.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, body).unwrap();
        };

        // --- Files a search must find ------------------------------------------------
        //
        // One occurrence per file, deliberately. `content_search_finds_every_file_grep_finds`
        // compares the *whole* result set against grep, which is only a meaningful comparison
        // while nothing is withheld — and the usage list truncates at eight. A denser fixture
        // pushed it over that line, and the test then passed by conceding rather than by
        // agreeing. The exclusion plants below are what give the fixture its size.
        write(
            "src/alpha.rs",
            "//! Fixture alpha.\n\
             \n\
             pub struct ZorbulSentinel {\n\
             \x20   pub id: u64,\n\
             }\n",
        );
        write(
            "src/nested/beta.rs",
            "use crate::alpha::ZorbulSentinel;\n\
             \n\
             pub fn consume(z: &u64) -> u64 {\n\
             \x20   *z\n\
             }\n",
        );
        write(
            "lib/gamma.py",
            "class ZorbulSentinel:\n\
             \x20   def __init__(self):\n\
             \x20       self.id = 2\n",
        );
        write(
            "cmd/delta.go",
            "package cmd\n\
             \n\
             type ZorbulSentinel struct {\n\
             \tID int\n\
             }\n",
        );
        write(
            "docs/notes.md",
            "# Notes\n\nThe ZorbulSentinel type is documented here.\n",
        );

        // --- Files a search must NOT find, one per documented exclusion --------------

        // Pruned directory.
        write(
            "node_modules/pkg/index.js",
            "export const ZorbulSentinel = 1;\n",
        );
        write("target/debug/gen.rs", "struct ZorbulSentinel;\n");
        write("vendor/third/party.c", "int ZorbulSentinel = 0;\n");

        // Minified by name.
        write("web/bundle.min.js", "var ZorbulSentinel=1;\n");
        write("web/other-min.js", "var ZorbulSentinel=2;\n");

        // Over the size cap: 600 KB, plant on the first line so a naive reader would find it.
        let mut big = String::from("const ZorbulSentinel = 1;\n");
        while big.len() < 600_000 {
            big.push_str("// filler filler filler filler filler filler filler filler\n");
        }
        write("web/huge.js", &big);

        // Minified by content: over 100 KB, under 500 KB, almost no newlines.
        let mut min = String::from("var ZorbulSentinel=1;");
        while min.len() < 150_000 {
            min.push_str("var a=1;b=2;c=3;d=4;e=5;f=6;g=7;h=8;i=9;j=10;k=11;l=12;");
        }
        min.push('\n');
        write("web/unmarked.js", &min);

        Self { dir }
    }

    fn root(&self) -> &Path {
        self.dir.path()
    }
}

// ---------------------------------------------------------------------------
// Property 1 — `cat`: a full read reproduces the file
// ---------------------------------------------------------------------------

/// Every line of the file, in order, byte-for-byte, in a `--full` read.
///
/// This is the whole of the `cat` claim. It is checked as an ordered subsequence rather than by
/// slicing off a fixed header, because the header's shape is formatting and is allowed to change;
/// what may not change is that no line is dropped, altered or reordered.
fn assert_cat_fidelity(root: &Path, rel: &Path) {
    let abs = root.join(rel);
    let source = std::fs::read_to_string(&abs).expect("fixture file readable");
    let rendered = tilth(root, &[&rel.to_string_lossy(), "--full"]);

    let mut cursor = rendered.lines();
    for (i, want) in source.lines().enumerate() {
        // Blank lines carry no information and the renderer is free to reflow them.
        if want.trim().is_empty() {
            continue;
        }
        let found = cursor.by_ref().any(|got| got == want);
        assert!(
            found,
            "`{}` --full dropped or altered line {} of {}\n  want: {want:?}\n\
             full output was {} lines",
            rel.display(),
            i + 1,
            rel.display(),
            rendered.lines().count(),
        );
    }
}

#[test]
fn cat_full_read_reproduces_every_line_fixture() {
    let fx = Fixture::new();
    for rel in [
        "src/alpha.rs",
        "src/nested/beta.rs",
        "lib/gamma.py",
        "cmd/delta.go",
        "docs/notes.md",
    ] {
        assert_cat_fidelity(fx.root(), Path::new(rel));
    }
}

#[test]
fn cat_full_read_reproduces_every_line_own_repo() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    // A spread of real sizes and languages, including the two biggest source files in the crate —
    // a truncation that only bites past some threshold is exactly what this needs to catch.
    for rel in [
        "src/util.rs",
        "src/error.rs",
        "src/classify.rs",
        "src/search/glob.rs",
        "src/lang/detection.rs",
        "Cargo.toml",
        "README.md",
    ] {
        assert_cat_fidelity(root, Path::new(rel));
    }
}

// ---------------------------------------------------------------------------
// Property 2 — `find`: glob totals agree
// ---------------------------------------------------------------------------

#[test]
fn glob_total_matches_an_independent_walk() {
    let fx = Fixture::new();
    let root = fx.root();

    let cases: &[(&str, &str)] = &[
        ("**/*.rs", "rs"),
        ("**/*.py", "py"),
        ("**/*.go", "go"),
        ("**/*.md", "md"),
    ];

    for (pattern, ext) in cases {
        let expected = oracle_walk(root)
            .into_iter()
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some(ext))
            .count() as u64;

        let out = tilth(root, &[pattern]);
        let got = header_count(&out)
            .unwrap_or_else(|| panic!("no count in glob header for {pattern}:\n{out}"));

        assert_eq!(
            got, expected,
            "glob {pattern} counted {got}, independent walk found {expected}\n{out}"
        );
    }
}

#[test]
fn glob_total_matches_an_independent_walk_own_repo() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let expected = oracle_walk(root)
        .into_iter()
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("rs"))
        // `fuzz/` and `examples/` are real .rs files and must be counted; nothing is excused here.
        .count() as u64;

    let out = tilth(root, &["**/*.rs"]);
    let got = header_count(&out).expect("glob header count");
    assert_eq!(
        got, expected,
        "glob **/*.rs over the tilth checkout counted {got}, independent walk found {expected}"
    );
}

#[test]
fn glob_listed_files_all_exist_and_match() {
    let fx = Fixture::new();
    let root = fx.root();
    let out = tilth(root, &["**/*.rs"]);

    let mut listed = 0;
    for line in out.lines() {
        let trimmed = line.trim();
        if !trimmed.ends_with("tokens)") {
            continue;
        }
        let rel = trimmed.split("  (~").next().unwrap().trim();
        let abs = root.join(rel);
        assert!(abs.is_file(), "glob listed a non-file: {rel}\n{out}");
        assert_eq!(
            abs.extension().and_then(|e| e.to_str()),
            Some("rs"),
            "glob **/*.rs listed a non-.rs file: {rel}"
        );
        listed += 1;
    }
    assert!(listed > 0, "glob listed nothing:\n{out}");
}

// ---------------------------------------------------------------------------
// Property 3 — `grep`: precision, then recall
// ---------------------------------------------------------------------------

/// Every `path:line` a search reports really contains the term. Holds on any tree.
///
/// Definition blocks are checked against their whole range rather than a single line: the header
/// spans the definition body, and the name legitimately appears only in the signature. Usage blocks
/// name individual lines and are held to each one.
fn assert_search_precision(root: &Path, query: &str, needle: &str, args: &[&str]) -> usize {
    let mut argv = vec![query];
    argv.extend_from_slice(args);
    let out = tilth(root, &argv);

    let mut checked = 0;
    for hit in parse_hits(&out) {
        let abs = root.join(&hit.path);
        let Ok(bytes) = std::fs::read(&abs) else {
            panic!(
                "search reported a path that cannot be read: {}\n  header: {}",
                hit.path.display(),
                hit.raw
            );
        };
        let text = String::from_utf8_lossy(&bytes);
        let lines: Vec<&str> = text.lines().collect();

        // Line numbers are 1-based and must be inside the file.
        for &n in &hit.lines {
            assert!(
                n as usize <= lines.len(),
                "search reported line {n} of {} but the file has {} lines\n  header: {}",
                hit.path.display(),
                lines.len(),
                hit.raw
            );
        }

        let present = if hit.is_definition {
            hit.lines
                .iter()
                .any(|&n| lines[n as usize - 1].contains(needle))
        } else {
            hit.lines
                .iter()
                .all(|&n| lines[n as usize - 1].contains(needle))
        };
        assert!(
            present,
            "search for {query:?} pointed at {}:{:?}, which does not contain {needle:?}\n\
             \x20 header: {}\n  line(s): {:?}",
            hit.path.display(),
            hit.lines.first(),
            hit.raw,
            hit.lines
                .iter()
                .take(4)
                .map(|&n| lines[n as usize - 1])
                .collect::<Vec<_>>(),
        );
        checked += 1;
    }
    checked
}

#[test]
fn content_search_reports_only_real_hits() {
    let fx = Fixture::new();
    let checked = assert_search_precision(fx.root(), "ZorbulSentinel", "ZorbulSentinel", &[]);
    assert!(checked > 0, "precision check saw no results to check");
}

#[test]
fn content_search_finds_every_file_grep_finds() {
    let fx = Fixture::new();
    let root = fx.root();
    let out = tilth(root, &["ZorbulSentinel"]);

    let expected: BTreeSet<PathBuf> = oracle_grep(root, "ZorbulSentinel").into_keys().collect();
    let got: BTreeSet<PathBuf> = parse_hits(&out).into_iter().map(|h| h.path).collect();

    // The fixture is small enough that nothing should be withheld. If that ever stops being true
    // the assertion below would silently weaken, so pin it.
    assert!(
        !declares_truncation(&out),
        "fixture search unexpectedly declared truncation, weakening this test:\n{out}"
    );

    let missing: Vec<_> = expected.difference(&got).collect();
    assert!(
        missing.is_empty(),
        "grep found {} file(s) tilth did not report: {missing:?}\n{out}",
        missing.len()
    );

    let extra: Vec<_> = got.difference(&expected).collect();
    assert!(
        extra.is_empty(),
        "tilth reported {} file(s) grep did not find: {extra:?}\n{out}",
        extra.len()
    );
}

#[test]
fn excluded_files_are_not_reported() {
    let fx = Fixture::new();
    let out = tilth(fx.root(), &["ZorbulSentinel"]);

    // Positive control first. Every assertion below is a `!contains`, so a search that returned
    // nothing at all would satisfy all seven and report green — this is what makes them mean
    // "the excluded files were excluded" rather than "there was no output".
    assert!(
        out.contains("alpha.rs"),
        "search returned nothing, so the exclusion assertions below prove nothing:\n{out}"
    );

    // Every one of these plants the term and is excluded for a *different* documented reason, so a
    // regression in any single exclusion shows up as a named failure rather than a count that moved.
    for (rel, why) in [
        ("node_modules", "pruned directory"),
        ("target", "pruned directory"),
        ("vendor", "pruned directory"),
        ("bundle.min.js", "minified by name"),
        ("other-min.js", "minified by name"),
        ("huge.js", "over MAX_SEARCH_FILE_SIZE"),
        ("unmarked.js", "minified by content"),
    ] {
        assert!(
            !out.contains(rel),
            "search surfaced {rel}, which is excluded as: {why}\n{out}"
        );
    }
}

#[test]
fn content_search_finds_every_file_grep_finds_own_repo() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    // A term that occurs in real source, in few enough places that nothing is truncated.
    let needle = "MINIFIED_CHECK_THRESHOLD";
    let out = tilth(root, &[needle]);

    let expected: BTreeSet<PathBuf> = oracle_grep(root, needle).into_keys().collect();
    let got: BTreeSet<PathBuf> = parse_hits(&out).into_iter().map(|h| h.path).collect();

    if declares_truncation(&out) {
        // Truncated: recall is not promised, but precision still is — everything shown is real.
        assert!(
            got.is_subset(&expected),
            "truncated search reported files grep did not find: {:?}",
            got.difference(&expected).collect::<Vec<_>>()
        );
        return;
    }
    let missing: Vec<_> = expected.difference(&got).collect();
    assert!(
        missing.is_empty(),
        "grep for {needle:?} found {} file(s) tilth did not report: {missing:?}\n{out}",
        missing.len()
    );
}

// ---------------------------------------------------------------------------
// Property 4 — symbol search: reported definitions are really definitions
// ---------------------------------------------------------------------------

/// The AST path's own precision claim: a `[definition]` range must contain the symbol name, and the
/// line must exist. This is the claim that has no cheap fallback for an agent — a definition
/// pointing at the wrong line is silently wrong in a way grep never is.
fn assert_definitions_are_real(root: &Path, symbol: &str) -> usize {
    let out = tilth(root, &[symbol]);
    let mut checked = 0;

    for hit in parse_hits(&out).into_iter().filter(|h| h.is_definition) {
        let abs = root.join(&hit.path);
        let Ok(bytes) = std::fs::read(&abs) else {
            panic!("definition in unreadable file {}", hit.path.display());
        };
        let text = String::from_utf8_lossy(&bytes);
        let lines: Vec<&str> = text.lines().collect();

        let (&first, &last) = (hit.lines.first().unwrap(), hit.lines.last().unwrap());
        assert!(
            last as usize <= lines.len(),
            "definition of {symbol} spans to line {last} but {} has {} lines\n  header: {}",
            hit.path.display(),
            lines.len(),
            hit.raw
        );
        assert!(
            hit.lines
                .iter()
                .any(|&n| lines[n as usize - 1].contains(symbol)),
            "definition of {symbol} at {}:{first}-{last} contains no occurrence of the name\n\
             \x20 header: {}\n  first line: {:?}",
            hit.path.display(),
            hit.raw,
            lines[first as usize - 1],
        );
        checked += 1;
    }
    checked
}

#[test]
fn symbol_definitions_are_real_fixture() {
    let fx = Fixture::new();
    let checked = assert_definitions_are_real(fx.root(), "ZorbulSentinel");
    assert!(
        checked >= 3,
        "expected definitions across rust/python/go, checked {checked}"
    );
}

#[test]
fn symbol_definitions_are_real_own_repo() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut total = 0;
    for symbol in [
        "OutlineCache",
        "strip_noise",
        "base_walk_builder",
        "BoundedRetain",
        "SKIP_DIRS",
        "atomic_write_bytes",
        "is_minified_by_name",
        "parse_unified_diff",
    ] {
        total += assert_definitions_are_real(root, symbol);
    }
    assert!(total > 0, "no definitions checked");
}

// ---------------------------------------------------------------------------
// Property 5 — every language's declaration forms resolve as definitions
// ---------------------------------------------------------------------------

/// One source file per language, and the names it is expected to declare.
///
/// **This is the property whose absence let the Go gap ship.** Every other test here was
/// green while `tilth_search` on a Go type name returned zero definitions and a Go file's
/// outline omitted every struct in it: nothing in the suite asked "for each language, is each
/// of its declaration forms findable?", so the answer being *no functions and methods only*
/// for one of seventeen supported languages was invisible.
///
/// Names are unique across the whole table so a single-scope search cannot pass by finding
/// the right name in the wrong language's file.
const DECLARATION_FORMS: &[(&str, &str, &[&str])] = &[
    (
        "a.rs",
        "pub struct RsStruct { pub id: u64 }\n\
         pub enum RsEnum { X }\n\
         pub trait RsTrait { fn f(&self); }\n\
         pub type RsAlias = u64;\n\
         pub const RS_CONST: u64 = 1;\n\
         pub static RS_STATIC: u64 = 2;\n\
         pub fn rs_func() -> u64 { 1 }\n\
         pub mod rs_module { }\n",
        &[
            "RsStruct",
            "RsEnum",
            "RsTrait",
            "RsAlias",
            "RS_CONST",
            "RS_STATIC",
            "rs_func",
            "rs_module",
        ],
    ),
    (
        "b.go",
        "package b\n\n\
         type GoStruct struct{ ID int }\n\n\
         type GoIface interface{ Serve() error }\n\n\
         type GoAlias = int\n\n\
         type GoNamed int\n\n\
         type (\n\tGoGroupedA struct{ X int }\n\tGoGroupedB int\n)\n\n\
         const GoConst = 5\n\n\
         const (\n\tGoGroupedConst = 1\n)\n\n\
         var GoVar = 7\n\n\
         var (\n\tGoGroupedVar int\n)\n\n\
         func GoFunc() int { return 1 }\n\n\
         func (g *GoStruct) GoMethod() int { return 2 }\n",
        &[
            "GoStruct",
            "GoIface",
            "GoAlias",
            "GoNamed",
            "GoGroupedA",
            "GoGroupedB",
            "GoConst",
            "GoGroupedConst",
            "GoVar",
            "GoGroupedVar",
            "GoFunc",
            "GoMethod",
        ],
    ),
    (
        "c.ts",
        "export class TsClass { }\n\
         export interface TsIface { x: number }\n\
         export type TsType = number;\n\
         export enum TsEnum { A }\n\
         export const TS_CONST = 1;\n\
         export function tsFunc(): number { return 1 }\n",
        &[
            "TsClass", "TsIface", "TsType", "TsEnum", "TS_CONST", "tsFunc",
        ],
    ),
    (
        "d.py",
        "class PyClass:\n\
         \x20   def py_method(self): pass\n\n\
         def py_func(): pass\n",
        &["PyClass", "py_method", "py_func"],
    ),
    (
        "e.java",
        "public class JavaClass {\n\
         \x20   public interface JavaIface { }\n\
         \x20   public void javaMethod() { }\n\
         }\n\
         enum JavaEnum { A }\n\
         record JavaRecord(int x) { }\n",
        &[
            "JavaClass",
            "JavaIface",
            "javaMethod",
            "JavaEnum",
            "JavaRecord",
        ],
    ),
    (
        "f.cs",
        "public class CsClass {\n\
         \x20   public interface CsIface { }\n\
         \x20   public int CsProp { get; set; }\n\
         \x20   public void CsMethod() { }\n\
         }\n\
         public enum CsEnum { A }\n\
         public struct CsStruct { }\n\
         public record CsRecord(int X);\n",
        &[
            "CsClass", "CsIface", "CsProp", "CsMethod", "CsEnum", "CsStruct", "CsRecord",
        ],
    ),
    (
        "g.rb",
        "class RbClass\n\
         \x20 def rb_method; end\n\
         end\n\
         module RbModule; end\n",
        &["RbClass", "rb_method", "RbModule"],
    ),
    (
        "h.cpp",
        "class CppClass { public: void CppMethod(); };\n\
         struct CppStruct { int x; };\n\
         union CppUnion { int a; float b; };\n\
         enum CppEnum { A };\n\
         enum class CppEnumClass : int { B };\n\
         typedef int CppAlias;\n\
         using CppUsing = float;\n\
         namespace CppNamespace { void Inner(); }\n\
         template <typename T> class CppTemplate { T v; };\n\
         template <typename T> T CppTemplateFunc(T a) { return a; }\n\
         void CppFunc() { }\n",
        &[
            "CppClass",
            "CppMethod",
            "CppStruct",
            "CppUnion",
            "CppEnum",
            "CppEnumClass",
            "CppAlias",
            "CppUsing",
            "CppNamespace",
            "CppTemplate",
            "CppTemplateFunc",
            "CppFunc",
        ],
    ),
];

// ---------------------------------------------------------------------------
// Property 6 — Unreal-flavoured C++, the fork's primary corpus
// ---------------------------------------------------------------------------

/// A header in the shape UE actually writes them, and every name it declares.
///
/// The reflection macros are not decoration here — they are the thing under test. A `UENUM`
/// whose enumerators carry `UMETA` used to make tree-sitter-cpp swallow the `enum` keyword
/// into an `ERROR` node that extended over **every declaration after it**, so a header like
/// this one outlined to nothing and none of the names below the enum resolved. One macro at
/// the top of a file, and the rest of the file was invisible.
const UE_HEADER: &str = "\
#pragma once

UENUM(BlueprintType)
enum class EProbeMode : uint8
{
    ModeAlpha UMETA(DisplayName = \"Alpha\"),
    ModeBeta  UMETA(ToolTip = \"Fire (primary)\")
};

USTRUCT(BlueprintType)
struct PROBEMODULE_API FProbeData
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = \"Probe\")
    int32 ProbeCount = 0;

    UPROPERTY()
    FString ProbeName;
};

UCLASS(Blueprintable)
class PROBEMODULE_API AProbeActor : public AActor
{
    GENERATED_BODY()

public:
    AProbeActor();

    UFUNCTION(BlueprintCallable, Category = \"Probe\")
    void DoProbeThing(int32 Amount);

    virtual void BeginPlay() override;

private:
    UPROPERTY(VisibleAnywhere)
    class UProbeComponent* ProbeComp;
};

UINTERFACE(MinimalAPI)
class UProbeInterface : public UInterface { GENERATED_BODY() };
";

const UE_DECLARED_NAMES: &[&str] = &[
    "EProbeMode",
    "FProbeData",
    "ProbeCount",
    "ProbeName",
    "AProbeActor",
    "DoProbeThing",
    "ProbeComp",
    "UProbeInterface",
];

#[test]
fn ue_header_declarations_all_resolve() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path();
    std::fs::write(root.join("Probe.h"), UE_HEADER).unwrap();

    let mut missing = Vec::new();
    for name in UE_DECLARED_NAMES {
        let out = tilth(root, &[name]);
        if parse_hits(&out).iter().all(|h| !h.is_definition) {
            missing.push(*name);
        }
    }
    assert!(
        missing.is_empty(),
        "declared in the UE header but not findable as definitions: {missing:?}"
    );
}

/// The outline is the surface the collapse was most visible on: it reads the parse tree's
/// top-level children, so one `ERROR` spanning the file left it with nothing to render.
///
/// Goes through MCP, not the CLI — see `tilth_mcp_read`. The CLI promotes piped reads to full
/// content, so the first version of this test was asserting that the *source* contained these
/// names and passed with the fix reverted.
///
/// Padded past `TOKEN_THRESHOLD` because a small file is returned whole either way; the outline
/// path only runs on files big enough to need it, which is where a real UE header lands.
#[test]
fn ue_header_outline_is_not_empty() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path();

    let mut src = String::from(UE_HEADER);
    src.push_str("\nclass FPadding\n{\npublic:\n");
    for i in 0..400 {
        src.push_str(&format!(
            "    UFUNCTION(BlueprintCallable, Category = \"Probe\")\n    void PadMethod{i}(int32 AmountValue);\n\n"
        ));
    }
    src.push_str("};\n");
    let path = root.join("Probe.h");
    std::fs::write(&path, &src).unwrap();

    let out = tilth_mcp_read(&path);
    // Assert the mode first. Without this the name assertions below could be satisfied by full
    // file content, which is exactly how this test used to pass while broken.
    //
    // Checked against the *header line*, not the whole payload: the mode marker lives there,
    // and a `contains` over everything would also be satisfied by a fixture that merely
    // mentions `[outline]` in its own text — a smaller version of the same vacuity.
    let header = out.lines().next().unwrap_or_default();
    assert!(
        header.contains("[outline]"),
        "expected an outline rendering, got header {header:?} in:\n{}",
        &out[..out.len().min(400)]
    );
    for name in ["EProbeMode", "FProbeData", "AProbeActor"] {
        assert!(
            out.contains(name),
            "{name} missing from the outline of a UE header:\n{}",
            &out[..out.len().min(1200)]
        );
    }
}

#[test]
fn every_language_resolves_its_declaration_forms() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path();
    for (name, body, _) in DECLARATION_FORMS {
        std::fs::write(root.join(name), body).unwrap();
    }

    let mut missing: Vec<String> = Vec::new();
    for (file, _, names) in DECLARATION_FORMS {
        for name in *names {
            let out = tilth(root, &[name]);
            let defs = parse_hits(&out).into_iter().filter(|h| h.is_definition);
            if defs.count() == 0 {
                missing.push(format!("{file}: {name}"));
            }
        }
    }

    assert!(
        missing.is_empty(),
        "these declarations resolve as usages but not as definitions — a search for them \
         returns no Definitions section at all:\n  {}",
        missing.join("\n  ")
    );
}

// ---------------------------------------------------------------------------
// External roots — precision only, on trees this repository does not describe
// ---------------------------------------------------------------------------

/// Roots from `TILTH_ORACLE_ROOTS`, semicolon-separated. Empty when unset.
fn external_roots() -> Vec<PathBuf> {
    std::env::var("TILTH_ORACLE_ROOTS")
        .ok()
        .into_iter()
        .flat_map(|v| {
            v.split(';')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(PathBuf::from)
                .collect::<Vec<_>>()
        })
        .filter(|p| p.is_dir())
        .collect()
}

/// Symbols to probe an unknown tree with.
///
/// Chosen to *return results* on a tree nobody described in advance, so the precision assertions
/// have something to check — not for meaning anything in particular. Anything that finds nothing
/// is skipped, not failed.
///
/// Weighted towards C++ and Unreal spellings because that is this fork's primary corpus: the
/// generic verbs below hit any language, and the type-prefix conventions (`F`, `U`, `A`, `T`,
/// `E`) put the probes inside real UE headers, where the macro-heavy shapes live.
const PROBES: &[&str] = &[
    // Generic, any language.
    "init",
    "run",
    "get",
    "set",
    "update",
    "parse",
    "close",
    "read",
    "write",
    "handle",
    "create",
    "delete",
    "config",
    "Result",
    "Error",
    "Context",
    "Manager",
    "Buffer",
    // C++ and Unreal.
    "Tick",
    "BeginPlay",
    "GetWorld",
    "Serialize",
    "FVector",
    "FString",
    "FName",
    "UObject",
    "AActor",
    "TArray",
    "TMap",
    "USceneComponent",
    "OnRep",
    "StaticClass",
];

#[test]
fn external_roots_report_only_real_hits() {
    let roots = external_roots();
    if roots.is_empty() {
        eprintln!("SKIP: set TILTH_ORACLE_ROOTS to run this");
        return;
    }

    let mut total = 0;
    for root in &roots {
        for probe in PROBES {
            total += assert_search_precision(root, probe, probe, &[]);
        }
        eprintln!(
            "{}: precision-checked {total} result blocks",
            root.display()
        );
    }
    assert!(
        total > 0,
        "no results across any external root — the probes found nothing, so nothing was verified"
    );
}

#[test]
fn external_roots_definitions_are_real() {
    let roots = external_roots();
    if roots.is_empty() {
        eprintln!("SKIP: set TILTH_ORACLE_ROOTS to run this");
        return;
    }

    let mut total = 0;
    for root in &roots {
        for probe in PROBES {
            total += assert_definitions_are_real(root, probe);
        }
        eprintln!("{}: verified {total} definition ranges", root.display());
    }
    assert!(total > 0, "no definitions found across any external root");
}

#[test]
fn external_roots_full_read_reproduces_files() {
    let roots = external_roots();
    if roots.is_empty() {
        eprintln!("SKIP: set TILTH_ORACLE_ROOTS to run this");
        return;
    }

    for root in &roots {
        // Sample real source files from the tree rather than naming any: the paths in these trees
        // are not this repository's to record.
        let sample: Vec<PathBuf> = oracle_walk(root)
            .into_iter()
            .filter(|p| {
                matches!(
                    p.extension().and_then(|e| e.to_str()),
                    Some("rs" | "py" | "go" | "ts" | "js" | "c" | "h" | "cpp" | "cs" | "java")
                )
            })
            .filter(|p| std::fs::metadata(p).is_ok_and(|m| m.len() > 200 && m.len() < 60_000))
            .step_by(97)
            .take(25)
            .collect();

        assert!(
            !sample.is_empty(),
            "{}: no source files sampled",
            root.display()
        );
        for abs in sample {
            let rel = abs.strip_prefix(root).unwrap();
            assert_cat_fidelity(root, rel);
        }
    }
}
