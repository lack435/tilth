use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::time::SystemTime;

use crate::types::{is_test_file, Match};

const VENDOR_DIRS: &[&str] = &[
    "node_modules",
    "vendor",
    "dist",
    "build",
    ".git",
    "target",
    "__pycache__",
    ".venv",
    "venv",
    "pkg",
    "out",
];

/// Reorder `slice` so the element at index `i` ends up at `dest[i]`.
///
/// `dest` must be a permutation of `0..slice.len()`. It is modified in place and left as
/// the identity.
///
/// Applied with cycle swaps, so this allocates nothing and never holds a second buffer of
/// `T`. That is the whole point: `T` here is `Match`, and the sets it runs on stopped
/// being bounded when the search walks were made to complete. Rust's stable sort asks for
/// `n/2 * size_of::<T>()` of scratch — measured at 68 bytes per match, 163 MB on a
/// 2.4M-match search — and every caller below replaced a sort that paid it.
///
/// Correctness rests on `dest` staying a permutation: each iteration puts one element at
/// its final index and creates one new fixed point, so the total work is O(n), the inner
/// loop cannot spin, and no cycle can be left unresolved once `i` moves past it.
pub(crate) fn apply_destination_permutation<T>(slice: &mut [T], dest: &mut [usize]) {
    debug_assert_eq!(
        slice.len(),
        dest.len(),
        "dest must be a permutation of the slice's indices"
    );
    // A `dest` that is not a permutation — a duplicate or out-of-range entry — makes the
    // loop below spin forever instead of failing, because `dest[i] != i` can then never
    // settle. Worth an O(n) debug-only check: a caller that miscomputes offsets hangs the
    // process otherwise, which is exactly how it presented when a mutation test broke the
    // counting-sort offsets in `stratify_for_display`.
    debug_assert!(
        {
            let mut seen = vec![false; dest.len()];
            dest.iter()
                .all(|&d| d < seen.len() && !std::mem::replace(&mut seen[d], true))
        },
        "dest must be a permutation of 0..len (found a duplicate or out-of-range index)"
    );
    for i in 0..slice.len() {
        while dest[i] != i {
            let j = dest[i];
            slice.swap(i, j);
            dest.swap(i, j);
        }
    }
}

/// Sort matches by score (highest first). Deterministic: same inputs, same order.
/// When `context` is provided, matches near the context file are boosted.
///
/// `score` is evaluated exactly **once per match**, not inside the comparator. It used to
/// be called twice per comparison, and it is not a cheap function — `incidental_text_penalty`
/// lowercases the whole matched line, so each call allocates. That was free while an
/// early-quit threshold held `n` to ~80 entries; once the walks were made to complete (so
/// results stop varying run to run) `n` became the true match count.
///
/// Measured on a dense 400-file fixture, ~2.4M matches, one session, by stubbing pieces out:
///
/// ```text
/// walk + render, no sort at all      0.36s   380 MB
/// + score once (2.4M calls)          1.92s   382 MB
/// + index sort and permute (this)    3.93s   414 MB
/// the sort this replaced             32.5s   462 MB
/// ```
///
/// So `sort` was 98.9% of that search's wall time. Note what the rows do and do not
/// attribute: scoring is 1.56s of the new 3.93s and the sort machinery is the other 2.0s,
/// so it would be wrong to read the 32s as `score` alone — stubbing the sort also removed
/// n log n moves of a 136-byte element and the stable sort's scratch buffer. `score` costs
/// ~650ns per call here; the old sort made order-1e8 calls, this one makes 2.4e6.
///
/// The key has since been extended to a total order — see the comment on the comparator. For
/// every input that already had a unique `(score, path, line)` per match, which is all of them
/// outside the overload case, the ordering is unchanged and output is byte-identical.
///
/// Sorting an index permutation rather than using `sort_by_cached_key` is deliberate: a
/// cached key must *own* what it compares, so breaking ties on path means cloning a
/// `PathBuf` per match on a path already resident, and `sort_by_cached_key` allocates its
/// own `Vec<(K, usize)>` on top. `scores` + `order` + `dest` peak at 20 bytes per match,
/// against `size_of::<Match>()` of 136 plus its heap.
pub fn sort(matches: &mut [Match], query: &str, scope: &Path, context: Option<&Path>) {
    // Pre-compute context's package root once (same for entire batch)
    let ctx_parent = context.and_then(|c| c.parent());
    let ctx_pkg_root = context
        .and_then(crate::lang::package_root)
        .map(std::path::Path::to_path_buf);

    // Cache package roots for match paths — avoids repeated stat walks
    let mut pkg_cache: HashMap<PathBuf, Option<PathBuf>> = HashMap::new();
    // Capture now once so scoring does not call SystemTime::now() per match.
    let now = SystemTime::now();

    let scores: Vec<i32> = matches
        .iter()
        .map(|m| {
            score(
                m,
                query,
                scope,
                ctx_parent,
                ctx_pkg_root.as_ref(),
                &mut pkg_cache,
                now,
            )
        })
        .collect();

    // Sort indices, so the comparator reads scores by index and paths by reference.
    //
    // `(score, path, line)` alone is **not** a total order: two overload declarations on one
    // line compare equal, and `SameSpanDedupe` deliberately keeps both. That used to be covered
    // by arrival order — a stable sort plus the "one contiguous block per file" invariant in
    // `symbol.rs` meant ties could only ever be within a file, in a fixed within-file order.
    //
    // Relying on arrival order caps how the callers can collect. A bounded retention heap has
    // to be able to drop a match from the middle, which destroys contiguity, so symbol search
    // could not be bounded while determinism rested on arrival. Extending the key to a genuine
    // total order removes that coupling: ties are now broken by data the match carries, so any
    // collection order gives the same output.
    //
    // `def_range` separates the overload case: same path, same line, different span. `text` is
    // the backstop for anything else reaching here — two matches agreeing on all four are
    // indistinguishable to a reader, so their relative order is not observable. Both are `Ord`
    // and neither allocates in the comparator.
    let mut order: Vec<usize> = (0..matches.len()).collect();
    order.sort_by(|&i, &j| {
        scores[j]
            .cmp(&scores[i])
            .then_with(|| matches[i].path.cmp(&matches[j].path))
            .then_with(|| matches[i].line.cmp(&matches[j].line))
            .then_with(|| matches[i].def_range.cmp(&matches[j].def_range))
            .then_with(|| matches[i].text.cmp(&matches[j].text))
    });

    // `order[k]` is the index of the element that belongs at position `k`; the helper
    // wants the inverse — where does the element at `i` belong. Inverting is load-bearing,
    // not bookkeeping: feeding `order` in directly applies the inverse permutation, which
    // is a real and silent misordering.
    let mut dest: Vec<usize> = vec![0; order.len()];
    for (k, &from) in order.iter().enumerate() {
        dest[from] = k;
    }

    apply_destination_permutation(matches, &mut dest);
}

/// Scores a match against one query, for use *during* a walk.
///
/// Exists so a search can decide what to retain while it walks, instead of retaining
/// everything and deciding afterwards. Owns the per-query context that `sort` otherwise
/// recomputes, so a worker thread can construct one and score its own matches.
///
/// `selection_score` deliberately **omits the recency term**, and that is the whole reason
/// this is a separate entry point rather than a call to `score`. `recency` buckets by age
/// at 1h/1d/1w/1mo, so a file crossing a boundary between two runs changes its score. That
/// is harmless when every match is retained — `now` only reorders the displayed page — but a
/// score-gated *retention* bound would let the wall clock decide which matches exist at all,
/// which is the defect removed from `overview::hot_files` and from the search walks before
/// it. Selecting on a time-independent key keeps membership a function of the tree, and the
/// full ranking (recency included) still orders the retained set.
///
/// The residual is bounded, and the bound is *not* "recency is small". Recency is worth up to
/// 100 points, and a content match's entire score is about 230 — `is_definition` and `exact` are
/// both false for every content match, which removes two 500-point terms. So the retained set
/// must be deep enough that a match within 100 points of the retention cut is still kept; see
/// `content::MAX_RETAINED`, which was set an order of magnitude too small on the strength of an
/// unchecked claim that scores are "in the thousands".
pub(crate) struct Scorer<'a> {
    query: &'a str,
    scope: &'a Path,
    ctx_parent: Option<&'a Path>,
    ctx_pkg_root: Option<PathBuf>,
    pkg_cache: HashMap<PathBuf, Option<PathBuf>>,
}

impl<'a> Scorer<'a> {
    pub(crate) fn new(query: &'a str, scope: &'a Path, context: Option<&'a Path>) -> Self {
        Scorer {
            query,
            scope,
            ctx_parent: context.and_then(Path::parent),
            ctx_pkg_root: context
                .and_then(crate::lang::package_root)
                .map(std::path::Path::to_path_buf),
            pkg_cache: HashMap::new(),
        }
    }

    /// Time-independent ranking score. See the struct docs for why recency is excluded.
    pub(crate) fn selection_score(&mut self, m: &Match) -> i32 {
        score_inner(
            m,
            self.query,
            self.scope,
            self.ctx_parent,
            self.ctx_pkg_root.as_ref(),
            &mut self.pkg_cache,
            None,
        )
    }
}

/// Ranking function. Each match gets a score — no floating point, no randomness.
/// All boosts are positive (added), all penalties are positive (subtracted).
fn score(
    m: &Match,
    query: &str,
    scope: &Path,
    ctx_parent: Option<&Path>,
    ctx_pkg_root: Option<&PathBuf>,
    pkg_cache: &mut HashMap<PathBuf, Option<PathBuf>>,
    now: SystemTime,
) -> i32 {
    score_inner(
        m,
        query,
        scope,
        ctx_parent,
        ctx_pkg_root,
        pkg_cache,
        Some(now),
    )
}

/// `now = None` omits the recency term, yielding a score that does not depend on the clock.
/// Every other term is identical, so the two callers cannot drift apart.
fn score_inner(
    m: &Match,
    query: &str,
    scope: &Path,
    ctx_parent: Option<&Path>,
    ctx_pkg_root: Option<&PathBuf>,
    pkg_cache: &mut HashMap<PathBuf, Option<PathBuf>>,
    now: Option<SystemTime>,
) -> i32 {
    let mut s = 0i32;

    if m.is_definition {
        s += i32::from(m.def_weight) * 10;
        s += definition_name_boost(m, query);
    }
    if m.exact {
        s += 500;
    }

    s += query_intent_boost(m, query);
    s += multi_word_boost(m, query);
    s += scope_proximity(&m.path, scope) as i32;
    if let Some(now) = now {
        s += recency(m.mtime, now) as i32;
    }

    if m.file_lines > 0 && m.file_lines < 200 {
        s += 50;
    }

    // Context-aware boosts
    if ctx_parent.is_some() || ctx_pkg_root.is_some() {
        s += context_proximity(&m.path, ctx_parent, ctx_pkg_root, pkg_cache);
    }

    s += basename_boost(&m.path, query);
    s += exported_api_boost(m);
    s -= non_code_penalty(&m.path);
    s -= incidental_text_penalty(m, query);

    // Scope-relative: a `__tests__` directory *above* the searched tree must not dock every
    // result in it 120 points. See `is_test_file`.
    if is_test_file(m.path.strip_prefix(scope).unwrap_or(&m.path)) && !looks_like_test_query(query)
    {
        s -= 120;
    }
    s -= fixture_penalty(m);

    // Vendor penalty (always active)
    if is_vendor_path(&m.path) {
        s -= 200;
    }

    s
}

/// Boost matches whose file stem matches the query.
fn basename_boost(path: &Path, query: &str) -> i32 {
    if query.is_empty() {
        return 0;
    }

    let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
        return 0;
    };
    let stem_lower = stem.to_ascii_lowercase();
    let query_lower = query.to_ascii_lowercase();

    if stem_lower == query_lower {
        return 500;
    }
    if stem_lower.starts_with(&query_lower)
        && stem_lower
            .as_bytes()
            .get(query_lower.len())
            .is_some_and(|&b| b == b'_' || b == b'.' || b == b'-')
    {
        return 350;
    }
    if stem_lower.ends_with(&query_lower) {
        return 250;
    }
    if stem_lower.contains(&query_lower) {
        return 180;
    }

    let parent_name = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("");
    if parent_name.eq_ignore_ascii_case(query) {
        return 200;
    }

    0
}

/// 0-200, closer to scope root = higher.
fn scope_proximity(path: &Path, scope: &Path) -> u32 {
    let rel = path.strip_prefix(scope).unwrap_or(path);
    let depth = rel.components().count();
    200u32.saturating_sub(depth as u32 * 20)
}

/// Context-aware proximity boost with cached package roots.
fn context_proximity(
    match_path: &Path,
    ctx_parent: Option<&Path>,
    ctx_pkg_root: Option<&PathBuf>,
    pkg_cache: &mut HashMap<PathBuf, Option<PathBuf>>,
) -> i32 {
    let mut score = 0;

    // Same directory as context file
    if let Some(cp) = ctx_parent {
        if match_path.parent() == Some(cp) {
            score += 100;
        } else if shared_prefix_depth(cp, match_path.parent().unwrap_or(match_path)) >= 2 {
            score += 40;
        }
    }

    // Same package root (cached)
    if let Some(cp_root) = ctx_pkg_root {
        let match_dir = match match_path.parent() {
            Some(d) => d.to_path_buf(),
            None => return score,
        };
        let match_root = pkg_cache.entry(match_dir).or_insert_with_key(|dir| {
            crate::lang::package_root(dir).map(std::path::Path::to_path_buf)
        });
        if let Some(ref mr) = match_root {
            if mr == cp_root {
                score += 75;
            }
        }
    }

    score
}

fn definition_name_boost(m: &Match, query: &str) -> i32 {
    let Some(name) = m.def_name.as_deref() else {
        return 0;
    };

    let query_lower = query.to_ascii_lowercase();
    let name_lower = name.to_ascii_lowercase();

    if name == query {
        220
    } else if name_lower == query_lower {
        180
    } else if m.impl_target.as_deref() == Some(query) {
        120
    } else if name_lower.starts_with(&query_lower) {
        80
    } else if name_lower.contains(&query_lower) {
        40
    } else {
        0
    }
}

fn query_intent_boost(m: &Match, query: &str) -> i32 {
    if query.is_empty() {
        return 0;
    }

    let looks_type = query.chars().next().is_some_and(char::is_uppercase);
    let looks_fn = query.chars().next().is_some_and(char::is_lowercase);
    // BOM-aware, because `str::trim_start` is not — see `types::match_text`.
    //
    // Unreachable today, and deliberately kept: all nine `Match.text` construction sites go
    // through `match_text`, and nothing in live code assigns that field otherwise, so no
    // `Match` currently reaches ranking carrying a BOM. The rank test has to hand-build one to
    // exercise this, which is the honest signal that it is belt-and-braces rather than a live
    // fix. It stays because the failure it guards is invisible: a construction site added later
    // that forgets the helper costs a visible rendering glyph, but without this it would *also*
    // cost 130 points of score with nothing in the output to show it.
    let text = crate::lang::outline::trim_start_bom_aware(&m.text);

    if looks_type
        && (text.starts_with("struct ")
            || text.starts_with("pub struct ")
            || text.starts_with("enum ")
            || text.starts_with("pub enum ")
            || text.starts_with("trait ")
            || text.starts_with("pub trait ")
            || text.starts_with("interface ")
            || text.starts_with("export interface ")
            || text.starts_with("type ")
            || text.starts_with("export type ")
            || text.starts_with("class ")
            || text.starts_with("export class ")
            || text.starts_with("impl "))
    {
        return 90;
    }

    if looks_fn
        && (text.starts_with("fn ")
            || text.starts_with("pub fn ")
            || text.starts_with("pub(crate) fn ")
            || text.starts_with("async fn ")
            || text.starts_with("pub async fn ")
            || text.starts_with("function ")
            || text.starts_with("export function ")
            || text.starts_with("export default function ")
            || text.starts_with("export async function "))
    {
        return 70;
    }

    0
}

fn exported_api_boost(m: &Match) -> i32 {
    // BOM-aware — see `definition_kind_boost`.
    let text = crate::lang::outline::trim_start_bom_aware(&m.text);

    if text.starts_with("export default ") {
        90
    } else if text.starts_with("export ") {
        75
    } else if text.starts_with("pub ") {
        60
    } else {
        0
    }
}

/// Penalize matches in test fixtures, mocks, stubs, etc. Returns 90 for a path
/// with a fixture component, else 0.
fn fixture_penalty(m: &Match) -> i32 {
    // Anchor path matching to a PATH COMPONENT to avoid false positives like
    // `examples_parser.rs` being penalized because "examples" is a substring.
    let has_fixture_component = m.path.components().any(|c| {
        c.as_os_str().to_str().is_some_and(|s| {
            let s = s.to_ascii_lowercase();
            matches!(
                s.as_str(),
                "mock"
                    | "mocks"
                    | "fixture"
                    | "fixtures"
                    | "stub"
                    | "stubs"
                    | "fake"
                    | "fakes"
                    | "example"
                    | "examples"
            )
        })
    });

    if has_fixture_component {
        90
    } else {
        0
    }
}

/// Penalize matches that appear only in comments (not code).
fn incidental_text_penalty(m: &Match, query: &str) -> i32 {
    if m.is_definition {
        return 0;
    }

    // BOM-aware — see `definition_kind_boost`. Here the sign is reversed: a BOM'd line-1
    // comment failed `starts_with("//")` and so *escaped* the 150-point penalty, ranking
    // above where it belonged.
    let text = crate::lang::outline::trim_start_bom_aware(&m.text).trim_end();
    let q_lower = query.to_ascii_lowercase();

    // Only use unambiguous comment prefixes — avoid '#' (Python/C preprocessor/Rust attrs)
    // and '*' (could be pointer deref, multiplication, glob, etc.)
    // Exempt /// doc comments — they're often the most useful context for a symbol.
    let is_comment = (text.starts_with("//") && !text.starts_with("///"))
        || text.starts_with("/*")
        || text.starts_with("<!--");

    // For '#' lines: only treat as comment in languages where # is always a comment
    let is_hash_comment = text.starts_with('#') && {
        let ext = m.path.extension().and_then(|e| e.to_str()).unwrap_or("");
        matches!(
            ext,
            "py" | "rb" | "sh" | "bash" | "zsh" | "yaml" | "yml" | "toml" | "pl" | "r" | "R"
        )
    };

    if is_comment || is_hash_comment {
        return 150;
    }

    // Check if query only appears in a trailing comment (after //)
    // Skip false positives: :// is a URL scheme separator, not a comment
    // Skip // at start of line — that's a full-line comment, not trailing
    let t_lower = text.to_ascii_lowercase();
    if let Some(slash_pos) = t_lower.find("//") {
        let is_url = slash_pos > 0 && t_lower.as_bytes()[slash_pos - 1] == b':';
        if slash_pos > 0
            && !is_url
            && t_lower[slash_pos..].contains(&q_lower)
            && !t_lower[..slash_pos].contains(&q_lower)
        {
            return 100;
        }
    }

    0
}

fn multi_word_boost(m: &Match, query: &str) -> i32 {
    if !query.contains(' ') {
        return 0;
    }

    let words: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|w| !w.is_empty())
        .map(str::to_ascii_lowercase)
        .collect();
    if words.len() < 2 {
        return 0;
    }

    let path_lower = m.path.to_string_lossy().to_ascii_lowercase();
    // BOM-aware — see `definition_kind_boost`. The BOM would fuse onto the first word, so
    // `﻿pub` did not match the query word `pub` under whole-word splitting below.
    let text_lower = crate::lang::outline::strip_bom(&m.text).to_ascii_lowercase();
    let haystack = format!("{path_lower} {text_lower}");

    // Whole-word matching: split haystack on non-alphanumeric boundaries so
    // short words like "in" or "to" don't match unrelated substrings.
    let haystack_words: Vec<&str> = haystack
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|w| !w.is_empty())
        .collect();
    let matched = words
        .iter()
        .filter(|w| haystack_words.contains(&w.as_str()))
        .count();

    if matched == words.len() {
        300
    } else if matched >= words.len() - 1 {
        120
    } else {
        0
    }
}

/// Penalize non-code files: docs, config examples, generated output.
/// Returns positive value (subtracted from score by caller).
/// Note: `dist/`, `build/` are NOT penalized here — they are already covered by `VENDOR_DIRS`.
fn non_code_penalty(path: &Path) -> i32 {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    // Match on path components to avoid false positives (redoc/, javadoc/, pydoc/)
    let has_docs_component = path.components().any(|c| {
        c.as_os_str()
            .to_str()
            .is_some_and(|s| s == "docs" || s == "doc")
    });

    let is_docs = ext == "md" || ext == "mdx" || ext == "txt" || ext == "rst" || has_docs_component;

    let path_str = path.to_string_lossy();
    let has_example_component = path.components().any(|c| {
        c.as_os_str().to_str().is_some_and(|s| {
            matches!(
                s,
                "example" | "examples" | "sample" | "samples" | "template" | "templates"
            )
        })
    });
    let is_config_example = has_example_component && (ext == "md" || ext == "txt" || ext == "rst");

    let is_generated = path_str.contains("generated");

    let mut penalty = 0;
    if is_docs {
        penalty += 250;
    }
    if is_config_example {
        penalty += 80;
    }
    if is_generated {
        penalty += 150;
    }
    penalty
}

fn looks_like_test_query(query: &str) -> bool {
    let q = query.to_ascii_lowercase();
    q.contains("test") || q.contains("spec") || q.starts_with("should")
}

fn shared_prefix_depth(a: &Path, b: &Path) -> usize {
    a.components()
        .filter(|c| matches!(c, Component::Normal(_)))
        .zip(b.components().filter(|c| matches!(c, Component::Normal(_))))
        .take_while(|(l, r)| l == r)
        .count()
}
/// Check if path contains a vendor directory component.
fn is_vendor_path(path: &Path) -> bool {
    path.components().any(|c| {
        c.as_os_str()
            .to_str()
            .is_some_and(|s| VENDOR_DIRS.contains(&s))
    })
}

/// 0-100, newer = higher. Files modified within the last hour get max score.
fn recency(mtime: SystemTime, now: SystemTime) -> u32 {
    let age = now.duration_since(mtime).unwrap_or_default().as_secs();

    match age {
        0..=3_600 => 100,          // last hour
        3_601..=86_400 => 80,      // last day
        86_401..=604_800 => 50,    // last week
        604_801..=2_592_000 => 20, // last month
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::{score, sort, Scorer};
    use crate::types::Match;
    use std::path::{Path, PathBuf};
    use std::time::SystemTime;

    fn make_match(path: &str, text: &str, is_definition: bool, def_name: Option<&str>) -> Match {
        Match {
            path: PathBuf::from(path),
            line: 1,
            text: text.to_string(),
            is_definition,
            exact: true,
            file_lines: 40,
            mtime: SystemTime::now(),
            def_range: None,
            def_name: def_name.map(ToString::to_string),
            def_weight: if is_definition { 80 } else { 0 },
            impl_target: None,
        }
    }

    /// `sort`'s key must be a **total order**, so the result cannot depend on the order matches
    /// arrived in.
    ///
    /// This is what lets a caller bound retention. `symbol.rs` used to guarantee determinism by
    /// appending each file's matches as one contiguous block, so a stable sort's ties were always
    /// within a file; a bounded retention heap has to drop matches from the middle, which breaks
    /// contiguity, so the key had to stop relying on arrival order.
    ///
    /// The fixture is the case the old comment named as genuinely tied: two overload declarations
    /// on one line, same path, same line, different `def_range` — which `SameSpanDedupe`
    /// deliberately keeps both of. Feeding them in both orders must give the same output. Without
    /// the `def_range` tie-break this fails, because a stable sort preserves whichever order it
    /// was handed.
    #[test]
    fn sort_is_order_independent_for_matches_tied_on_path_and_line() {
        let mk = |span: (u32, u32)| {
            let mut m = make_match("src/a.rs", "fn overload(x: i32);", true, Some("overload"));
            m.def_range = Some(span);
            m
        };
        let (scope, query) = (Path::new("."), "overload");

        // `text` is the last level, so it needs a pair of its own: identical in every other field,
        // differing only in the matched line. Without such a pair, deleting `text` from the key
        // breaks no test and the order silently stops being total again.
        let mk_text = |t: &str| {
            let mut m = make_match("src/a.rs", t, true, Some("overload"));
            m.def_range = Some((2, 6));
            m
        };

        let mut forward = vec![
            mk((1, 4)),
            mk((1, 9)),
            mk_text("fn overload(a: i32);"),
            mk_text("fn overload(b: i32);"),
        ];
        let mut reverse: Vec<Match> = forward.iter().rev().cloned().collect();
        sort(&mut forward, query, scope, None);
        sort(&mut reverse, query, scope, None);

        let key = |v: &[Match]| -> Vec<(Option<(u32, u32)>, String)> {
            v.iter().map(|m| (m.def_range, m.text.clone())).collect()
        };
        assert_eq!(
            key(&forward),
            key(&reverse),
            "sort depends on arrival order, so retention cannot be bounded without changing output"
        );
    }

    /// A UTF-8 BOM on line 1 must not change a match's score.
    ///
    /// This is why #51 is not the cosmetic issue it was filed as. Four ranking terms test the
    /// *start* of `m.text` — `definition_kind_boost` (`starts_with("pub fn ")` and friends),
    /// `exported_api_boost` (`"pub "` / `"export "`), `incidental_text_penalty`
    /// (`starts_with("//")`) and `multi_word_boost` (first whole word) — and all four reach it
    /// through `str::trim_start` or `str::trim`, neither of which removes U+FEFF. So on a
    /// BOM'd file:
    ///
    ///   * a line-1 `pub fn` definition silently loses both its kind boost and its
    ///     exported-API boost, and sorts below the identical line in a file without a BOM;
    ///   * a line-1 `//` comment escapes the incidental-text penalty and sorts *above* where
    ///     it belongs.
    ///
    /// Only line 1 of a BOM'd file is affected — a BOM occurs once, at file start — which is
    /// exactly why it went unnoticed: the same line anywhere else in the file scores right.
    ///
    /// Asserted as equality against the BOM-free spelling rather than against a literal
    /// score, so it keeps holding as the weights change.
    #[test]
    fn a_bom_on_line_one_does_not_change_the_score() {
        let scope = PathBuf::from("/repo/src");
        let now = SystemTime::now();
        // (query, line text, is_definition) — one case per affected ranking term.
        let cases: &[(&str, &str, bool)] = &[
            ("alpha_thing", "pub fn alpha_thing() {", true),
            ("Widget", "pub struct Widget {", true),
            ("export_thing", "export function export_thing() {", true),
            ("handle auth", "pub fn handle_auth() {", true),
            ("thing", "// thing is mentioned here", false),
        ];

        for (query, text, is_def) in cases {
            let plain = make_match("/repo/src/a.rs", text, *is_def, Some("x"));
            // The BOM is built from its bytes, per the #35/#41 convention.
            let bommed_text =
                String::from_utf8([&[0xEF, 0xBB, 0xBF][..], text.as_bytes()].concat()).unwrap();
            let bommed = make_match("/repo/src/a.rs", &bommed_text, *is_def, Some("x"));

            let mut cache = std::collections::HashMap::new();
            let s_plain = score(&plain, query, &scope, None, None, &mut cache, now);
            let mut cache = std::collections::HashMap::new();
            let s_bommed = score(&bommed, query, &scope, None, None, &mut cache, now);

            assert_eq!(
                s_bommed, s_plain,
                "a BOM changed the score for {text:?} (query {query:?}): \
                 {s_bommed} vs {s_plain}"
            );
        }
    }

    /// `sort` scores once per match and applies a hand-rolled index permutation, replacing
    /// a `sort_by` that recomputed `score` inside the comparator. This pins the refactor
    /// against a reference implementation of the *old* shape: same key, same stability,
    /// so the two must agree element for element on any input.
    ///
    /// Worth having because the permutation is applied with cycle swaps rather than by
    /// collecting into a new vector — a wrong inversion there would silently mis-order
    /// results rather than fail loudly, and the ordering is what every caller reads.
    #[test]
    fn score_once_permutation_matches_a_reference_comparator_sort() {
        let scope = PathBuf::from("/repo/src");
        let query = "handleAuth";

        // Deliberately includes exact score ties (same score, differing path/line) and
        // repeated paths, so the tie-break chain and the stability both get exercised.
        let build = || {
            let mut v = Vec::new();
            for (i, dir) in ["a", "b", "c", "zz"].iter().enumerate() {
                for f in 0..7 {
                    let mut m = make_match(
                        &format!("/repo/src/{dir}/f{f}.rs"),
                        "handleAuth(user)",
                        f % 3 == 0,
                        if f % 3 == 0 { Some("handleAuth") } else { None },
                    );
                    m.line = u32::try_from((i * 7 + f) % 5).unwrap() + 1;
                    v.push(m);
                }
            }
            // Same path, identical text (hence identical score), differing lines, pushed
            // out of order — the only case where the *line* tie-break decides anything.
            // Without it the path comparison always resolves first and a flipped line
            // ordering goes undetected. Verified: it did.
            for line in [9u32, 3, 7] {
                let mut m = make_match("/repo/src/tie.rs", "handleAuth(same)", false, None);
                m.line = line;
                v.push(m);
            }
            // Two matches sharing a path *and* line — the case where the key genuinely
            // compares Equal, so only stability decides.
            v.push(make_match(
                "/repo/src/a/f0.rs",
                "handleAuth(x)",
                true,
                Some("handleAuth"),
            ));
            v.push(make_match(
                "/repo/src/a/f0.rs",
                "handleAuth(y)",
                true,
                Some("handleAuth"),
            ));
            v
        };

        // Run the comparison both without and *with* a context path. The context arm is
        // what exercises `context_proximity`, which is the sole reader and writer of
        // `pkg_cache` — and therefore the only part of `score` whose result could
        // conceivably depend on evaluation order, which is exactly what changed here
        // (lazy scoring inside a comparator became one eager pass). Without this arm the
        // guard covers everything except the one thing worth being nervous about.
        for context in [None, Some(Path::new("/repo/src/a/f0.rs"))] {
            check_against_reference(&scope, query, context, &build());
        }
    }

    /// Sort `input` with `sort`, sort a copy with a reference implementation of the old
    /// comparator shape, and require the two to agree element for element.
    fn check_against_reference(scope: &Path, query: &str, context: Option<&Path>, input: &[Match]) {
        use super::score;
        use std::collections::HashMap;

        let clone_of = |v: &[Match]| -> Vec<Match> {
            v.iter()
                .map(|m| {
                    let mut c = make_match(
                        m.path.to_str().unwrap(),
                        &m.text,
                        m.is_definition,
                        m.def_name.as_deref(),
                    );
                    c.line = m.line;
                    c.mtime = m.mtime;
                    c
                })
                .collect()
        };

        let mut actual = clone_of(input);
        sort(&mut actual, query, scope, context);

        // Reference: the pre-refactor shape, scoring inside the comparator.
        let mut expected = clone_of(input);
        let mut pkg_cache: HashMap<PathBuf, Option<PathBuf>> = HashMap::new();
        let now = SystemTime::now();
        let ctx_parent = context.and_then(Path::parent);
        let ctx_pkg_root = context
            .and_then(crate::lang::package_root)
            .map(std::path::Path::to_path_buf);
        expected.sort_by(|a, b| {
            let sa = score(
                a,
                query,
                scope,
                ctx_parent,
                ctx_pkg_root.as_ref(),
                &mut pkg_cache,
                now,
            );
            let sb = score(
                b,
                query,
                scope,
                ctx_parent,
                ctx_pkg_root.as_ref(),
                &mut pkg_cache,
                now,
            );
            sb.cmp(&sa)
                .then_with(|| a.path.cmp(&b.path))
                .then_with(|| a.line.cmp(&b.line))
        });

        let key = |v: &[Match]| {
            v.iter()
                .map(|m| format!("{}:{}:{}", m.path.display(), m.line, m.text))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            key(&actual),
            key(&expected),
            "score-once permutation disagreed with the reference comparator sort (context: {context:?})"
        );
        assert_eq!(
            actual.len(),
            33,
            "fixture size changed — retune the assertion"
        );
    }

    /// `Scorer::selection_score` must not include recency. This is the pin for the whole
    /// bounded-retention design.
    ///
    /// A retention bound gated on the ranking score would let the wall clock decide which
    /// matches exist: `recency` buckets by age at 1h/1d/1w/1mo, so a file crossing a boundary
    /// between two runs changes its score by 20-100 points and can enter or leave the
    /// retained set. That is the defect removed from `overview::hot_files`. Selection
    /// therefore uses a time-independent key, and this asserts it stays that way.
    #[test]
    fn selection_score_ignores_recency_while_the_full_score_does_not() {
        use std::collections::HashMap;
        use std::time::Duration;

        let scope = PathBuf::from("/repo/src");
        let query = "handleAuth";

        let fresh_mtime = SystemTime::now();
        // Two months old: a different `recency` bucket from "within the last hour".
        let stale_mtime = fresh_mtime - Duration::from_hours(1440);

        let mut fresh = make_match("/repo/src/a.rs", "handleAuth(user)", false, None);
        fresh.mtime = fresh_mtime;
        let mut stale = make_match("/repo/src/a.rs", "handleAuth(user)", false, None);
        stale.mtime = stale_mtime;

        let mut scorer = Scorer::new(query, &scope, None);
        assert_eq!(
            scorer.selection_score(&fresh),
            scorer.selection_score(&stale),
            "selection must not depend on mtime, or a bound on it would depend on the clock"
        );

        // And the full score *must* still differ, or the fixture is not exercising recency
        // and the assertion above is vacuous.
        let mut cache: HashMap<PathBuf, Option<PathBuf>> = HashMap::new();
        let now = SystemTime::now();
        let full_fresh = score(&fresh, query, &scope, None, None, &mut cache, now);
        let full_stale = score(&stale, query, &scope, None, None, &mut cache, now);
        assert!(
            full_fresh > full_stale,
            "fixture must straddle a recency bucket ({full_fresh} vs {full_stale})"
        );
    }

    /// Every input element present exactly once after sorting.
    ///
    /// Kept as a cheap tripwire, with an honest note on its strength: `slice::swap` is a
    /// permutation primitive, so no bug expressible in `apply_destination_permutation` can
    /// change the multiset — this cannot catch a misordering that the reference-comparator
    /// test above misses. It guards against a future rewrite that stops using swaps
    /// (indexed assignment, `copy_within`, an `unsafe` variant), where dropping or
    /// duplicating an element becomes possible again.
    #[test]
    fn sort_preserves_the_multiset_of_matches() {
        let scope = PathBuf::from("/repo/src");
        let mut matches: Vec<Match> = (0..64)
            .map(|i| {
                let mut m = make_match(
                    &format!("/repo/src/d{}/f{i}.rs", i % 5),
                    "handleAuth(user)",
                    i % 4 == 0,
                    if i % 4 == 0 { Some("handleAuth") } else { None },
                );
                m.line = i % 9 + 1;
                m
            })
            .collect();

        let mut before: Vec<String> = matches
            .iter()
            .map(|m| format!("{}:{}", m.path.display(), m.line))
            .collect();
        sort(&mut matches, "handleAuth", &scope, None);
        let mut after: Vec<String> = matches
            .iter()
            .map(|m| format!("{}:{}", m.path.display(), m.line))
            .collect();

        assert_eq!(after.len(), 64, "sort changed the element count");
        before.sort();
        after.sort();
        assert_eq!(before, after, "sort lost or duplicated elements");
    }

    /// Degenerate lengths must not panic — the cycle-swap loop indexes `dest` directly.
    #[test]
    fn sort_handles_empty_and_single_element_slices() {
        let scope = PathBuf::from("/repo/src");
        let mut empty: Vec<Match> = Vec::new();
        sort(&mut empty, "q", &scope, None);
        assert!(empty.is_empty());

        let mut one = vec![make_match("/repo/src/a.rs", "q()", false, None)];
        sort(&mut one, "q", &scope, None);
        assert_eq!(one.len(), 1);
    }

    #[test]
    fn prefers_exact_definition_name_over_usage() {
        let scope = PathBuf::from("/repo/src");
        let mut matches = vec![
            make_match("/repo/src/auth.rs", "handleAuth(user)", false, None),
            make_match(
                "/repo/src/auth.rs",
                "pub fn handleAuth(req: Request) -> Response {",
                true,
                Some("handleAuth"),
            ),
        ];

        sort(&mut matches, "handleAuth", &scope, None);

        assert!(matches[0].is_definition);
        assert_eq!(matches[0].def_name.as_deref(), Some("handleAuth"));
    }

    #[test]
    fn prefers_non_test_match_for_non_test_query() {
        let scope = PathBuf::from("/repo/src");
        let mut matches = vec![
            make_match(
                "/repo/src/__tests__/auth.test.ts",
                "export function handleAuth() {",
                true,
                Some("handleAuth"),
            ),
            make_match(
                "/repo/src/auth.ts",
                "export function handleAuth() {",
                true,
                Some("handleAuth"),
            ),
        ];

        sort(&mut matches, "handleAuth", &scope, None);

        assert_eq!(matches[0].path, PathBuf::from("/repo/src/auth.ts"));
    }

    #[test]
    fn prefers_same_subtree_as_context() {
        let scope = PathBuf::from("/repo/src");
        let context = PathBuf::from("/repo/src/auth/controller.rs");
        let mut matches = vec![
            make_match(
                "/repo/src/payments/service.rs",
                "pub fn handleAuth() {",
                true,
                Some("handleAuth"),
            ),
            make_match(
                "/repo/src/auth/service.rs",
                "pub fn handleAuth() {",
                true,
                Some("handleAuth"),
            ),
        ];

        sort(&mut matches, "handleAuth", &scope, Some(&context));

        assert_eq!(matches[0].path, PathBuf::from("/repo/src/auth/service.rs"));
    }

    #[test]
    fn prefers_exported_api_over_local_definition() {
        let scope = PathBuf::from("/repo/src");
        let mut matches = vec![
            make_match(
                "/repo/src/internal/auth.ts",
                "function handleAuth() {",
                true,
                Some("handleAuth"),
            ),
            make_match(
                "/repo/src/public/auth.ts",
                "export function handleAuth() {",
                true,
                Some("handleAuth"),
            ),
        ];

        sort(&mut matches, "handleAuth", &scope, None);

        assert_eq!(matches[0].path, PathBuf::from("/repo/src/public/auth.ts"));
    }

    #[test]
    fn prefers_real_definition_over_fixture_match() {
        let scope = PathBuf::from("/repo/src");
        let mut matches = vec![
            make_match(
                "/repo/src/fixtures/auth-fixture.ts",
                "export function handleAuth() {",
                true,
                Some("handleAuth"),
            ),
            make_match(
                "/repo/src/auth.ts",
                "export function handleAuth() {",
                true,
                Some("handleAuth"),
            ),
        ];

        sort(&mut matches, "handleAuth", &scope, None);

        assert_eq!(matches[0].path, PathBuf::from("/repo/src/auth.ts"));
    }

    #[test]
    fn prefers_thinking_logic_over_schema_for_concept_query() {
        let scope = PathBuf::from("/repo/src");
        let mut matches = vec![
            make_match(
                "/repo/src/internal/interfaces/client_models.go",
                "ThinkingConfig *GenerationConfigThinkingConfig `json:\"thinkingConfig,omitempty\"`",
                false,
                None,
            ),
            make_match(
                "/repo/src/internal/util/thinking.go",
                "func NormalizeThinkingBudget(model string, requested int) int {",
                true,
                Some("NormalizeThinkingBudget"),
            ),
        ];

        sort(&mut matches, "thinking", &scope, None);

        assert!(
            matches[0].path.to_string_lossy().contains("thinking.go"),
            "expected thinking.go first, got {:?}",
            matches[0].path,
        );
    }

    #[test]
    fn prefers_model_mapping_logic_over_docs_for_alias_query() {
        let scope = PathBuf::from("/repo/src");
        let mut matches = vec![
            make_match(
                "/repo/src/docs/FORCE_HANDLER_GUIDE.md",
                "Alias routing example",
                false,
                None,
            ),
            make_match(
                "/repo/src/internal/api/modules/amp/model_mapping.go",
                "func (m *DefaultModelMapper) MapModel(requestedModel string) string {",
                true,
                Some("MapModel"),
            ),
        ];

        sort(&mut matches, "alias", &scope, None);

        assert!(
            matches[0].path.to_string_lossy().contains("model_mapping"),
            "expected model_mapping.go first, got {:?}",
            matches[0].path,
        );
    }

    // --- Unit tests for individual penalty/boost functions ---

    #[test]
    fn non_code_penalty_docs_positive() {
        // Docs get penalized (positive return value, subtracted by caller)
        let path = PathBuf::from("/repo/docs/guide.md");
        assert!(super::non_code_penalty(&path) > 0);
    }

    #[test]
    fn non_code_penalty_no_double_penalty_for_dist() {
        // dist/ should NOT be penalized here — VENDOR_DIRS handles it
        let path = PathBuf::from("/repo/dist/bundle.js");
        assert_eq!(super::non_code_penalty(&path), 0);
    }

    #[test]
    fn non_code_penalty_no_double_penalty_for_build() {
        let path = PathBuf::from("/repo/build/output.js");
        assert_eq!(super::non_code_penalty(&path), 0);
    }

    #[test]
    fn non_code_penalty_generated_without_dist() {
        let path = PathBuf::from("/repo/src/generated/types.ts");
        assert!(super::non_code_penalty(&path) > 0);
    }

    #[test]
    fn non_code_penalty_normal_code_zero() {
        let path = PathBuf::from("/repo/src/auth.rs");
        assert_eq!(super::non_code_penalty(&path), 0);
    }

    #[test]
    fn fixture_penalty_flags_fixture_component() {
        // A path with one or more fixture components returns the flat 90 penalty
        // (there is no longer a 200 cap — the max contributor is a single 90).
        let m = make_match(
            "/repo/src/fixtures/mock_stub_fake.ts",
            "example fixture mock stub fake",
            false,
            None,
        );
        let penalty = super::fixture_penalty(&m);
        assert_eq!(
            penalty, 90,
            "a fixture-component path must return the flat 90 penalty, got {penalty}"
        );
    }

    #[test]
    fn fixture_penalty_zero_for_normal_code() {
        let m = make_match(
            "/repo/src/auth.ts",
            "export function handleAuth() {",
            true,
            Some("handleAuth"),
        );
        assert_eq!(super::fixture_penalty(&m), 0);
    }

    #[test]
    fn fixture_penalty_examples_component_penalized() {
        // `examples/` as a path COMPONENT should be penalized.
        let m = make_match("/repo/examples/demo.rs", "fn main() {}", false, None);
        assert!(super::fixture_penalty(&m) > 0);
    }

    #[test]
    fn fixture_penalty_examples_substring_not_penalized() {
        // `examples_parser.rs` contains "examples" as a substring but NOT as a
        // standalone path component — must NOT be penalized.
        let m = make_match("/repo/src/examples_parser.rs", "fn parse() {}", false, None);
        assert_eq!(super::fixture_penalty(&m), 0);
    }

    #[test]
    fn incidental_text_penalty_comment_line() {
        // Lines starting with // should be penalized
        let m = make_match(
            "/repo/src/lib.rs",
            "// handleAuth is deprecated",
            false,
            None,
        );
        assert_eq!(super::incidental_text_penalty(&m, "handleAuth"), 150);
    }

    #[test]
    fn incidental_text_penalty_no_hash_false_positive() {
        // # in C/Rust files should NOT trigger comment penalty
        let m = make_match("/repo/src/main.c", "#include <stdio.h>", false, None);
        assert_eq!(super::incidental_text_penalty(&m, "stdio"), 0);
    }

    #[test]
    fn incidental_text_penalty_hash_comment_in_python() {
        // # in .py files IS a comment — should be penalized
        let m = make_match(
            "/repo/src/main.py",
            "# handle_auth is deprecated",
            false,
            None,
        );
        assert_eq!(super::incidental_text_penalty(&m, "handle_auth"), 150);
    }

    #[test]
    fn incidental_text_penalty_no_star_false_positive() {
        // * should NOT trigger comment penalty
        let m = make_match("/repo/src/main.c", "*ptr = value;", false, None);
        assert_eq!(super::incidental_text_penalty(&m, "ptr"), 0);
    }

    #[test]
    fn incidental_text_penalty_no_string_literal_heuristic() {
        // String literals should NOT be penalized (fragile heuristic removed)
        let m = make_match(
            "/repo/src/lib.rs",
            r#"let msg = "handleAuth error";"#,
            false,
            None,
        );
        assert_eq!(super::incidental_text_penalty(&m, "handleAuth"), 0);
    }

    #[test]
    fn incidental_text_penalty_trailing_comment() {
        // Query only in trailing comment should be penalized
        let m = make_match(
            "/repo/src/lib.rs",
            "let x = 1; // handleAuth workaround",
            false,
            None,
        );
        assert_eq!(super::incidental_text_penalty(&m, "handleAuth"), 100);
    }

    #[test]
    fn incidental_text_penalty_url_not_comment() {
        // :// is a URL scheme — should NOT be treated as trailing comment
        let m = make_match(
            "/repo/src/lib.rs",
            r#"let url = "https://handleAuth.example.com";"#,
            false,
            None,
        );
        assert_eq!(super::incidental_text_penalty(&m, "handleAuth"), 0);
    }

    #[test]
    fn incidental_text_penalty_skip_definitions() {
        // Definitions should never be penalized
        let m = make_match(
            "/repo/src/lib.rs",
            "// handleAuth docs",
            true,
            Some("handleAuth"),
        );
        assert_eq!(super::incidental_text_penalty(&m, "handleAuth"), 0);
    }

    #[test]
    fn incidental_text_penalty_doc_comment_exempt() {
        // /// doc comments should NOT be penalized — they provide useful symbol context
        let m = make_match(
            "/repo/src/lib.rs",
            "/// Handles auth validation for incoming requests",
            false,
            None,
        );
        assert_eq!(super::incidental_text_penalty(&m, "auth"), 0);
    }

    #[test]
    fn sign_convention_all_penalties_positive() {
        // All penalty functions should return >= 0 (positive values, subtracted by score())
        let doc_path = PathBuf::from("/repo/docs/guide.md");
        assert!(super::non_code_penalty(&doc_path) >= 0);

        let fixture = make_match("/repo/fixtures/mock.ts", "mock data", false, None);
        assert!(super::fixture_penalty(&fixture) >= 0);

        let comment = make_match("/repo/src/lib.rs", "// TODO fix auth", false, None);
        assert!(super::incidental_text_penalty(&comment, "auth") >= 0);
    }

    #[test]
    fn vendor_path_detects_dist_and_build() {
        // dist/ and build/ are in VENDOR_DIRS — this is where the penalty comes from
        assert!(super::is_vendor_path(&PathBuf::from(
            "/repo/dist/bundle.js"
        )));
        assert!(super::is_vendor_path(&PathBuf::from(
            "/repo/build/output.js"
        )));
        assert!(!super::is_vendor_path(&PathBuf::from("/repo/src/auth.rs")));
    }
    #[test]
    fn non_code_penalty_example_substring_not_penalized() {
        // `examples_parser.rs` contains "example" as a substring but NOT as a
        // standalone path component — must not be penalized (mirrors fixture_penalty fix).
        let path = PathBuf::from("/repo/src/examples_parser.rs");
        assert_eq!(super::non_code_penalty(&path), 0);
    }

    #[test]
    fn non_code_penalty_example_component_penalized() {
        // `examples/guide.md` has "examples" as a path component AND a doc ext.
        let path = PathBuf::from("/repo/examples/guide.md");
        assert!(super::non_code_penalty(&path) > 0);
    }
}
