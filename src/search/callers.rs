use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use streaming_iterator::StreamingIterator;

use crate::error::TilthError;
use crate::lang::detect_file_type;
use crate::lang::outline::outline_language;
use crate::types::FileType;

const MAX_MATCHES: usize = 10;
/// Max unique caller functions to trace for 2nd hop. Above this = wide fan-out, skip.
const IMPACT_FANOUT_THRESHOLD: usize = 10;
/// Max 2nd-hop results to display.
const IMPACT_MAX_RESULTS: usize = 15;
/// Match-count cap when `--full` is set. Mirrors the symbol/content search caps.
const FULL_MAX_MATCHES: usize = 100;

// The batch caller walk used to stop once a shared counter crossed a raw-match
// threshold (`BATCH_EARLY_QUIT = 50`). That made results **non-deterministic**: the
// walk is parallel, the counter is only checked at the start of each file callback,
// and each in-flight file can add many matches, so how far the walk got before
// quitting depended on thread scheduling.
//
// Two measurements, both six identical consecutive runs:
//
//   175k-file C++ tree, one hot symbol   53, 69, 55, 52, 53, 53   true count 9581
//   leveldb, `Slice`                     52, 59, 70, 94, 114, 128 true count  203
//
// So it was undercounting as well as varying. Symbols whose true count sat under the
// threshold were perfectly stable, which is exactly the signature of a count cutoff
// and not of file ordering — glob order was never the variable here.
//
// It stayed invisible for a long time because a language whose call sites tilth
// could not resolve never reached 50 matches in the first place. Making C++
// traversal work exposed it.
//
// A varying answer to a fixed question is worse than a slow one for a tool an agent
// reasons about, and no bound can be both count-based and deterministic under a
// parallel walk — so the walk now completes and the caps below apply afterwards, to
// a fully collected and ranked set. Work is still bounded per file by the size gate
// and bloom pre-check in `bloom_walk::read_with_bloom_check`.
//
// The cost is real and worth knowing. On the 175k-file tree above, that query went
// from ~0.1s to ~9.5s, and now returns 9581 every time. That is inside the 90s
// request timeout, and the walk is what makes the answer true, but a hot symbol on a
// very large tree is now a multi-second call — and `search_callers_multi_expanded`
// runs one such walk per target plus a second hop each, so a 5-target query is a
// multiple of it. Peak RSS at that scale is dominated by `BloomFilterCache`, which
// holds one filter per code file walked and is currently unbounded.
//
// Not fixed here: `search::symbol` and `search::content` still gate their parallel
// walks on a shared count (`EARLY_QUIT_THRESHOLD_DEFINITIONS` and friends), so the
// same class of instability remains on the most-used search paths. Removing it there
// costs a full walk on every `tilth_search`, which needs its own measurement.

/// A single caller match — a call site of a target symbol.
///
/// Deliberately holds no file content. It used to carry an `Arc<String>` of the whole
/// file so `expand` would not re-read it, which was free while the walk stopped at ~50
/// matches. Now that the walk completes, that turned into every matched file in the
/// repository staying resident: on a 175k-file C++ tree a single hot-symbol query peaked
/// at 410 MB, of which ~73 MB was this field. At most `expand` matches (≤10) are ever
/// expanded, so the survivors re-read their own file instead.
#[derive(Debug)]
pub struct CallerMatch {
    pub path: PathBuf,
    pub line: u32,
    pub calling_function: String,
    pub call_text: String,
    /// Line range of the calling function (for expand).
    pub caller_range: Option<(u32, u32)>,
}

/// Scan `scope` for the literal `target` byte sequence. Used by the
/// single-symbol `search_callers_expanded` path to distinguish "typo,
/// doesn't exist" from "real symbol with no direct callers" (indirect
/// dispatch, dead code, framework registration, …) when the caller walk
/// returned zero matches. mmap is lazy, so the scan only pages in regions
/// that contain the needle prefix.
fn target_seen_in_scope(target: &str, scope: &Path, glob: Option<&str>) -> bool {
    let Ok(walker) = super::walker(scope, glob) else {
        return false;
    };
    let needle = target.as_bytes();
    let seen = AtomicBool::new(false);

    walker.run(|| {
        let seen = &seen;
        Box::new(move |entry| {
            if seen.load(Ordering::Relaxed) {
                return ignore::WalkState::Quit;
            }
            let Ok(entry) = entry else {
                return ignore::WalkState::Continue;
            };
            if !entry.file_type().is_some_and(|ft| ft.is_file()) {
                return ignore::WalkState::Continue;
            }
            let path = entry.path();
            let Ok(file) = std::fs::File::open(path) else {
                return ignore::WalkState::Continue;
            };
            let Ok(mmap) = (unsafe { memmap2::Mmap::map(&file) }) else {
                return ignore::WalkState::Continue;
            };
            if memchr::memmem::find(&mmap, needle).is_some() {
                seen.store(true, Ordering::Relaxed);
                return ignore::WalkState::Quit;
            }
            ignore::WalkState::Continue
        })
    });

    seen.load(Ordering::Relaxed)
}

/// Find all call sites of any symbol in `targets` across the codebase using a single walk.
/// Returns tuples of (`target_name`, match) so callers know which symbol was matched.
///
/// Walks every candidate file: there is deliberately no match-count cutoff, because a
/// count-based cutoff over a parallel walk yields a different answer on every run. See
/// the note above `find_callers_batch`'s constants. Callers apply their own deterministic
/// caps after ranking.
pub(crate) fn find_callers_batch(
    targets: &HashSet<String>,
    scope: &Path,
    bloom: &crate::index::bloom::BloomFilterCache,
    glob: Option<&str>,
) -> Result<Vec<(String, CallerMatch)>, TilthError> {
    let matches: Mutex<Vec<(String, CallerMatch)>> = Mutex::new(Vec::new());

    let walker = super::walker(scope, glob)?;

    walker.run(|| {
        let matches = &matches;

        Box::new(move |entry| {
            let Ok(entry) = entry else {
                return ignore::WalkState::Continue;
            };

            if !entry.file_type().is_some_and(|ft| ft.is_file()) {
                return ignore::WalkState::Continue;
            }

            let path = entry.path();

            // Read + size-gate + bloom prefilter in one shared step.
            let Some((content, _mtime)) = super::bloom_walk::read_with_bloom_check(
                path,
                targets,
                bloom,
                super::bloom_walk::MAX_FILE_SIZE,
            ) else {
                return ignore::WalkState::Continue;
            };

            // Fast byte check via memchr::memmem (SIMD) — cheap second pass that
            // eliminates bloom false positives before tree-sitter parses.
            if !targets
                .iter()
                .any(|t| memchr::memmem::find(content.as_bytes(), t.as_bytes()).is_some())
            {
                return ignore::WalkState::Continue;
            }

            // Only process files with tree-sitter grammars
            let file_type = detect_file_type(path);
            let FileType::Code(lang) = file_type else {
                return ignore::WalkState::Continue;
            };

            let Some(ts_lang) = outline_language(lang) else {
                return ignore::WalkState::Continue;
            };

            let file_callers =
                find_callers_treesitter_batch(path, targets, &ts_lang, &content, lang);

            if !file_callers.is_empty() {
                let mut all = matches
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                all.extend(file_callers);
            }

            ignore::WalkState::Continue
        })
    });

    Ok(matches
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner))
}

/// Tree-sitter call site detection for a set of target symbols.
/// Returns tuples of (`matched_target_name`, `CallerMatch`).
fn find_callers_treesitter_batch(
    path: &Path,
    targets: &HashSet<String>,
    ts_lang: &tree_sitter::Language,
    content: &str,
    lang: crate::types::Lang,
) -> Vec<(String, CallerMatch)> {
    // Get the query string for this language
    let Some(query_str) = super::callee_query::callee_query_str(lang) else {
        return Vec::new();
    };

    let mut parser = tree_sitter::Parser::new();
    if parser.set_language(ts_lang).is_err() {
        return Vec::new();
    }

    let Some(tree) = parser.parse(content, None) else {
        return Vec::new();
    };

    let content_bytes = content.as_bytes();
    let lines: Vec<&str> = content.lines().collect();

    let Some(callers) = super::callee_query::with_callee_query(ts_lang, query_str, |query| {
        let Some(callee_idx) = query.capture_index_for_name("callee") else {
            return Vec::new();
        };

        let mut cursor = tree_sitter::QueryCursor::new();
        let mut matches = cursor.matches(query, tree.root_node(), content_bytes);
        let mut callers = Vec::new();

        while let Some(m) = matches.next() {
            for cap in m.captures {
                if cap.index != callee_idx {
                    continue;
                }

                // Check if the captured text matches any of our target symbols
                let Ok(text) = cap.node.utf8_text(content_bytes) else {
                    continue;
                };

                if !targets.contains(text) {
                    continue;
                }

                let matched_target = text.to_string();

                // Found a call site! Now walk up to find the calling function
                let line = cap.node.start_position().row as u32 + 1;

                // Get the call text (the whole call expression, not just the callee)
                let call_node = cap.node.parent().unwrap_or(cap.node);
                let same_line = call_node.start_position().row == call_node.end_position().row;
                let call_text: String = if same_line {
                    let row = call_node.start_position().row;
                    if row < lines.len() {
                        lines[row].trim().to_string()
                    } else {
                        matched_target.clone()
                    }
                } else {
                    matched_target.clone()
                };

                // Walk up the tree to find the enclosing function
                let (calling_function, caller_range) =
                    find_enclosing_function(cap.node, &lines, lang);

                callers.push((
                    matched_target,
                    CallerMatch {
                        path: path.to_path_buf(),
                        line,
                        calling_function,
                        call_text,
                        caller_range,
                    },
                ));
            }
        }

        callers
    }) else {
        return Vec::new();
    };

    callers
}

/// Walk up the AST from a node to find the enclosing function definition.
/// Returns (`function_name`, `line_range`). Top-level renders as `"<top-level>"`.
fn find_enclosing_function(
    node: tree_sitter::Node,
    lines: &[&str],
    lang: crate::types::Lang,
) -> (String, Option<(u32, u32)>) {
    match super::scope::walk_to_enclosing_definition(node, lines, lang) {
        Some((_, name, range)) => (name, Some(range)),
        None => ("<top-level>".to_string(), None),
    }
}

/// Format and rank caller search results with optional expand.
pub fn search_callers_expanded(
    target: &str,
    scope: &Path,
    bloom: &crate::index::bloom::BloomFilterCache,
    expand: usize,
    context: Option<&Path>,
    glob: Option<&str>,
    full: bool,
) -> Result<String, TilthError> {
    let max_matches = if full { FULL_MAX_MATCHES } else { MAX_MATCHES };
    let single: HashSet<String> = std::iter::once(target.to_string()).collect();
    let raw = find_callers_batch(&single, scope, bloom, glob)?;
    let callers: Vec<CallerMatch> = raw.into_iter().map(|(_, m)| m).collect();

    if callers.is_empty() {
        let target_seen = target_seen_in_scope(target, scope, glob);
        return Ok(no_callers_message(target, scope, target_seen, glob));
    }

    // Sort by relevance (context file first, then by proximity)
    let mut sorted_callers = callers;
    rank_callers(&mut sorted_callers, scope, context);

    let total = sorted_callers.len();

    // Collect unique caller names BEFORE truncation for accurate fan-out threshold
    let all_caller_names: HashSet<String> = sorted_callers
        .iter()
        .filter(|c| c.calling_function != "<top-level>")
        .map(|c| c.calling_function.clone())
        .collect();

    sorted_callers.truncate(max_matches);

    let mut output = String::new();
    write_caller_bucket(&mut output, target, scope, total, &sorted_callers, expand);
    write_second_hop_impact(
        &mut output,
        &all_caller_names,
        &sorted_callers,
        scope,
        bloom,
        glob,
    );

    let tokens = crate::types::estimate_tokens(output.len() as u64);
    let _ = write!(
        output,
        "\n\n({} tokens)",
        crate::search::format_token_count(tokens)
    );
    Ok(output)
}

/// Render one target's caller bucket in the canonical shape shared by both
/// the single-target and multi-target callers search: a
/// `# Callers of "<target>" in <scope> — N call site(s)` header, then one
/// `## <path>:<line> [caller: <fn>]` block per call site (with an optional
/// expanded source excerpt). Multi-target search calls this once per target
/// so a bucket inside a comma query renders byte-identically to what a lone
/// single-target search of the same symbol, scope, and hits would produce.
fn write_caller_bucket(
    output: &mut String,
    target: &str,
    scope: &Path,
    total: usize,
    sorted_callers: &[CallerMatch],
    expand: usize,
) {
    let _ = writeln!(
        output,
        "# Callers of \"{}\" in {} — {} call site{}",
        target,
        scope.display(),
        total,
        if total == 1 { "" } else { "s" }
    );

    for (i, caller) in sorted_callers.iter().enumerate() {
        // Header: file:line [caller: calling_function]
        let _ = write!(
            output,
            "\n## {}:{} [caller: {}]\n",
            caller
                .path
                .strip_prefix(scope)
                .unwrap_or(&caller.path)
                .display(),
            caller.line,
            caller.calling_function
        );

        // Show the call text
        let _ = writeln!(output, "-> {}", caller.call_text);

        // Expand if requested and we have the range
        if i < expand {
            if let Some((start, end)) = caller.caller_range {
                // Read on demand: only the first `expand` matches are ever expanded, so
                // retaining every matched file's content through the walk is not worth
                // the memory it costs on a large tree. See `CallerMatch`.
                let Ok(file_content) = std::fs::read_to_string(&caller.path) else {
                    continue;
                };
                let lines: Vec<&str> = file_content.lines().collect();
                // `caller_range` came from the content read during the walk, which on a
                // large tree can be seconds earlier. If the file shrank in between — a
                // formatter, a code generator, a branch switch, an editor save — the
                // range no longer fits, and clamping only `end` leaves `start > end`,
                // which panics on the slice below. Skip the block instead: the header
                // and call text are already written, so nothing else is lost.
                let start_idx = (start as usize).saturating_sub(1);
                let end_idx = (end as usize).min(lines.len());
                if start_idx >= end_idx {
                    continue;
                }

                output.push('\n');
                output.push_str("```\n");

                for (idx, line) in lines[start_idx..end_idx].iter().enumerate() {
                    let line_num = start_idx + idx + 1;
                    let prefix = if line_num == caller.line as usize {
                        "> "
                    } else {
                        "  "
                    };
                    let _ = writeln!(output, "{prefix}{line_num:4} | {line}");
                }

                output.push_str("```\n");
            }
        }
    }
}

/// Adaptive 2nd-hop impact analysis, shared by single- and multi-target
/// callers search (extracted so multi-target reuses this exact block per
/// target bucket instead of re-implementing it — PR #138 review HIGH
/// finding: the multi-target path originally omitted this entirely).
///
/// `all_caller_names` must be the target's unique direct-caller names
/// collected BEFORE `sorted_callers` truncation, so the fan-out threshold
/// check reflects the true hop-1 breadth rather than the display-capped one.
fn write_second_hop_impact(
    output: &mut String,
    all_caller_names: &HashSet<String>,
    sorted_callers: &[CallerMatch],
    scope: &Path,
    bloom: &crate::index::bloom::BloomFilterCache,
    glob: Option<&str>,
) {
    if all_caller_names.is_empty() || all_caller_names.len() > IMPACT_FANOUT_THRESHOLD {
        return;
    }
    let Ok(hop2) = find_callers_batch(all_caller_names, scope, bloom, glob) else {
        return;
    };

    // Filter out hop-1 matches (same file+line = same call site)
    let hop1_locations: HashSet<(PathBuf, u32)> = sorted_callers
        .iter()
        .map(|c| (c.path.clone(), c.line))
        .collect();

    let mut hop2_filtered: Vec<_> = hop2
        .into_iter()
        .filter(|(_, m)| !hop1_locations.contains(&(m.path.clone(), m.line)))
        .collect();

    if hop2_filtered.is_empty() {
        return;
    }

    // Total order before the dedup-and-cap loop below. `find_callers_batch` returns
    // matches in walk order, which is thread-scheduling order — so without this the
    // `IMPACT_MAX_RESULTS` cap kept a different 15 of them on each run, and the dedup
    // kept a different representative per (function, file). Removing the walk's early
    // quit made the *total* stable; it did nothing for this rendering, and an unranked
    // truncation of an unordered vector is the same class of bug.
    hop2_filtered
        .sort_by(|(via_a, a), (via_b, b)| (&a.path, a.line, via_a).cmp(&(&b.path, b.line, via_b)));

    output.push_str("\n-- impact (2nd hop) --\n");

    let mut seen: HashSet<(String, PathBuf)> = HashSet::new();
    let mut count = 0;
    for (via, m) in &hop2_filtered {
        let key = (m.calling_function.clone(), m.path.clone());
        if !seen.insert(key) {
            continue;
        }
        if count >= IMPACT_MAX_RESULTS {
            break;
        }

        let rel_path = m.path.strip_prefix(scope).unwrap_or(&m.path).display();
        let _ = writeln!(
            output,
            "  {:<20} {}:{}  -> {}",
            m.calling_function, rel_path, m.line, via
        );
        count += 1;
    }

    let unique_total = hop2_filtered
        .iter()
        .map(|(_, m)| (&m.calling_function, &m.path))
        .collect::<HashSet<_>>()
        .len();
    if unique_total > IMPACT_MAX_RESULTS {
        let _ = writeln!(
            output,
            "  ... and {} more",
            unique_total - IMPACT_MAX_RESULTS
        );
    }

    // Affected-count formula matches main's evolution: pre-truncation
    // hop-1 caller-name count + deduplicated hop-2 unique_total (NOT the
    // post-truncation display `count`, which under-reports once the
    // `IMPACT_MAX_RESULTS` cap above stops incrementing it while more
    // unique callers still exist beyond the cap).
    let _ = writeln!(
        output,
        "\n{} functions affected across 2 hops.",
        all_caller_names.len() + unique_total
    );
}

/// Multi-target caller search: find call sites of 2..=5 symbols in a single
/// walk via `find_callers_batch`, then render one labeled section per target.
/// Mirrors `search_multi_symbol_expanded` for the `kind=callers` comma path.
///
/// Each target's bucket renders via the same `write_caller_bucket` +
/// `write_second_hop_impact` helpers the single-target path uses, so a
/// bucket here is byte-identical to what a lone `search_callers_expanded`
/// call for that target would produce (PR #138 review: HIGH — 2nd-hop parity;
/// MED — header shape parity).
///
/// The walk-wide early-quit budget this used to scale by target count is gone: it was
/// the source of the non-determinism described above `find_callers_batch`'s constants,
/// and starvation of a later target by a hit-rich earlier one is not possible once
/// every candidate file is visited.
///
/// Cost note: `write_second_hop_impact` runs inside the per-target loop, so a 5-target
/// query is one primary walk plus up to five second-hop walks, each now a full
/// traversal. On a very large tree that is the multiple of the single-walk cost quoted
/// above — bounded by the request timeout, not by a match count.
pub fn search_callers_multi_expanded(
    targets: &[&str],
    scope: &Path,
    bloom: &crate::index::bloom::BloomFilterCache,
    expand: usize,
    context: Option<&Path>,
    glob: Option<&str>,
    full: bool,
) -> Result<String, TilthError> {
    let max_matches = if full { FULL_MAX_MATCHES } else { MAX_MATCHES };

    // Dedupe targets, preserving first-seen order: a repeated target (e.g.
    // query "foo,foo") must not render an empty no-callers section on its
    // second occurrence after the first consumed the matched bucket. The
    // deduped list also feeds the batch search, so the input is deduped once.
    let mut seen: HashSet<&str> = HashSet::new();
    let ordered: Vec<&str> = targets
        .iter()
        .copied()
        .filter(|t| seen.insert(*t))
        .collect();

    let target_set: HashSet<String> = ordered.iter().map(ToString::to_string).collect();
    let raw = find_callers_batch(&target_set, scope, bloom, glob)?;

    // Bucket matches by which target they call. Preserve the caller-supplied
    // target order so output is deterministic.
    let mut by_target: std::collections::HashMap<String, Vec<CallerMatch>> =
        std::collections::HashMap::new();
    for (name, m) in raw {
        by_target.entry(name).or_default().push(m);
    }

    let mut output = String::new();
    for target in &ordered {
        let mut callers = by_target.remove(*target).unwrap_or_default();

        if callers.is_empty() {
            let target_seen = target_seen_in_scope(target, scope, glob);
            output.push_str(&no_callers_message(target, scope, target_seen, glob));
            output.push_str("\n\n");
            continue;
        }

        rank_callers(&mut callers, scope, context);
        let total = callers.len();

        // Unique direct-caller names BEFORE truncation, same as the
        // single-target path — feeds the 2nd-hop fan-out threshold check
        // with the true hop-1 breadth rather than the display-capped one.
        let all_caller_names: HashSet<String> = callers
            .iter()
            .filter(|c| c.calling_function != "<top-level>")
            .map(|c| c.calling_function.clone())
            .collect();

        callers.truncate(max_matches);

        write_caller_bucket(&mut output, target, scope, total, &callers, expand);
        write_second_hop_impact(&mut output, &all_caller_names, &callers, scope, bloom, glob);
        output.push('\n');
    }

    let tokens = crate::types::estimate_tokens(output.len() as u64);
    // Single leading '\n' (not single-target's '\n\n'): the per-bucket loop
    // above already ends each target's section with its own `output.push('\n')`,
    // so a second blank line here would double up before the token count.
    let _ = write!(
        output,
        "\n({} tokens)",
        crate::search::format_token_count(tokens)
    );
    Ok(output)
}

/// Build the user-facing message when callers search returns no hits.
/// Splits two cases that mean very different things to an agent:
/// `target_seen = true` means the symbol exists somewhere but has no direct
/// call sites — probable indirect dispatch, so we show a richer hint
/// listing the common indirection mechanisms. `target_seen = false` means
/// the literal name never appears in scope — most often a typo or wrong
/// scope, so we suppress the indirect-dispatch hint to avoid misleading
/// the agent.
fn no_callers_message(target: &str, scope: &Path, target_seen: bool, glob: Option<&str>) -> String {
    if !target_seen {
        return format!(
            "# Callers of \"{target}\" in {scope_disp} — no call sites found\n\n\
             The name \"{target}\" does not appear anywhere in scope. \
             Check the spelling, or widen scope if you expected hits outside this directory.",
            scope_disp = scope.display()
        );
    }
    // Only mention glob-driven test exclusion when a glob was actually used.
    // Otherwise the line implies a filter that the caller didn't apply, which
    // would mislead an agent reasoning about what tilth searched.
    let glob_hint = if glob.is_some() {
        "\n  • test files (if `glob` excluded them)"
    } else {
        ""
    };
    format!(
        "# Callers of \"{target}\" in {scope_disp} — no direct call sites found\n\n\
         \"{target}\" appears in the codebase but has no syntactic call sites. \
         tilth detects only direct, by-name calls; this symbol may still be reachable via:\n\
         \n  • interface / trait dispatch (Rust `dyn Trait`, Go interface, Java/Kotlin abstract method)\
         \n  • reflection or dynamic dispatch (`getattr`, `Method::invoke`, `eval`)\
         \n  • framework registration (HTTP routes, JSON-RPC, plugin systems, decorators)\
         \n  • function values stored in maps, structs, or passed as callbacks{glob_hint}\n\
         \nVerify with `tilth_search \"{target}\"` to see how it's referenced before assuming dead code.",
        scope_disp = scope.display()
    )
}

/// Simple ranking: context file first, then by path length (proximity heuristic).
fn rank_callers(callers: &mut [CallerMatch], scope: &Path, context: Option<&Path>) {
    callers.sort_by(|a, b| {
        // Context file wins
        if let Some(ctx) = context {
            match (a.path == ctx, b.path == ctx) {
                (true, false) => return std::cmp::Ordering::Less,
                (false, true) => return std::cmp::Ordering::Greater,
                _ => {}
            }
        }

        // Shorter paths (more similar to scope) rank higher
        let a_rel = a.path.strip_prefix(scope).unwrap_or(&a.path);
        let b_rel = b.path.strip_prefix(scope).unwrap_or(&b.path);
        a_rel
            .components()
            .count()
            .cmp(&b_rel.components().count())
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.line.cmp(&b.line))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_callers_message_for_unseen_symbol_says_typo_or_scope() {
        let msg = no_callers_message("doesNotExist", Path::new("/repo"), false, None);
        assert!(msg.contains("does not appear anywhere in scope"));
        assert!(msg.contains("Check the spelling"));
        // Must NOT include the indirect-dispatch hint — that would mislead.
        assert!(!msg.contains("interface"));
        assert!(!msg.contains("reflection"));
    }

    #[test]
    fn no_callers_message_for_seen_symbol_lists_indirection_modes() {
        let msg = no_callers_message("Foo", Path::new("/repo"), true, None);
        assert!(msg.contains("appears in the codebase"));
        assert!(msg.contains("interface"));
        assert!(msg.contains("reflection"));
        assert!(msg.contains("framework registration"));
        assert!(msg.contains("Verify with `tilth_search"));
        // Must NOT pretend the symbol is missing — different signal than typo case.
        assert!(!msg.contains("does not appear"));
    }

    /// The "test files (if glob excluded them)" hint is only meaningful when
    /// the caller actually used a glob. Without a glob it would mislead an
    /// agent into thinking tilth filtered something it did not.
    #[test]
    fn no_callers_message_omits_glob_hint_when_no_glob() {
        let msg = no_callers_message("Foo", Path::new("/repo"), true, None);
        assert!(
            !msg.contains("test files"),
            "glob-driven hint must not appear when glob is None: {msg}"
        );
    }

    #[test]
    fn no_callers_message_includes_glob_hint_when_glob_set() {
        let msg = no_callers_message("Foo", Path::new("/repo"), true, Some("*.rs"));
        assert!(
            msg.contains("test files"),
            "glob-driven hint should appear when glob is Some: {msg}"
        );
    }
    /// A qualified static call — `Class::Func()` — is a `call_expression` whose
    /// function is a `qualified_identifier`, not the `field_expression` an
    /// `obj.Func()` member call produces. Only the member-call pattern was in the
    /// C/C++ callee query, so every static helper and every `Super::` call site
    /// reported "no direct call sites found" while symbol search plainly showed one.
    /// The member call is asserted alongside it to pin that adding the C++-only
    /// patterns did not cost the case that already worked.
    #[test]
    fn cpp_finds_qualified_static_and_member_call_sites() {
        let dir = tempfile::tempdir().unwrap();
        let bloom = crate::index::bloom::BloomFilterCache::new();
        std::fs::write(
            dir.path().join("Probe.h"),
            "class Holder\n{\npublic:\n\tstatic void StaticWork();\n\tvoid MemberWork();\n};\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("Probe.cpp"),
            "#include \"Probe.h\"\n\
             void Holder::StaticWork() {}\n\
             void Holder::MemberWork() {}\n\
             void CallEverything()\n\
             {\n\
             \tHolder::StaticWork();\n\
             \tHolder H;\n\
             \tH.MemberWork();\n\
             }\n",
        )
        .unwrap();

        for target in ["StaticWork", "MemberWork"] {
            let out =
                search_callers_expanded(target, dir.path(), &bloom, 0, None, None, false).unwrap();
            assert!(
                out.contains("1 call site"),
                "expected exactly one call site for {target}, got:\n{out}"
            );
            assert!(
                out.contains("[caller: CallEverything]"),
                "call site for {target} should be attributed to CallEverything \
                 (the enclosing function had to resolve by name too), got:\n{out}"
            );
        }
    }

    /// Files in the fixture below. A count-based cutoff quits the walk *between* files,
    /// so what defeats it is a fixture with far more files than the cutoff admits — not
    /// merely more matches. A first version of these tests used 12 files holding 60
    /// matches: past the old threshold on matches, but the walker's threads consumed all
    /// 12 before the shared counter was ever read, so the exact-total assertion stayed
    /// green with the bug reintroduced (verified 12/12 runs). 400 files is comfortably
    /// past the point where that can happen, and still writes in well under a second.
    const DETERMINISM_FIXTURE_FILES: usize = 400;
    const DETERMINISM_FIXTURE_CALLS_PER_FILE: usize = 2;

    /// Write `DETERMINISM_FIXTURE_FILES` files each calling `target_fn`, and return the
    /// total number of call sites in the tree.
    fn write_determinism_fixture(dir: &Path) -> usize {
        for f in 0..DETERMINISM_FIXTURE_FILES {
            let mut src = String::from("fn target_fn() {}\n");
            for i in 0..DETERMINISM_FIXTURE_CALLS_PER_FILE {
                src.push_str(&format!("fn caller_{f}_{i}() {{ target_fn(); }}\n"));
            }
            std::fs::write(dir.join(format!("m{f}.rs")), src).unwrap();
        }
        DETERMINISM_FIXTURE_FILES * DETERMINISM_FIXTURE_CALLS_PER_FILE
    }

    /// The caller walk must report *every* call site, not however many a shared counter
    /// happened to admit before a parallel walk noticed it had crossed a threshold.
    ///
    /// The old `BATCH_EARLY_QUIT = 50` cutoff made this non-deterministic. Six identical
    /// runs against a 175k-file C++ tree returned 53, 69, 55, 52, 53 and 53 sites for one
    /// hot symbol whose true count is 9581; six runs over leveldb returned 52, 59, 70, 94,
    /// 114 and 128 for `Slice`, true count 203. Both unstable and undercounting. Symbols
    /// whose true count sat under the threshold were perfectly stable, which is what made
    /// the bug easy to miss for so long.
    ///
    /// The assertion is on an exact total, so a reintroduced cutoff fails outright rather
    /// than merely varying.
    #[test]
    fn callers_reports_every_call_site_past_the_old_early_quit_threshold() {
        let dir = tempfile::tempdir().unwrap();
        let bloom = crate::index::bloom::BloomFilterCache::new();
        let expected = write_determinism_fixture(dir.path());

        let targets: HashSet<String> = std::iter::once("target_fn".to_string()).collect();
        let found = find_callers_batch(&targets, dir.path(), &bloom, None).unwrap();
        assert_eq!(
            found.len(),
            expected,
            "expected every call site, got {}",
            found.len()
        );

        // And the rendered total must agree, since that number is what an agent reads.
        let out =
            search_callers_expanded("target_fn", dir.path(), &bloom, 0, None, None, false).unwrap();
        assert!(
            out.contains(&format!("{expected} call sites")),
            "header must report the true total: {}",
            out.lines().next().unwrap_or_default()
        );
    }

    /// Repeated identical runs must agree. Weaker than the exact-total assertion above but
    /// it fails for *any* source of instability, not just a count cutoff.
    #[test]
    fn callers_result_is_stable_across_repeated_runs() {
        let dir = tempfile::tempdir().unwrap();
        let bloom = crate::index::bloom::BloomFilterCache::new();
        write_determinism_fixture(dir.path());
        let targets: HashSet<String> = std::iter::once("target_fn".to_string()).collect();

        let counts: Vec<usize> = (0..5)
            .map(|_| {
                find_callers_batch(&targets, dir.path(), &bloom, None)
                    .unwrap()
                    .len()
            })
            .collect();
        assert!(
            counts.windows(2).all(|w| w[0] == w[1]),
            "caller counts must not vary run to run, got {counts:?}"
        );
    }

    /// The *rendered* second-hop block must be stable too, not just the walk's total.
    ///
    /// `write_second_hop_impact` dedups and caps at `IMPACT_MAX_RESULTS`, and it used to do
    /// that over `find_callers_batch`'s raw vector — which is in walk order, i.e. thread
    /// order. With more than 15 unique hop-2 callers, identical runs rendered a different
    /// 15 of them (measured: 5 distinct renderings in 6 runs). Fixing the walk's total did
    /// not fix this; an unranked truncation of an unordered vector is the same bug.
    #[test]
    fn second_hop_impact_block_is_stable_across_repeated_runs() {
        let dir = tempfile::tempdir().unwrap();
        let bloom = crate::index::bloom::BloomFilterCache::new();
        // 2 hop-1 callers keeps us under IMPACT_FANOUT_THRESHOLD; 40 hop-2 callers is well
        // past IMPACT_MAX_RESULTS, so the cap has to choose.
        std::fs::write(
            dir.path().join("target.rs"),
            "fn target_fn() {}\nfn hop1_a() { target_fn(); }\nfn hop1_b() { target_fn(); }\n",
        )
        .unwrap();
        for i in 0..40 {
            std::fs::write(
                dir.path().join(format!("h{i}.rs")),
                format!("fn hop2_{i}() {{ hop1_a(); }}\n"),
            )
            .unwrap();
        }

        let renders: Vec<String> = (0..6)
            .map(|_| {
                let out =
                    search_callers_expanded("target_fn", dir.path(), &bloom, 0, None, None, false)
                        .unwrap();
                out.split("-- impact (2nd hop) --")
                    .nth(1)
                    .unwrap_or_default()
                    .to_string()
            })
            .collect();
        assert!(
            !renders[0].trim().is_empty(),
            "fixture must produce a 2nd-hop block"
        );
        assert!(
            renders.windows(2).all(|w| w[0] == w[1]),
            "2nd-hop block must not vary run to run:\n{renders:#?}"
        );
    }

    /// `caller_range` is computed from the content read during the walk; expansion re-reads
    /// the file, which on a large tree is seconds later. A file that shrank in between
    /// leaves `start > end` after `end` alone is clamped, which panicked on the slice.
    #[test]
    fn expand_does_not_panic_when_the_file_shrank_after_the_walk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.rs");
        let mut src = String::from("fn target_fn() {}\n");
        for i in 0..30 {
            src.push_str(&format!("fn pad_{i}() {{ let _ = {i}; }}\n"));
        }
        src.push_str("fn late_caller() {\n    target_fn();\n}\n");
        std::fs::write(&path, &src).unwrap();

        let bloom = crate::index::bloom::BloomFilterCache::new();
        let targets: HashSet<String> = std::iter::once("target_fn".to_string()).collect();
        let mut raw = find_callers_batch(&targets, dir.path(), &bloom, None).unwrap();
        assert!(!raw.is_empty(), "fixture must produce a call site");
        let callers: Vec<CallerMatch> = raw.drain(..).map(|(_, m)| m).collect();

        // Truncate the file so every recorded range is now out of bounds.
        std::fs::write(&path, "fn target_fn() {}\n").unwrap();

        let mut out = String::new();
        write_caller_bucket(
            &mut out,
            "target_fn",
            dir.path(),
            callers.len(),
            &callers,
            5,
        );
        assert!(
            out.contains("target_fn"),
            "header and call text must still render: {out}"
        );
    }

    /// Real-code check on C++ qualified-static caller detection.
    ///
    /// `cpp_finds_qualified_static_and_member_call_sites` above is the CI gate — it is
    /// synthetic, always runs, and does fail if the `qualified_identifier` pattern is
    /// removed. This adds the same assurance against a real codebase, where shapes a
    /// hand-written fixture omits actually occur. Upstream v0.9.0 reports 3 call sites
    /// for `Corruption` in leveldb; with qualified statics detected it reports 33 — the
    /// 30 `Status::Corruption` calls plus 3 `Reporter::Corruption` — so a regression
    /// that drops the qualified ones collapses the number back toward 3.
    ///
    /// Be clear about its reach: this **does not run in CI**, which never clones the
    /// fixture, and a silent skip is indistinguishable from a pass in the default test
    /// output. Treat it as a local pre-release check, not protection. It is also
    /// deliberately not the benchmark's job — the `leveldb_corruption_callers` task
    /// covers the same ground but the agent picks its own route, taking the callers path
    /// in roughly 5 runs of 8 even when tilth is its only tool, which makes it useless
    /// as a gate.
    ///
    /// Populate the fixture with
    /// `python benchmark/fixtures/setup_repos.py --repos leveldb`, or point
    /// `TILTH_BENCH_REPOS` at an existing clone's parent directory.
    #[test]
    fn cpp_qualified_static_callers_on_real_fixture() {
        let Some(repo) = leveldb_fixture() else {
            eprintln!(
                "skipping cpp_qualified_static_callers_on_real_fixture: leveldb fixture \
                 not found (see the test's doc comment)"
            );
            return;
        };
        let bloom = crate::index::bloom::BloomFilterCache::new();
        // `full = false`: the path agents actually take. The header count is `total`,
        // computed before the display cap, so it reads 33 either way — but there is no
        // reason to exercise the rarer branch.
        let out = search_callers_expanded("Corruption", &repo, &bloom, 0, None, None, false)
            .expect("callers search on the leveldb fixture");

        let header = out.lines().next().unwrap_or_default();
        let count: usize = header
            .rsplit("— ")
            .next()
            .and_then(|s| s.split_whitespace().next())
            .and_then(|n| n.parse().ok())
            .unwrap_or_else(|| panic!("could not read a count from header: {header:?}"));

        assert_eq!(
            count, 33,
            "expected 33 call sites at the pinned leveldb commit (30 Status::Corruption \
             + 3 Reporter::Corruption); a value near 3 means qualified statics stopped \
             being detected. Header: {header}"
        );
        // Spot-check that the qualified form specifically is present, not just the count.
        assert!(
            out.contains("Status::Corruption("),
            "the qualified call form must appear in the results"
        );
    }

    /// Path to the leveldb benchmark fixture, if it has been cloned.
    ///
    /// `benchmark/config.py` sets `REPOS_DIR = /tmp/tilth_bench/repos`. On Windows that
    /// literal resolves against the *process's current drive*, so a checkout on `D:`
    /// clones to `D:\tmp\…` and neither hardcoded spelling finds it — hence the
    /// `TILTH_BENCH_REPOS` override, which takes the repos directory.
    ///
    /// Identified by a file it must contain rather than by the directory existing, so a
    /// half-finished clone skips instead of failing confusingly.
    fn leveldb_fixture() -> Option<PathBuf> {
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Ok(root) = std::env::var("TILTH_BENCH_REPOS") {
            candidates.push(PathBuf::from(root).join("leveldb"));
        }
        candidates.push(PathBuf::from(r"C:\tmp\tilth_bench\repos\leveldb"));
        candidates.push(PathBuf::from("/tmp/tilth_bench/repos/leveldb"));
        candidates
            .into_iter()
            .find(|c| c.join("include/leveldb/status.h").is_file())
    }

    /// Regression test: when there are more than MAX_MATCHES (10) hop-1 call
    /// sites but still <= IMPACT_FANOUT_THRESHOLD unique callers, the footer
    /// "N functions affected across 2 hops" must use the pre-truncation unique
    /// count, not the post-truncation count.
    ///
    /// Setup: 8 unique functions, each calling `target_fn` twice = 16 call
    /// sites. Truncation to MAX_MATCHES=10 only keeps the first ~5 functions,
    /// dropping functions 6-8. The old code rebuilt the hop-1 set from
    /// sorted_callers AFTER truncation and undercounted. The fix uses
    /// all_caller_names (pre-truncation) which always holds 8.
    #[test]
    fn footer_count_uses_pre_truncation_caller_set() {
        let dir = tempfile::tempdir().unwrap();
        let bloom = crate::index::bloom::BloomFilterCache::new();

        // 8 files: each declares one function that calls `target_fn` twice.
        // Total: 16 call sites from 8 unique caller names.
        // One hop-2 file calls caller_a_0 so the 2nd-hop block fires.
        for i in 0..8usize {
            let content = format!(
                "fn target_fn() {{}}\
                \nfn caller_a_{i}() {{ target_fn(); target_fn(); }}\
                \n"
            );
            std::fs::write(dir.path().join(format!("f{i}.rs")), content).unwrap();
        }
        std::fs::write(
            dir.path().join("hop2.rs"),
            "fn hop2_fn() { caller_a_0(); }\n",
        )
        .unwrap();

        let result =
            search_callers_expanded("target_fn", dir.path(), &bloom, 0, None, None, false).unwrap();

        let footer_line = result
            .lines()
            .find(|l| l.contains("functions affected across 2 hops"))
            .unwrap_or_else(|| panic!("footer line missing from output:\n{result}"));

        let reported: usize = footer_line
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap_or_else(|_| panic!("footer count not a number: {footer_line}"));

        // hop-1 = 8 (all_caller_names, pre-truncation); hop-2 = 1 (hop2_fn → caller_a_0).
        assert_eq!(
            reported, 9,
            "footer reported {reported} but expected exactly 9 (8 hop-1 + 1 hop-2); \
             old post-truncation rebuild would undercount: {footer_line}"
        );
    }
}
