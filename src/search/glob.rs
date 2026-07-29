use std::collections::HashSet;
use std::path::{Path, PathBuf};

use globset::Glob;

use crate::error::TilthError;
use crate::types::estimate_tokens;

const MAX_FILES: usize = 20;

pub struct GlobFileEntry {
    pub path: PathBuf,
    pub preview: Option<String>,
}

pub struct GlobResult {
    pub pattern: String,
    pub files: Vec<GlobFileEntry>,
    pub total_found: usize,
    pub available_extensions: Vec<String>,
}

/// Glob search using `ignore::WalkBuilder` (parallel via `super::walker` —
/// deliberately NOT .gitignore-aware, see `walker`'s doc comment).
///
/// The walk collects every matching path; the `MAX_FILES` cap is applied afterwards, to a
/// path-sorted list.
///
/// It used to keep whichever entries won a race — `if locked.len() < MAX_FILES { push }`
/// inside the parallel callback — and then render them in insertion order with no sort.
/// So `tilth_files` was non-deterministic in **both** membership and order: five identical
/// runs of `*.h` over one module of a 176k-file C++ tree produced five distinct outputs.
/// Unlike the search paths there is no ranking step here, so nothing downstream recovered
/// either property. This was the last instance of the count-gated-parallel-walk pattern
/// removed from `callers` and then from `symbol`/`content`.
///
/// Sorting by path — rather than ranking — is the deliberate choice for a file listing:
/// alphabetical truncation is what `ls` does and what a reader expects, and the header
/// reports the true total so a cap is visible rather than silent. It does mean a cap
/// favours alphabetically-early directories; that is a predictable bias, where the
/// previous behaviour was an unpredictable one.
///
/// Retaining every matching path costs one `PathBuf` per match rather than a bounded 20,
/// and that is the one real cost here. Measured on a 176k-file C++ tree, two reps each:
///
/// ```text
///            before                        after
/// *.h        360-634ms / 11 MB, varying    295-321ms / 18-34 MB, identical
/// *          685-698ms / 11-13 MB, varying 592ms     / 33-34 MB, identical
/// ```
///
/// So ~21 MB more on a `*` glob over 176k files, and *faster* despite collecting more —
/// because `file_preview` stats only the 20 rendered entries now instead of every match,
/// and the extension set is deduplicated per thread before touching the shared lock.
/// "varying" is literal: those before-columns are two different outputs in two runs.
pub fn search(pattern: &str, scope: &Path) -> Result<GlobResult, TilthError> {
    let glob = Glob::new(pattern).map_err(|e| TilthError::InvalidQuery {
        query: pattern.to_string(),
        reason: e.to_string(),
    })?;
    let matcher = glob.compile_matcher();

    let matched: std::sync::Mutex<Vec<PathBuf>> = std::sync::Mutex::new(Vec::new());
    let extensions: std::sync::Mutex<HashSet<String>> = std::sync::Mutex::new(HashSet::new());

    let walker = super::walker(scope, None)?;

    walker.run(|| {
        let matcher = &matcher;
        let matched = &matched;
        let extensions = &extensions;
        // Per-thread extension set, merged into the shared one when this thread's closure
        // is dropped. The shared set used to be locked once per *file in the tree* —
        // 176k uncontended-at-best acquisitions for a value only read when there are zero
        // matches — because extensions are gathered for the "did you mean" suggestion.
        let mut local_exts: HashSet<String> = HashSet::new();

        Box::new(move |entry| {
            let Ok(entry) = entry else {
                return ignore::WalkState::Continue;
            };

            if !entry.file_type().is_some_and(|ft| ft.is_file()) {
                return ignore::WalkState::Continue;
            }

            let path = entry.path();

            // Collect extensions for zero-match suggestions
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if !local_exts.contains(ext) {
                    local_exts.insert(ext.to_string());
                    extensions
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .insert(ext.to_string());
                }
            }

            // Match against filename or relative path
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let rel = path.strip_prefix(scope).unwrap_or(path);

            if matcher.is_match(name) || matcher.is_match(rel) {
                // Just the path. No cap decision here — that is what made membership
                // depend on thread scheduling — and no preview: `file_preview` stats the
                // file, and all but `MAX_FILES` of those stats were thrown away.
                matched
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(path.to_path_buf());
            }

            ignore::WalkState::Continue
        })
    });

    let mut matched = matched
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let extensions = extensions
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    // `total_found` is the true total, so a cap is reported rather than hidden.
    let total = matched.len();

    // Sort before truncating, so both *which* files survive and the order they render in
    // are fixed by the tree's contents and not by how the walk's threads were scheduled.
    matched.sort_unstable();
    matched.truncate(MAX_FILES);

    // Previews only for the survivors: one `metadata` call per rendered entry instead of
    // one per match.
    let files: Vec<GlobFileEntry> = matched
        .into_iter()
        .map(|path| {
            let preview = file_preview(&path);
            GlobFileEntry { path, preview }
        })
        .collect();

    let available_extensions: Vec<String> = if files.is_empty() {
        let mut exts: Vec<String> = extensions.into_iter().collect();
        exts.sort();
        exts.truncate(10);
        exts
    } else {
        Vec::new()
    };

    Ok(GlobResult {
        pattern: pattern.to_string(),
        files,
        total_found: total,
        available_extensions,
    })
}

/// Quick preview: token estimate, or "test file", or "module" based on exports.
fn file_preview(path: &Path) -> Option<String> {
    let meta = std::fs::metadata(path).ok()?;
    let tokens = estimate_tokens(meta.len());
    Some(format!("~{tokens} tokens"))
}
