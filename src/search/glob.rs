use std::collections::{BinaryHeap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

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
/// It used to keep whichever entries won a race — `if locked.len() < MAX_FILES { push }`
/// inside the parallel callback — and then render them in insertion order with no sort.
/// So `tilth_files` was non-deterministic in **both** membership and order: five identical
/// runs of `*.h` over one module of a 176k-file C++ tree produced five distinct outputs.
/// Unlike the search paths there is no ranking step here, so nothing downstream recovered
/// either property. This was the last count-gated parallel walk, after `callers` and then
/// `symbol`/`content`.
///
/// The fix keeps a bounded max-heap of the `MAX_FILES` **smallest paths seen so far**, plus
/// a counter for the true total. Both properties the old code lacked follow from that:
///
/// * *Membership* is a set — "the 20 smallest paths in the tree" does not depend on the
///   order they were offered, so thread scheduling cannot change it.
/// * *Order* comes from `into_sorted_vec` at the end, not from insertion.
///
/// The counter only ever **reports**. It is never read to decide whether to keep walking,
/// which is the distinction that made the old cutoffs non-deterministic across every one
/// of these walks: a counter that gates control flow over a parallel walk cannot be
/// deterministic, a counter that is read once after the threads join is fine.
///
/// A first version of this fix collected every matching path and sorted afterwards. Same
/// output, but retention then scaled with tree size — `tilth_files("*", scope: "/")`
/// matches every file on the volume, which is a plausible agent typo and would have added
/// a third unbounded-retention site to the two tracked in #19. The heap is O(`MAX_FILES`)
/// regardless of tree size, so that concern is gone rather than deferred.
///
/// Ordering is `Ord for Path`, which compares **component-wise** and case-sensitively. That
/// is not quite `ls`: a separator sorts below any character within a component, so
/// `src-gen/a.rs` sorts after `src/lib.rs`, and `Zebra.h` before `apple.h`. It is stable and
/// identical across platforms, which is what matters here. A cap still favours
/// lexicographically-early directories — a predictable bias, where the previous behaviour
/// was an unpredictable one, and the header reports the true total so the cap is visible.
///
/// Two incidental costs removed while here, both work that was thrown away:
/// `file_preview` stats the file and ran for *every* match before the cap check, so all but
/// `MAX_FILES` of those syscalls were discarded; and the shared extension set — used only
/// for the zero-match "did you mean" suggestion — was locked once per file in the tree
/// (strictly: per file whose extension is valid UTF-8) rather than per match.
///
/// Measured on a 176k-file C++ tree, three reps each. "varying" is literal — those are
/// distinct outputs from runs of an identical query (3 distinct in 3 for `*.h`, 2 in 3
/// for `*`); every other column was byte-identical across its reps.
///
/// ```text
///        before (racy)        collect-all + sort     bounded heap (this)
/// *.h    352-369ms, varying   295-325ms              200-230ms
/// *      668-690ms, varying   576-594ms              323-326ms
/// ```
///
/// Twice as fast as the code it replaces on `*`, mostly because `file_preview` ran per
/// match: for `*` over 176k files that was 176k `metadata` calls to render 20.
///
/// Peak RSS returns to the pre-fix level with the heap (10-13 MB on both patterns, against
/// 10-13 MB before); the intermediate collect-all version raised it to 32-35 MB on `*`,
/// which is what motivated the heap. Treat that column as order-of-magnitude only — peak
/// working set is sampled by polling here, and one heap rep read 27 MB on a query whose
/// retained set is twenty paths.
pub fn search(pattern: &str, scope: &Path) -> Result<GlobResult, TilthError> {
    let glob = Glob::new(pattern).map_err(|e| TilthError::InvalidQuery {
        query: pattern.to_string(),
        reason: e.to_string(),
    })?;
    let matcher = glob.compile_matcher();

    // Max-heap: the largest of the kept paths is at the top, so it is the one evicted once
    // the heap is over `MAX_FILES`. What survives is the `MAX_FILES` smallest.
    let kept: std::sync::Mutex<BinaryHeap<PathBuf>> = std::sync::Mutex::new(BinaryHeap::new());
    let total_found = AtomicUsize::new(0);
    let extensions: std::sync::Mutex<HashSet<String>> = std::sync::Mutex::new(HashSet::new());

    let walker = super::walker(scope, None)?;

    walker.run(|| {
        let matcher = &matcher;
        let kept = &kept;
        let total_found = &total_found;
        let extensions = &extensions;
        // Per-thread record of which extensions this thread has already contributed, so a
        // repeat costs a local lookup instead of the shared lock. It is not a deferred
        // merge — each newly-seen extension is inserted into the shared set eagerly, below.
        // The shared set was previously locked once per file in the tree, for a value only
        // read when there are zero matches.
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
                total_found.fetch_add(1, Ordering::Relaxed);

                // Allocate before taking the lock. Rust evaluates a method receiver before
                // its arguments, so `kept.lock().push(path.to_path_buf())` would run the
                // allocation *inside* the critical section — for a `*` glob that serialises
                // one allocation per file in the tree through a single mutex. The code this
                // replaced was explicitly careful about the same thing.
                let owned = path.to_path_buf();

                let mut heap = kept
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                heap.push(owned);
                if heap.len() > MAX_FILES {
                    // Evict the largest, keeping the smallest `MAX_FILES`.
                    heap.pop();
                }
            }

            ignore::WalkState::Continue
        })
    });

    let kept = kept
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let extensions = extensions
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    // Read once, after `walker.run` has joined every thread.
    let total = total_found.load(Ordering::Relaxed);

    // Ascending path order, independent of the order the heap received them.
    let selected = kept.into_sorted_vec();

    // Previews only for the survivors: one `metadata` call per rendered entry instead of
    // one per match.
    let files: Vec<GlobFileEntry> = selected
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
