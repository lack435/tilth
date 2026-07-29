use std::collections::{BinaryHeap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use globset::Glob;

use crate::error::TilthError;
use crate::types::estimate_tokens;

/// Files rendered per glob result. `pub(super)` so the tests in `search::mod` assert
/// against this rather than a hand-copied literal that can drift out of step with it.
pub(super) const MAX_FILES: usize = 20;

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
/// a third unbounded-retention site to the two tracked in #19. The heap holds
/// `MAX_FILES` paths regardless of tree size, so that concern is gone rather than deferred.
///
/// The *function* is not O(`MAX_FILES`), though, and it would be overclaiming to say so:
/// `extensions` still accumulates every distinct extension in the tree — plus a per-thread
/// copy — and is only truncated to 10 after the walk, for a value discarded whenever there
/// is at least one match. Bounded in practice by how many extensions a tree really has,
/// unbounded in principle (hash-suffixed artifacts, `.tmp1234`, split archives). Total
/// retention is O(`MAX_FILES` + distinct extensions × threads).
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
/// (strictly: per file whose extension is valid UTF-8). It is now once per thread per
/// distinct extension.
///
/// Measured on a 176k-file C++ tree, three reps each. "varying" is literal — those are
/// distinct outputs from runs of an identical query (3 distinct in 3 for `*.h`, 2 in 3
/// for `*`); every other column was byte-identical across its reps.
///
/// ```text
///        racy (before)        collect-all + sort   push-then-pop heap   peek-first (this)
/// *.h    352-369ms, varying   295-325ms            192-238ms            187-196ms
/// *      668-690ms, varying   576-594ms            312-337ms            185-199ms
/// ```
///
/// About 3.5x faster than the code this replaces on `*`, and every column after the first
/// is byte-identical to the others — the later ones are pure speedups, not different
/// answers.
///
/// The intermediate columns are kept because they carry the attribution, which is not what
/// it looks like from the endpoints alone. Moving `file_preview` to the survivors accounts
/// for 668 -> 585ms, under a third of the gain. Bounding retention took it to ~325ms. The
/// last step, peeking rather than push-then-pop, was expected to be a small tidy-up and was
/// worth another 325 -> 190ms: comparing against the current maximum is one `Path` compare
/// where sifting a doomed entry in and out is around twelve, and those compares run under
/// the lock over paths that share long prefixes.
///
/// Deliberately no memory column. Peak working set is sampled by polling here and the
/// readings were too noisy to carry a conclusion — one rep read 27 MB on a query whose
/// retained set is twenty paths. The case for the heap is the retention bound above, which
/// needs no measurement.
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
                // its arguments, so `kept.lock().push(path.to_path_buf())` would hold the
                // lock across the allocation — for a `*` glob that serialises one
                // allocation per file in the tree through a single mutex.
                let owned = path.to_path_buf();

                // Nothing that frees memory happens under the lock either, which is why
                // this peeks instead of the more obvious push-then-pop-if-over-capacity.
                // Once the heap is full, the overwhelmingly common case is a path that does
                // not belong in the page at all; push-then-pop would sift it to the root
                // and straight back out, so the buffer allocated just above would be freed
                // inside the critical section — 176k times for a `*` glob. Comparing
                // against the current maximum first costs one comparison instead of ~12,
                // and both the rejected path and any genuinely evicted one are released
                // after the guard is dropped.
                let discarded = {
                    let mut heap = kept
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);

                    if heap.len() < MAX_FILES {
                        heap.push(owned);
                        None
                    } else if heap.peek().is_some_and(|largest| *largest > owned) {
                        // Smaller than the largest kept, so it belongs in the page.
                        let evicted = heap.pop();
                        heap.push(owned);
                        evicted
                    } else {
                        // Larger than everything kept — cannot be in the smallest
                        // `MAX_FILES`, so it is never stored.
                        Some(owned)
                    }
                };
                drop(discarded);
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
