//! Shared utilities used by both `edit` and `install`.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// Worker threads for the rayon pool and every parallel file walk.
///
/// One function because it is one policy read by two callers — `main::configure_thread_pools` for
/// the rayon global pool and `search::walker` for the `ignore` walkers. It was the same expression
/// copied into both, which is a policy stated twice and enforced once; the two agreed only by
/// coincidence of nobody having edited one of them.
///
/// **`TILTH_THREADS` costs memory as well as CPU, and the memory half is the larger surprise.**
///
/// The `clamp(2, 6)` was added for CPU (#27): back-to-back searches in a long-lived MCP session
/// otherwise sustain high load on a big machine. That is still the reason for the *shape* of this,
/// but it is not the only reason the ceiling matters. `find_definitions` holds one tree-sitter tree
/// per walk thread, and a tree is a large multiple of its file's bytes — so peak RSS carries a
/// `threads × tree_size` term. Measured (#70), 60 files of 499 000 B, three reps, isolated against
/// the same bytes with the grammar removed so only the parse differs:
///
/// ```text
///                                   t=1        t=6       t=32     per thread
/// ordinary source (39 B/line)     25.0 MB    95.1 MB   448 MB      ~13 MB
/// dense source    (16 B/line)     48.2 MB   219.9 MB  1090 MB      ~32 MB
/// ```
///
/// So ~26x the file's bytes for ordinary code and ~65x for a line-dense file, and the per-thread
/// figure is flat across a 32x change in thread count — which is what identifies it as one live tree
/// per thread rather than anything about match counts.
///
/// The practical consequence: at this default of at most 6, a search over a tree of large source
/// files peaks around 80-195 MB. Setting `TILTH_THREADS=32` on a 32-core machine — the obvious thing
/// to do for speed, and previously undocumented as anything but a CPU trade — makes that 448 MB to
/// 1.1 GB. Raising it is still reasonable; it is just not free, and the cost is linear in the value.
#[must_use]
pub fn worker_threads() -> usize {
    std::env::var("TILTH_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism().map_or(4, |n| (n.get() / 2).clamp(2, 6))
        })
}

/// Write `bytes` to `path` atomically: write to a temp file in the same
/// directory, preserve the original file's permissions (if it exists), then
/// rename into place. A crash mid-write leaves the original intact.
///
/// The temp name is qualified with the process ID and a process-wide counter
/// so concurrent or batched writes in the same directory can't collide.
pub(crate) fn atomic_write_bytes(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    // `Path::new("foo.txt").parent()` returns `Some("")`, not `None`; treat an
    // empty parent as "no directory" so the temp file anchors to "." rather than
    // the empty-string path.
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let tmp = dir.join(format!(".tilth-tmp.{}.{n}", std::process::id()));
    std::fs::write(&tmp, bytes).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })?;
    // Preserve original file permissions so the rename doesn't widen or strip
    // the mode. Ignore errors — target may not exist yet or platform may not
    // support it; the write already succeeded.
    if let Ok(meta) = std::fs::metadata(path) {
        let _ = std::fs::set_permissions(&tmp, meta.permissions());
    }
    std::fs::rename(&tmp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}
