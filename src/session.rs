use std::collections::HashMap;
use std::fmt::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::SystemTime;

/// Tracks MCP activity across calls.
/// Stored alongside `OutlineCache` in server state.
pub struct Session {
    reads: AtomicUsize,
    searches: AtomicUsize,
    symbols: Mutex<HashMap<String, usize>>, // query → search count
    dir_hits: Mutex<HashMap<String, usize>>, // dir → count
    /// `path:line` → file mtime at expand-time. mtime versioning lets
    /// `is_expanded` detect stale records when the file has been edited
    /// since the expansion was first shown.
    expanded: Mutex<HashMap<String, SystemTime>>,
    /// Cumulative token estimates: sum of full-file baseline tokens and
    /// tokens actually returned across all reads in this session.
    baseline_tokens: AtomicU64,
    saved_tokens: AtomicU64,
}

impl Session {
    pub fn new() -> Self {
        Session {
            reads: AtomicUsize::new(0),
            searches: AtomicUsize::new(0),
            symbols: Mutex::new(HashMap::new()),
            dir_hits: Mutex::new(HashMap::new()),
            expanded: Mutex::new(HashMap::new()),
            baseline_tokens: AtomicU64::new(0),
            saved_tokens: AtomicU64::new(0),
        }
    }

    pub fn record_read(&self, path: &Path) {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.record_dir(path);
    }

    pub fn record_search(&self, query: &str) {
        self.searches.fetch_add(1, Ordering::Relaxed);
        let mut syms = self
            .symbols
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *syms.entry(query.to_string()).or_insert(0) += 1;
    }

    fn record_dir(&self, path: &Path) {
        if let Some(dir) = path.parent() {
            let key = dir.to_string_lossy().to_string();
            let mut dirs = self
                .dir_hits
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *dirs.entry(key).or_insert(0) += 1;
        }
    }

    /// Record a read event for savings accounting.
    /// `baseline_tokens`: estimated tokens for the full file (naive read).
    /// `returned_tokens`: estimated tokens for what tilth actually returned.
    /// Per-event clamp via `saturating_sub` ensures saved is never negative.
    pub fn record_savings(&self, baseline_tokens: u64, returned_tokens: u64) {
        if crate::cancel::worker_request_cancelled() {
            return;
        }
        self.baseline_tokens
            .fetch_add(baseline_tokens, Ordering::Relaxed);
        self.saved_tokens.fetch_add(
            baseline_tokens.saturating_sub(returned_tokens),
            Ordering::Relaxed,
        );
    }

    /// Returns `(baseline_tokens, saved_tokens)` accumulated this session.
    pub fn savings(&self) -> (u64, u64) {
        (
            self.baseline_tokens.load(Ordering::Relaxed),
            self.saved_tokens.load(Ordering::Relaxed),
        )
    }

    pub fn summary(&self) -> String {
        let reads = self.reads.load(Ordering::Relaxed);
        let searches = self.searches.load(Ordering::Relaxed);

        let mut out = format!("Files read: {reads} | Searches: {searches}");

        // Top symbols
        let syms = self
            .symbols
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !syms.is_empty() {
            let mut sorted: Vec<_> = syms.iter().collect();
            // Count descending, then name — the name tie-break is required, not cosmetic.
            // `syms` is a `HashMap`, `sort_by` is stable, and `take(5)` decides membership,
            // so equal counts were ordered by hash iteration and `RandomState` reseeds per
            // process. Ties are the *normal* case here: most queries are seen once. Same
            // defect as the `dirs:` line in `overview.rs` and the truncations fixed in
            // `callers`, `symbol`/`content` and `glob`.
            sorted.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
            let top: Vec<String> = sorted
                .iter()
                .take(5)
                .map(|(name, count)| format!("{name} ({count})"))
                .collect();
            let _ = write!(out, "\nTop queries: {}", top.join(", "));
        }

        // Hot paths
        let dirs = self
            .dir_hits
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !dirs.is_empty() {
            let mut sorted: Vec<_> = dirs.iter().collect();
            // Count descending, then path. Same reasoning as `Top queries` above.
            sorted.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
            let top: Vec<String> = sorted
                .iter()
                .take(5)
                .map(|(dir, count)| format!("{dir} ({count})"))
                .collect();
            let _ = write!(out, "\nHot paths: {}", top.join(", "));
        }

        out
    }

    pub fn reset(&self) {
        self.reads.store(0, Ordering::Relaxed);
        self.searches.store(0, Ordering::Relaxed);
        self.symbols
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
        self.dir_hits
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
        self.expanded
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
        self.baseline_tokens.store(0, Ordering::Relaxed);
        self.saved_tokens.store(0, Ordering::Relaxed);
    }

    /// Return true only when this `(path, line)` was previously expanded
    /// AND the recorded mtime matches `current_mtime`. After-edit re-grok
    /// falls back to a full re-inline.
    pub fn is_expanded(&self, path: &Path, line: u32, current_mtime: SystemTime) -> bool {
        let key = format!("{}:{}", path.display(), line);
        self.expanded
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&key)
            .is_some_and(|&recorded| recorded == current_mtime)
    }

    /// Both writers refuse an abandoned request, and the check belongs here rather than at the six
    /// call sites for a reason the walk check could not claim: this *is* the state that outlives a
    /// request, so guarding it is total, whereas guarding walk construction only covers the
    /// construction paths someone remembered to enumerate. See `cancel::worker_request_cancelled`
    /// for why a cancelled worker reaches these at all.
    pub fn record_expand(&self, path: &Path, line: u32, mtime: SystemTime) {
        if crate::cancel::worker_request_cancelled() {
            return;
        }
        let key = format!("{}:{}", path.display(), line);
        self.expanded
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(key, mtime);
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `summary` feeds `tilth_session`, so it must not vary run to run.
    ///
    /// `Top queries` and `Hot paths` both sorted a `HashMap`'s entries by count only, with a
    /// stable sort, then `take(5)`. Equal counts therefore kept hash-iteration order and
    /// `take` chose membership from it. Ties are the normal case, not an edge case: most
    /// queries in a session are seen exactly once, which is what this fixture reproduces —
    /// twelve distinct queries all at count 1, against a cap of five.
    ///
    /// A fresh `Session` per iteration is deliberate: `RandomState` reseeds per `HashMap`
    /// instantiation, so reusing one would test far less than it appears to.
    #[test]
    fn summary_is_byte_identical_across_sessions_with_tied_counts() {
        let render = || {
            let session = Session::new();
            for q in [
                "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
                "juliett", "kilo", "lima",
            ] {
                session.record_search(q);
            }
            for d in 0..12 {
                session.record_read(Path::new(&format!("d{d:02}/f.rs")));
            }
            session.summary()
        };

        let runs: Vec<String> = (0..8).map(|_| render()).collect();
        assert!(
            runs[0].contains("Top queries:"),
            "fixture produced no Top queries line, so this proves nothing:\n{}",
            runs[0]
        );
        assert!(
            runs.windows(2).all(|w| w[0] == w[1]),
            "summary varied across 8 sessions with identical input:\n{}",
            runs.join("\n---\n")
        );
        // All counts tie at 1, so the five shown must be the alphabetically first five.
        assert!(
            runs[0].contains("alpha (1), bravo (1), charlie (1), delta (1), echo (1)"),
            "tied counts must be broken by name:\n{}",
            runs[0]
        );
    }

    /// A worker whose request was abandoned must leave no trace in state that outlives it.
    ///
    /// `record_expand` is the one that can be *seen*: a later request reads it through
    /// `is_expanded` and prints `[shown earlier]` instead of a definition body. Since a cancelled
    /// walk stops wherever the deadline landed, without this guard how much got recorded would be
    /// a function of scheduling — and it would reach an answer that *is* returned, which is the
    /// shape #8/#18 removed.
    ///
    /// The uncancelled arm is what makes this a test rather than an assertion that the methods are
    /// broken: the same calls must still record normally when the request is live.
    #[test]
    fn an_abandoned_request_writes_nothing_that_outlives_it() {
        let _publish = crate::cancel::PUBLISH_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let session = Session::new();
        let request = crate::cancel::begin_request();
        let _bound = crate::cancel::bind_worker(request.token());
        let t = SystemTime::UNIX_EPOCH;

        session.record_expand(Path::new("live.rs"), 1, t);
        session.record_savings(1000, 200);
        assert!(
            session.is_expanded(Path::new("live.rs"), 1, t),
            "a live request must still record, or this test is passing for the wrong reason"
        );
        assert_eq!(session.savings(), (1000, 800));

        request.cancel();
        session.record_expand(Path::new("abandoned.rs"), 1, t);
        session.record_savings(5000, 100);
        assert!(
            !session.is_expanded(Path::new("abandoned.rs"), 1, t),
            "an abandoned request polluted the dedup state a later request reads"
        );
        assert_eq!(
            session.savings(),
            (1000, 800),
            "an abandoned request's discarded output was counted as savings"
        );
    }

    #[test]
    fn record_savings_accumulates_across_calls() {
        let session = Session::new();
        session.record_savings(1000, 200);
        session.record_savings(500, 100);
        let (baseline, saved) = session.savings();
        assert_eq!(baseline, 1500);
        assert_eq!(saved, 1200); // (1000-200) + (500-100)
    }

    #[test]
    fn record_savings_clamps_when_returned_exceeds_baseline() {
        let session = Session::new();
        // returned > baseline: saved contribution is 0, baseline still accumulates
        session.record_savings(100, 500);
        let (baseline, saved) = session.savings();
        assert_eq!(baseline, 100);
        assert_eq!(saved, 0);
    }

    #[test]
    fn record_savings_exact_match_adds_zero_saved() {
        let session = Session::new();
        session.record_savings(400, 400);
        let (baseline, saved) = session.savings();
        assert_eq!(baseline, 400);
        assert_eq!(saved, 0);
    }

    #[test]
    fn savings_getter_returns_both_counters() {
        let session = Session::new();
        let (b, s) = session.savings();
        assert_eq!(b, 0);
        assert_eq!(s, 0);
        session.record_savings(300, 50);
        let (b2, s2) = session.savings();
        assert_eq!(b2, 300);
        assert_eq!(s2, 250);
    }

    #[test]
    fn reset_zeroes_savings_counters() {
        let session = Session::new();
        session.record_savings(1000, 100);
        let (b, s) = session.savings();
        assert!(
            b > 0 && s > 0,
            "precondition: counters non-zero before reset"
        );
        session.reset();
        let (b2, s2) = session.savings();
        assert_eq!(b2, 0, "baseline_tokens must be zero after reset");
        assert_eq!(s2, 0, "saved_tokens must be zero after reset");
    }
}
