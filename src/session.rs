use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::SystemTime;

/// Tracks MCP activity across calls.
/// Stored alongside `OutlineCache` in server state.
///
/// Every field here is read by a tool a client can actually call: `expanded` by every expanding
/// surface, the token counters by `tilth_savings`. That is the invariant, and it is newer than it
/// looks — this type also carried reads, searches and per-query/per-directory histograms whose only
/// reader was `tilth_session`, a tool the server stopped advertising in 6ea62e8 and kept answering.
/// Removing it (#86) left them write-only, so they went with it, and with them a mutex acquisition
/// on every read and every search. Anything added here needs a reader a client can reach, or it is
/// the same mistake.
pub struct Session {
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
            expanded: Mutex::new(HashMap::new()),
            baseline_tokens: AtomicU64::new(0),
            saved_tokens: AtomicU64::new(0),
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
}
