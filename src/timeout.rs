//! Per-request wall-clock timeout wrapper for synchronous tool calls.
//!
//! Rust has no sync-world equivalent of Java's `Future.get(timeout, unit)`, so
//! we spawn a fresh thread per call, wait on a `crossbeam_channel` bounded(1)
//! with `select! { default(timeout) => ... }`, and detach the worker on timeout.
//!
//! Because Rust cannot forcibly cancel a running thread, a timed-out worker
//! keeps running until its `FnOnce` completes naturally. A bounded counter
//! ([`ThreadTracker`]) tracks how many detached workers are still in flight and
//! refuses new work at [`MAX_ABANDONED_THREADS`] to prevent unbounded thread
//! accumulation on pathologically slow operations.
//!
//! It cannot be *forced* to stop, but it can be asked. Each spawn publishes a
//! [`crate::cancel`] token that the tree walks capture as they are built, and
//! the deadline arm sets it, so an abandoned worker's walk prunes itself
//! instead of running to completion (#61). The flag is set **only** inside the
//! branch that has already won `ThreadCoord::claim_timeout` — that CAS is what
//! guarantees the result is discarded, and it is the whole reason a per-file
//! flag check here is not the nondeterminism #8/#18 removed. `cancel`'s module
//! header states the property in full; `cancelling_cannot_change_a_returned_result`
//! below pins it against the race.
//!
//! [`ThreadCoord`] is a `RUNNING` → (`TIMED_OUT` | `FINISHED`) CAS state
//! machine that guarantees the tracker is incremented and decremented exactly
//! once per spawn even when the worker's channel send races the main
//! thread's deadline.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::{bounded, select, RecvError};

/// Warn-once threshold: print a single stderr line the first time the
/// abandoned-thread count crosses this value. The deadline arm checks
/// `if n == ABANDONED_THREAD_WARN { ... }` — deliberately `==`, not `>=`, so
/// the operator gets one signal at the threshold rather than a flood as each
/// subsequent timeout piles on. The hard cap at [`MAX_ABANDONED_THREADS`]
/// stops accepting work entirely, so silence past this point is bounded.
const ABANDONED_THREAD_WARN: usize = 3;
/// Hard cap: refuse new work when this many prior threads are still running
/// after timeout. Prevents unbounded thread accumulation on stuck operations.
const MAX_ABANDONED_THREADS: usize = 8;

/// Live count of threads that timed out and are still running in the background.
/// Owned by `Services`, so tests instantiate their own instance rather than
/// serializing on a global.
pub(crate) struct ThreadTracker {
    count: AtomicUsize,
}

impl ThreadTracker {
    pub(crate) fn new() -> Self {
        Self {
            count: AtomicUsize::new(0),
        }
    }

    /// `Acquire` here pairs with the `AcqRel` RMWs in `record_timeout` and
    /// `record_finish_after_timeout`. `Relaxed` would suffice for the counter
    /// value alone (atomic RMWs are atomic at any ordering); we pay the cheap
    /// upgrade so the pairing is canonical and the next reader doesn't have
    /// to re-derive that the value is consistent with the rest of the spawn
    /// state machine.
    pub(crate) fn is_at_cap(&self) -> bool {
        self.count.load(Ordering::Acquire) >= MAX_ABANDONED_THREADS
    }

    /// `AcqRel` is the canonical pessimistic ordering for an RMW that's read
    /// from another thread: documents intent better than `Release` alone, and
    /// guarantees the RMW sees any prior `Release` write to this atomic before
    /// computing its new value. The cost is negligible on x86 and one extra
    /// barrier on weakly-ordered architectures.
    fn record_timeout(&self) -> usize {
        self.count.fetch_add(1, Ordering::AcqRel) + 1
    }

    fn record_finish_after_timeout(&self) {
        self.count.fetch_sub(1, Ordering::AcqRel);
    }

    #[cfg(test)]
    pub(crate) fn current(&self) -> usize {
        self.count.load(Ordering::Acquire)
    }

    /// Pre-load the counter to the hard cap so callers can assert the
    /// `is_at_cap()` branch without launching real timeouts.
    #[cfg(test)]
    pub(crate) fn saturate(&self) {
        self.count.store(MAX_ABANDONED_THREADS, Ordering::Release);
    }
}

/// Per-request coordination between the main thread and the worker thread.
/// Exactly one of `claim_timeout` / `claim_finish` wins, so the tracker count
/// is updated at most once per spawn even when a worker's send and the main
/// thread's `select!` deadline race.
struct ThreadCoord {
    state: AtomicU8,
    /// Set to `true` after the main thread has incremented the tracker. The
    /// worker thread, if it lost the CAS to `TIMED_OUT`, must wait for this
    /// flag before decrementing — otherwise it could race ahead of the
    /// increment and underflow the counter.
    timeout_acked: AtomicBool,
}

impl ThreadCoord {
    const RUNNING: u8 = 0;
    const TIMED_OUT: u8 = 1;
    const FINISHED: u8 = 2;

    fn new() -> Self {
        Self {
            state: AtomicU8::new(Self::RUNNING),
            timeout_acked: AtomicBool::new(false),
        }
    }

    /// Main-thread side. Returns true if we transitioned `RUNNING` → `TIMED_OUT`;
    /// the caller must then increment the tracker and call `ack_timeout`.
    /// False means the worker already reached `FINISHED` — no counter change.
    fn claim_timeout(&self) -> bool {
        self.state
            .compare_exchange(
                Self::RUNNING,
                Self::TIMED_OUT,
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .is_ok()
    }

    /// Worker-thread side. Returns true if we transitioned `RUNNING` → `FINISHED`;
    /// no counter change needed. False means the main thread already flipped
    /// to `TIMED_OUT` and will increment — the caller must wait for
    /// `timeout_acked` and then decrement to undo it.
    fn claim_finish(&self) -> bool {
        self.state
            .compare_exchange(
                Self::RUNNING,
                Self::FINISHED,
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .is_ok()
    }

    fn ack_timeout(&self) {
        self.timeout_acked.store(true, Ordering::Release);
    }

    /// Spin until the main thread signals the tracker increment is visible.
    /// The main thread runs only a single counter update between `claim_timeout`
    /// and `ack_timeout`, so this loop terminates promptly in practice.
    ///
    /// `spin_loop()` is a CPU pipeline hint, not a scheduler yield — on a
    /// single-CPU container where the worker was scheduled before the main
    /// thread, a pure spin would burn the worker's full quantum (~10ms) before
    /// the main thread gets a turn to set the flag. The trailing `yield_now()`
    /// surrenders the rest of the quantum so the ack becomes visible in tens
    /// of microseconds instead.
    fn wait_for_timeout_ack(&self) {
        while !self.timeout_acked.load(Ordering::Acquire) {
            std::hint::spin_loop();
            std::thread::yield_now();
        }
    }
}

/// Reasons a `spawn_with_timeout` call did not return a value. Marked
/// `#[non_exhaustive]` so a future failure mode (e.g. OS-level thread spawn
/// failure) can be added without churning every call site.
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub(crate) enum SpawnFailure {
    Timeout,
    Panic,
}

/// Per-request timeout for tool calls. If a tool doesn't respond within this
/// duration, the MCP server returns a timeout error instead of hanging.
/// Override with `TILTH_TIMEOUT` env var (seconds). Default: 90s.
pub(crate) fn request_timeout() -> Duration {
    let secs = std::env::var("TILTH_TIMEOUT")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(90);
    Duration::from_secs(secs)
}

/// Run an arbitrary closure on a fresh thread with a wall-clock timeout.
/// Returns `Ok(result)` on success. On timeout, returns `Err(SpawnFailure::Timeout)`
/// and detaches the worker; the tracker is incremented and the worker will
/// decrement it when it eventually exits. On worker panic, returns `Err(Panic)`.
pub(crate) fn spawn_with_timeout<F, R>(
    tracker: &Arc<ThreadTracker>,
    timeout: Duration,
    work: F,
) -> Result<R, SpawnFailure>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let (tx, rx) = bounded::<R>(1);
    let coord = Arc::new(ThreadCoord::new());
    let coord_worker = Arc::clone(&coord);
    let tracker_worker = Arc::clone(tracker);
    // Published before the worker starts, so every walk `work` builds captures this request's
    // token rather than a previous request's or none.
    //
    // Shared with the worker rather than owned here, and that is a correctness point rather than
    // lifetime bookkeeping: `RequestCancel` un-publishes on drop, so if the main thread owned it
    // alone the publication would end when the *wait* ends — at the deadline — while the abandoned
    // worker is still running and still building walks. Those walks would find nothing published.
    // Holding it on both sides keeps it published for as long as the worker lives.
    let cancel = Arc::new(crate::cancel::begin_request());
    let cancel_worker = Arc::clone(&cancel);

    let handle = std::thread::spawn(move || {
        // Keeps this request's token published for the worker's whole life; see the comment on
        // `cancel` above. Never read here — held for its `Drop`.
        let _cancel_published = cancel_worker;
        // catch_unwind ensures claim_finish / record_finish_after_timeout run
        // even if work() panics after the main thread has already timed out.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work));
        if let Ok(val) = result {
            let _ = tx.send(val);
        }
        // tx is dropped here on panic, so main thread gets RecvError.
        if !coord_worker.claim_finish() {
            // Main thread already claimed the timeout. It will increment the
            // tracker before signalling `timeout_acked`; wait for that signal
            // before decrementing so we cannot underflow the counter.
            coord_worker.wait_for_timeout_ack();
            tracker_worker.record_finish_after_timeout();
        }
    });

    select! {
        recv(rx) -> msg => match msg {
            Ok(result) => {
                let _ = handle.join();
                Ok(result)
            }
            Err(RecvError) => Err(SpawnFailure::Panic),
        },
        default(timeout) => {
            // Claim the timeout before touching the tracker so a concurrent
            // `is_at_cap()` cannot observe an inflated count that we then roll
            // back. If the worker already won the CAS, we leave the tracker
            // alone entirely.
            if coord.claim_timeout() {
                // Inside the won CAS, and nowhere else. Winning here means the worker had not
                // reached `claim_finish`, so its result cannot be returned — which is the
                // precondition `cancel`'s safety argument requires. Setting the flag before
                // `record_timeout` only shortens the window; either order is correct, since
                // nothing reads the flag to decide what to *retain*.
                cancel.cancel();
                let n = tracker.record_timeout();
                coord.ack_timeout();
                if n == ABANDONED_THREAD_WARN {
                    eprintln!(
                        "tilth: warning: {n} abandoned threads still running. \
                         Consider reducing scope or increasing TILTH_TIMEOUT."
                    );
                }
            }
            Err(SpawnFailure::Timeout)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Serializes env-var tests so parallel `cargo test` execution doesn't
    /// race on process-global `TILTH_TIMEOUT`. Any test that mutates this
    /// env var must take this lock for its duration.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Every spawn publishes a cancellation token into a process-global slot, so parallel
    /// `cargo test` lets one test's spawn replace another's publication. Any test in this module
    /// must hold this for as long as it spawns.
    ///
    /// This is not hygiene, it is the module's own hazard reproduced in miniature: a request that
    /// begins while another is in flight takes over the slot, and a worker that captures after the
    /// takeover gets the wrong token. `mcp::serve` dispatches serially so it cannot happen in the
    /// server; the test suite is the only place two requests overlap, and this restores the
    /// serialisation the server has for free. Found by `only_the_expired_request_is_cancelled`
    /// failing about one run in four before the other tests here took the lock.
    fn spawning<'a>() -> std::sync::MutexGuard<'a, ()> {
        crate::cancel::PUBLISH_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Drives the real CAS path: a short-timeout `spawn_with_timeout` call
    /// races against a worker that sleeps past the deadline. The main thread
    /// must win the CAS (increment), and the worker must observe the lost CAS
    /// when it eventually exits (decrement). Ends with the counter back at zero.
    #[test]
    fn abandoned_counter_roundtrips_through_cas() {
        let _publish = spawning();
        let tracker = Arc::new(ThreadTracker::new());
        assert_eq!(tracker.current(), 0);

        let result: Result<(), SpawnFailure> =
            spawn_with_timeout(&tracker, Duration::from_millis(20), || {
                std::thread::sleep(Duration::from_millis(200));
            });

        assert_eq!(result, Err(SpawnFailure::Timeout));
        assert_eq!(tracker.current(), 1, "timeout must increment tracker");

        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while tracker.current() > 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(tracker.current(), 0, "worker exit must decrement tracker");
    }

    /// The two halves of #61 in the order the server produces them: an expired request's worker
    /// must *notice*, and the next request must not inherit the notice.
    ///
    /// Both are asserted rather than inferred. The first half is the fix — before it, the only
    /// evidence a worker had stopped was that it eventually stopped on its own. The second half is
    /// the property that keeps a per-file flag check distinct from the count-based early quits #18
    /// removed: request B runs to completion here, exactly as `mcp::serve` would run it after A's
    /// deadline, and its walks must never see a cancelled token. A global flag that is set on
    /// timeout and not cleared per request passes the first assertion and fails the second, which
    /// is the regression worth owning a test.
    ///
    /// Timings are one-sided: A is given 5 s to observe a cancel that fires at ~20 ms, so the
    /// assertion is about the mechanism working at all, not about a machine's scheduling latency.
    #[test]
    fn only_the_expired_request_is_cancelled() {
        let _publish = spawning();
        let tracker = Arc::new(ThreadTracker::new());
        // Written by the abandoned worker, read here after it exits.
        let observed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let observed_worker = Arc::clone(&observed);

        let expired: Result<(), SpawnFailure> =
            spawn_with_timeout(&tracker, Duration::from_millis(20), move || {
                // The worker runs on a fresh thread, so it opts that thread in before reading —
                // see `cancel::make_visible_on_this_thread`. Production needs no equivalent.
                let _visible = crate::cancel::make_visible_on_this_thread();
                // Captured as a walk builder captures it: once, at the start of the work.
                let token = crate::cancel::current();
                let deadline = std::time::Instant::now() + Duration::from_secs(5);
                while std::time::Instant::now() < deadline {
                    if token.is_cancelled() {
                        observed_worker.store(true, Ordering::Release);
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
            });
        assert_eq!(expired, Err(SpawnFailure::Timeout));

        // The abandoned worker decrements the tracker on its way out, so waiting for that is
        // waiting for it to have exited — no sleep-and-hope.
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while tracker.current() > 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(tracker.current(), 0, "abandoned worker never exited");
        assert!(
            observed.load(Ordering::Acquire),
            "an expired request's worker never saw the cancel, so it ran to completion"
        );

        // Now the next request, as the serial loop would issue it.
        let seen_by_next = spawn_with_timeout(&tracker, Duration::from_secs(5), || {
            let _visible = crate::cancel::make_visible_on_this_thread();
            let token = crate::cancel::current();
            std::thread::sleep(Duration::from_millis(20));
            token.is_cancelled()
        });
        assert_eq!(
            seen_by_next,
            Ok(false),
            "a live request inherited the previous request's cancellation"
        );
    }

    #[test]
    fn fast_work_returns_ok_without_counter_change() {
        let _publish = spawning();
        let tracker = Arc::new(ThreadTracker::new());
        let result = spawn_with_timeout(&tracker, Duration::from_secs(5), || 42_i32);
        assert_eq!(result.expect("fast work should not timeout"), 42);
        assert_eq!(tracker.current(), 0);
    }

    #[test]
    fn worker_panic_surfaces_as_panic_failure() {
        let _publish = spawning();
        let tracker = Arc::new(ThreadTracker::new());
        let result: Result<(), SpawnFailure> =
            spawn_with_timeout(&tracker, Duration::from_secs(5), || {
                panic!("boom");
            });
        assert_eq!(result, Err(SpawnFailure::Panic));
        assert_eq!(tracker.current(), 0, "panic must not leak a tracker slot");
    }

    #[test]
    fn saturated_tracker_reports_at_cap() {
        let tracker = Arc::new(ThreadTracker::new());
        assert!(!tracker.is_at_cap());
        tracker.saturate();
        assert!(tracker.is_at_cap());
    }

    #[test]
    fn request_timeout_reads_env() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::set_var("TILTH_TIMEOUT", "7");
        assert_eq!(request_timeout(), Duration::from_secs(7));
        std::env::remove_var("TILTH_TIMEOUT");
        assert_eq!(request_timeout(), Duration::from_secs(90));
    }
}
