//! Per-file Bloom filters for fast "does file X contain symbol Y?" queries.
//!
//! Used to pre-filter candidate files before expensive tree-sitter parsing
//! in callee/caller resolution. A Bloom filter can definitively say "no"
//! (symbol is NOT in this file) but may produce false positives.
//!
//! Identifier extraction uses a simple byte-level state machine -- no
//! tree-sitter needed -- making it fast enough to run on every uncached file.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use std::sync::atomic::Ordering::Relaxed;

use dashmap::DashMap;
use fastbloom::BloomFilter;

use crate::lang::detect_file_type;
use crate::types::{FileType, Lang};

// ---------------------------------------------------------------------------
// BloomFilterCache
// ---------------------------------------------------------------------------

/// Approximate memory ceiling for cached filters, in bytes.
///
/// The cache held one filter per code file it had ever been asked about, with no bound, and
/// the MCP server keeps one instance for the process lifetime — so resident memory climbed to
/// "one filter per code file in the tree" and stayed there. It plateaus rather than leaking,
/// but a server sitting at a few hundred MB after one query is a real cost on a machine also
/// running an editor and a compiler.
///
/// Sized against measured peak RSS. On a 176k-file C++ tree, one `kind: "callers"` query,
/// three reps in a single MCP session:
///
/// ```text
/// ceiling        peak RSS      wall
/// unbounded      188-214 MB    2947-3572ms
/// 64 MB          136-147 MB    3064-3190ms
/// 8 MB            70-78 MB     3583-3670ms
/// disabled        44-50 MB     3097-3917ms
/// ```
///
/// The ceiling controls peak memory — monotonic across four settings, which is what this bound
/// is for. Three things about that table are worth stating precisely, because an earlier version
/// of this comment got each of them wrong:
///
/// * Growth over the disabled baseline fits `~17 MB + 1.2 x ceiling` far better than any single
///   multiplier. The earlier text quoted "~1.5x", which is the 64 MB row alone; the row that
///   shipped implies 2.1x. The marginal undercount is ~1.2x, and there is a fixed component the
///   disabled baseline does not capture.
/// * The wall-time ranges do **not** all overlap: 64 MB (3064-3190ms) and 8 MB (3583-3670ms) are
///   disjoint, so the table does contain evidence that a tight ceiling costs time.
/// * Disabled was never the fastest setting here — its best is 3097ms against unbounded's
///   2947ms. It was faster than 8 MB, which is a different claim.
///
/// Within-setting spread is ~21% on the single-target rows at n=3, so differences smaller than
/// that are not resolvable from this data and no percentage is claimed for them.
///
/// 32 MB is what shipped. Against `main`, three reps each, with the exact `num_bits()`
/// accounting:
///
/// ```text
///                 unbounded                 32 MB ceiling
/// single-target   186-214 MB / 2995-4597ms   93-109 MB / 2919-3484ms
/// 5-target        213-248 MB / 10.5-10.8s   104-115 MB / 11.4-12.2s
/// ```
///
/// ~53% off peak RSS. The 5-target row is ~13% slower, and unlike the single-target differences
/// that one is resolvable: both ranges have ~7% internal spread and they are disjoint. It also
/// had a known mechanism — `bloom_walk::read_with_bloom_check` called `contains` once per target,
/// so a refused admission made every target rebuild the same filter instead of one building it
/// and the rest hitting. Unbounded, refusal only happened on a cold entry; with a ceiling it is
/// the steady state once the budget is full, which turned a non-issue into an N x multiplier on
/// filter-building work — as bad as not caching at all, and N x what a single build per file
/// would cost.
///
/// #34 fixed that by hoisting the build out of the per-target loop; see `contains_any`, where the
/// measurement lives. That measurement is a 2x2 over {pre-fix, post-fix} x {this ceiling,
/// unbounded} on a ~147k-file tree — a different tree from this table, so the absolute numbers do
/// not transfer, but the *ratio* asked about here does. It put the ceiling's wall-time cost at
/// +16.4% pre-fix, in the same region as the ~13% above, and at +3.4% post-fix. So most of this
/// row was the bug rather than an intrinsic cost of bounding the cache — but not all of it, and
/// the residual is still resolvable, so the ceiling is not free.
///
/// Output is unaffected at every setting — verified byte-identical with the cache unbounded,
/// bounded and disabled, and again across the #34 fix. It must be: a filter is only ever a
/// pre-filter ahead of a real `memmem` check and a parse, so a miss costs work and never a wrong
/// answer.
const MAX_CACHE_BYTES: usize = 32 * 1024 * 1024;

/// Fixed cost of one cache entry, beyond its bit array.
///
/// `CachedFilter` embeds the `BloomFilter` struct, its mtime and its own byte count; the map
/// also stores a `PathBuf` key inline. The rest is the key's heap allocation plus hashbrown's
/// control byte and load-factor slack, which is where the 128 comes from — deliberately
/// generous, since over-counting the overhead only makes the ceiling more conservative.
const PER_ENTRY_OVERHEAD: usize =
    std::mem::size_of::<CachedFilter>() + std::mem::size_of::<PathBuf>() + 128;

/// Consecutive sweeps an entry may go unprobed before the cache reclaims its bytes.
///
/// **Sweeps, not passes.** A sweep runs at the close of a pass that refused an admission, and only
/// then — see `BloomFilterCache::end_pass` — so a pass with room to spare neither ages an entry nor
/// clears its `probed` bit. The distinction is easy to lose and quiet when it is: a first version
/// of `scopes_visited_in_alternation_do_not_evict_each_other` oversubscribed the *pair* of scopes
/// while leaving each one fitting on its own, so half the passes refused nothing, no sweep ran on
/// them, the resident bits survived the round, and the test passed at 1 and at 2 alike —
/// discriminating nothing.
///
/// What this constant buys is tolerance of a workload that alternates between scopes rather than
/// staying in one. 2 is the smallest value with that property, and the property is structural
/// rather than tuned: a scope walked every other pass reads probed on its own pass, so its idle
/// count returns to 0 before it can reach 2. At 1 it is cleared by its neighbour's sweep in the
/// round before its own turn — measured on the corrected test as the whole resident scope evicted
/// in round 0, against zero evictions across four rounds at 2. (In that fixture only one of the two
/// scopes is ever resident, so "they take turns evicting each other" would overstate what was
/// seen: one scope is evicted, and the other never gets in either way.)
///
/// Two is not a general bound on rotation length, and the arithmetic that suggests it — "k
/// tolerates a cycle of period <= k" — does not survive contact with the fixtures. Whether a
/// rotation of period 3 survives at 2 depends on the ceiling, because a scope that fits entirely
/// refuses nothing on its own pass, skips a sweep, and hands its neighbours a round of grace:
/// measured, 3 scopes at a 12-entry ceiling are stable and the same 3 at 8 are not. Rotations
/// longer than the alternation are held by `EVICT_FREE_FRACTION` instead, which bounds the damage
/// rather than the period. See there.
const IDLE_SWEEPS_BEFORE_EVICTION: u8 = 2;

/// Largest share of the ceiling one sweep may free.
///
/// Without this the sweep is a synchronised clear of everything that aged out at once, and that
/// has a regime where it scores **zero**. Review found it and it reproduces to the digit: rotate
/// over k >= 3 disjoint scopes under memory pressure and every scope's entries age together, so
/// each one is dropped a pass or two before its turn comes round and no probe ever hits. Six
/// rounds, scopes of 10 files, arms paired inside one run:
///
/// ```text
/// scopes  ceiling     refusing   uncapped   capped
///      2   8 entries     33.3%      33.3%    33.3%
///      3   8 entries     22.2%       0.0%    19.4%
///      3  12 entries     33.3%      33.3%    33.3%
///      4  12 entries     25.0%       0.0%    20.8%
///      4  15 entries     31.2%       0.0%    27.1%
///      5  25 entries     41.7%       0.0%    36.7%
///     10  70 entries     58.3%       0.0%    52.5%
/// ```
///
/// Zero is worse than having no cache at all — it holds the memory and pays the eviction churn for
/// nothing — and it is the same failure the four rejected triggers were rejected for: throwing away
/// a cache that was earning. A larger `IDLE_SWEEPS_BEFORE_EVICTION` does not repair it: any
/// constant tolerates rotations shorter than itself and collapses on the next one up.
///
/// Two rows survive uncapped, for two different reasons, and neither generalises. The 2-scope row
/// is the alternation `IDLE_SWEEPS_BEFORE_EVICTION` is set to cover. The 3-scope, 12-entry row
/// survives because one 10-file scope fits that ceiling entirely, so its own pass refuses nothing,
/// skips a sweep, and hands its neighbours a round of grace — a property of that fixture's
/// arithmetic, not of the policy, and the same 3 scopes at 8 entries collapse.
///
/// Capping the *bytes* one sweep may release changes the shape of the failure rather than moving
/// its threshold: a synchronised clear becomes partial retention, and whatever survives a sweep is
/// re-probed on its own pass, resets to idle 0, and stays. The cache settles on a stable subset
/// instead of taking turns destroying itself. **This is a reduction, not an elimination** — the
/// capped column is still 0 to 4.5 points under refusal, because a sweep always releases at least
/// one entry and a rotation always has one aged. What it removes is the cliff.
///
/// It costs nothing on the case this exists to fix, because there the dead set is large and the
/// new one is small: 32 MB of unreachable filters against a subtree that caches in 1.9 MB, so one
/// sweep's 3.2 MB already covers the whole incoming scope. Measured over the same synthetic
/// fixture, first all-hit pass after a broad query that filled a 200-entry ceiling, incoming scope
/// 10% of it:
///
/// ```text
/// fraction   first all-hit pass   worst rotation row
/// uncapped                   3          0.0% (-25.0)
/// 1/4                        3         18.8%  (-6.2)
/// 1/10                       3         20.8%  (-4.2)
/// 1/25                       5         22.9%  (-2.1)
/// 1/50                       7         22.9%  (-2.1)
/// ```
///
/// A tenth is the last row that costs nothing on the target case. Below it the rotation keeps
/// improving, but only by ~2 points, and adaptation — the whole point of #40 — starts doubling.
/// The real ratio is more favourable than the row measured here (1.9 MB into 32 MB is 6%, not
/// 10%), so a tenth clears the incoming scope in one sweep with room to spare.
///
/// Matching `cache::EVICT_FREE_FRACTION`, which bounds the sibling `OutlineCache` for an unrelated
/// reason — eviction there scans linearly for the coldest entry, so the fraction amortises scan
/// cost rather than protecting a retention set — and arrived at the same value. Expressed as a
/// divisor for the reason recorded there, that the percentage form truncates to zero on small
/// budgets, and the cap is checked *before* each eviction rather than after, so a sweep always
/// releases at least one entry however small a ceiling a test picks.
const EVICT_FREE_FRACTION: usize = 10;

/// Consecutive **drain** sweeps an entry may go unprobed before the cache reclaims its bytes.
///
/// `IDLE_SWEEPS_BEFORE_EVICTION`'s counterpart for the no-pressure path, and much larger for a
/// reason that is the whole of #104's difficulty. Under pressure, two is right because something is
/// waiting for the bytes and the cost of being wrong is one rebuild. With nothing waiting, being
/// wrong costs a rebuild for no gain at all, so the only reason to reclaim is that the memory is
/// genuinely dead — and the horizon is what separates "dead" from "waiting its turn in a rotation".
///
/// An entry in a rotation over k scopes is re-probed every k passes, so any bound below k evicts a
/// live working set; a dead entry is never probed again and crosses any bound eventually. So this
/// wants to be comfortably larger than the longest rotation a session plausibly runs, and it costs
/// nothing but latency to make it so.
///
/// 16 protects rotations up to sixteen disjoint scopes. Measured on
/// `a_rotation_over_three_or_more_scopes_does_not_collapse`, whose worst row is ten scopes at 0.70x
/// the union: identical to the no-drain policy on every row. A rotation longer than sixteen would
/// begin to lose entries, bounded by `EVICT_FREE_FRACTION` per sweep — a slope, not the cliff an
/// uncapped sweep had, and the reason this is a horizon rather than a cliff-edge test.
///
/// The cost is latency. A dead set needs sixteen quiet passes before the first byte comes back and
/// `EVICT_FREE_FRACTION` more to finish, so ~25 calls on the #40 fixture. That is the right trade
/// when the thing being recovered is memory that is costing no hit rate: nothing is waiting on it,
/// so nothing is gained by being quick and something real is lost by being wrong.
const QUIET_SWEEPS_BEFORE_RECLAIM: u8 = 16;

/// Which question a sweep is answering. See `BloomFilterCache::sweep`.
#[derive(Clone, Copy)]
enum Sweep {
    /// The pass refused an admission: something is waiting for these bytes.
    Pressured,
    /// Nothing was refused, and an earlier sweep left entries it could not fit under its cap.
    Drain,
}

/// Real byte cost of caching `filter`.
///
/// An earlier version estimated this from the identifier count, on the stated grounds that
/// "`BloomFilter` exposes no size accessor". That was simply false — `num_bits()` is public —
/// and the estimate was wrong in a way that mattered: it undercounted a one-identifier file by
/// 3.2x and a 20k-identifier file by 1.0x, so the same nominal ceiling meant ~34 MB on a
/// large-file tree and ~100 MB on a header-heavy one. Asking the filter is exact and removes
/// the whole class.
fn entry_bytes(filter: &BloomFilter) -> usize {
    filter.num_bits() / 8 + PER_ENTRY_OVERHEAD
}

/// Thread-safe cache of per-file Bloom filters, keyed by path and validated
/// by mtime. Stale entries are automatically rebuilt on access.
///
/// Bounded by `ceiling`, and **adaptive** within it: admission is refused when the budget is
/// full, and entries the workload has stopped probing are released so the refusal does not become
/// permanent. `begin_pass` carries the design; this is what it is for and what it cost to arrive
/// at.
///
/// # The regime this exists to fix
///
/// The cache used to refuse and nothing more, on the grounds that eviction needs an access order
/// `DashMap` does not keep and a miss only costs a rebuild. The justification originally offered
/// for that — "repeated walks visit files in a similar order, so what gets in early is also what
/// gets re-probed" — does not hold. Successive tool calls routinely supply *different* scopes, and
/// the natural agent shape is adversarial: a broad query at the repo root fills the budget with
/// whatever the walk reached first, then the agent narrows to one subtree for the next fifty calls
/// and nothing it touches is ever admitted. Bounded, but useless.
///
/// #40 measured it, and it was worse than that paragraph guessed. A ~147k-file C++ tree, five
/// `kind: "callers"` calls scoped to one subtree after one broad call at the root,
/// `TILTH_THREADS=1`, against the same five on a cache that was never poisoned:
///
/// ```text
///                        hit rate   resident after
/// refusing (before)          0.0%          32.0 MB
/// sweeping (now)            40.0%          30.7 MB
/// fresh cache (control)     80.0%           1.9 MB
/// ```
///
/// The subtree caches completely in 1.9 MB, so the old cache was refusing the 1.9 MB that would
/// make it perfect in order to hold 32 MB it could never use.
///
/// Five calls is the window #40 specified, and it is too short to show the whole shape. The same
/// sequence run for thirty:
///
/// ```text
///                        hit rate   resident after
/// refusing                   0.0%          32.0 MB
/// sweeping                  90.0%           1.9 MB
/// fresh cache (control)     96.7%           1.9 MB
/// ```
///
/// Resident memory returns to exactly the fresh-cache figure, and the hit rate is 100.0% on every
/// call from the fourth onward — including every call during which the reclaim was running. That
/// second half is #104 and arrived after #40 was closed; see `drain_armed` for the mechanism and
/// `QUIET_SWEEPS_BEFORE_RECLAIM` for why it takes until call ~18 to start.
///
/// **The 40% is adaptation latency, not the steady state.** Per call, sweeping arm: calls 0 and 1
/// age the dead set out, call 2 admits the subtree, calls 3 and 4 run at 100.0%. The fresh
/// control's 80.0% is 4/5 by construction — it pays one cold call and then hits — so a long
/// session converges above it, not below. Five calls is the window #40 specified; it is short
/// enough that two sweeps of latency cost 40 points of it.
///
/// **The two halves are answered on different clocks, and the five-call table is why that is not
/// obvious.** One sweep releases `EVICT_FREE_FRACTION` of the ceiling — 3.2 MB — the subtree takes
/// 1.9 MB of it, and from there every call hits and refuses nothing. The hit rate is fixed at that
/// point; the remaining ~28.8 MB is not, because nothing is waiting for it and sweeping a cache
/// under no pressure is what `a_cache_with_room_to_spare_never_sweeps` forbids. #104 is the
/// separate, slower path that reclaims it without breaking that rule.
///
/// It is not an artefact of one subtree size or one tree. A 182-probe scope on the same tree, and a
/// 2129-probe subtree of a ~23.5k-file mixed-language tree, both read 0.0% / 40.0% / 80.0% over the
/// five-call window with the same per-call trace and the same residency at that point
/// (32.0 MB -> 29.1 MB and 32.0 MB -> 30.0 MB, before #104's reclaim has run).
/// All three incoming scopes are well under `EVICT_FREE_FRACTION` of the ceiling, which is the
/// condition for the trace to be that one — a scope larger than a tenth of the budget needs more
/// than one sweep to make room, and adaptation stretches accordingly. That is the ratio the
/// constant is chosen against, not a claim that scope size does not matter.
///
/// # Why the obvious fixes do not work
///
/// A generational reset — clear the map when the ceiling is hit — is what #40's body proposed, and
/// a first attempt at it was written, measured, and **rejected**. What that established is worth
/// more than the code was, because all of it is about why the obvious fixes fail, and because
/// every one of these failure modes is a way the present policy could have been written wrong:
///
/// * Refusal is a *good* policy when the working set genuinely exceeds the ceiling. Hit rate
///   degrades smoothly with the ceiling's share of the working set, because whatever got in keeps
///   earning. Any policy that clears has to beat that, and "clear when full" does not: both the
///   #40 regime and an overloaded cache are full, so fullness cannot be the trigger.
/// * A fixed miss-run trigger cannot work, for two separate reasons. Peak miss run per walk, one
///   fixture, five walks per ceiling:
///
///   ```text
///   ceiling / working set   walk 0   walks 1-4              (1-f)*W   ratio
///   0.25x                     1993   1354, 1354, 1354, 1354    1495    0.91
///   0.50x                     1993    856,  856,  856,  856     997    0.86
///   0.75x                     1993    384,  384,  384,  384     498    0.77
///   1.00x                     1993      0,    0,    0,    0       -       -
///   ```
///
///   `TILTH_THREADS=1`, three reps, every figure above bit-identical across all three — hit rates
///   included. `W` is the 1993 probes one walk makes over this scope.
///
///   First, **walk 0 is the full probe count at every ceiling** — 1993 on a 1993-probe scope,
///   including at 1.00x where nothing is ever refused and the cache is perfect. A first walk has no
///   hits available to interrupt it, so this is structural, not a property of being overloaded. Any
///   threshold below one walk's probe count fires on the first walk of every workload. (Walk 0 reads
///   1993 in every row for that reason; the 1.00x row is the clearest illustration, not extra
///   evidence.)
///
///   Second, and this is the one that kills a fixed bound: **the steady-state run is a function of
///   walker thread count**, because `miss_run` is one counter shared by the whole walk and any
///   thread's hit ends the run for all of them. The table above is `TILTH_THREADS=1`, where the four
///   steady walks are identical to the probe and reproduce bit-for-bit across reps. Raise the thread
///   count and the same cache behaviour reports a different number — roughly halving at 2 threads,
///   and landing anywhere in a wide band at the default, which is `available_parallelism() / 2`
///   clamped to 2..6 and therefore machine-dependent.
///
///   So the quantity is perfectly predictable per `(tree, ceiling fraction, thread count)` and
///   useless as a trigger anyway: a policy thresholding on it fires according to how many threads
///   the walker happened to start. An earlier reading of these rows claimed the run had "no stable
///   model" and was low by up to 6x against `~(1 - ceiling/working_set) x probes_per_walk`. That was
///   sampling noise, not a property of the run — under a fixed thread count the law holds at
///   0.77-0.91x of prediction here, and at a constant 1.13x on a second fixture. The law is a
///   serviceable upper bound with a tree-dependent constant; it is the thread dependence that rules
///   the signal out.
///
///   Note this carries forward to any replacement: normalising against resident entry count does
///   not help while the numerator is a thread-interleaved global count. A per-thread run, or a
///   quantity that does not depend on interleaving at all, is what the next attempt needs.
/// * Refused bytes measured against the ceiling also fails: at a ceiling of ~0.75x the working set
///   the refused portion is genuinely smaller than the ceiling even though the working set is not,
///   so it clears a cache that is earning well.
/// * Exponential backoff on repeated resets does not rescue a bad trigger if the backoff is
///   forgiven by any hit — an overloaded cache re-hits the block admitted just before each clear,
///   so the penalty never accumulates and the cache resets once per walk forever.
///
/// `peak_miss_run` remains as instrumentation. Nothing branches on it, and after the above nothing
/// should: it is the counter-example, not the signal.
///
/// # What works instead
///
/// Every rejected trigger above thresholds a **count**, and each has a regime where a cache that is
/// full and earning reads exactly like a cache that is full and dead. Fullness cannot be the
/// signal, because both are full.
///
/// What separates them is a **set**, not a count: which resident entries the workload is still
/// probing. A workload that keeps walking one oversized scope probes every resident entry every
/// time — that is precisely why refusal degrades gracefully there — while the #40 cache probes none
/// of them. So the decision is per-entry and needs no rate threshold: an entry that survived a
/// whole walk unprobed is not part of the current working set. `begin_pass` is where that is
/// implemented and argued.
///
/// Two constants qualify it, and both were put there by a regime that broke without them rather
/// than by tuning. `IDLE_SWEEPS_BEFORE_EVICTION` makes a workload alternating between two scopes
/// stable. `EVICT_FREE_FRACTION` bounds what one sweep may release, because a workload rotating
/// over three or more disjoint scopes is equally overloaded and does *not* re-probe everything each
/// pass — left uncapped that scores 0%, worse than refusal by up to 58 points. Each of those is
/// measured where it is defined.
///
/// The single-scope overloaded case comes out **identical**, not merely close. Repeated walks over
/// one scope, ceiling as a fraction of that scope's working set, paired within a single run so both
/// arms see the same tree in the same state (`TILTH_THREADS=1`):
///
/// ```text
/// ceiling / working set   ~147k-file C++ tree        ~23.5k-file mixed tree
///                         refusing  sweeping  evicted   refusing  sweeping  evicted
/// 0.25x                      25.6%     25.6%        0      12.4%     12.4%        0
/// 0.50x                      45.7%     45.7%        0      35.0%     35.0%        0
/// 0.75x                      64.6%     64.6%        0      57.0%     57.0%        0
/// 1.00x                      80.0%     80.0%        0      80.0%     80.0%        0
/// ```
///
/// Equal to the probe on every row, on two trees, because no sweep found anything to age. That is
/// the point of pairing the arms rather than comparing against the table in `mod adaptivity`: those
/// rows carry up to 15.9 points of run-to-run spread, so "within noise" would have been weak
/// evidence for the one property this policy must not break. Identical is not.
///
/// The shapes that *can* defeat "unprobed for a whole pass" are rotations, and none of the earlier
/// attempts had a fixture for any of them. Two-scope alternation is the benign end. Four rounds of
/// A, B against two disjoint subtrees, ceiling as a fraction of the pair's combined working set:
///
/// ```text
/// ceiling / working set   C++ pair                   C# pair
///                         refusing  sweeping  evicted   refusing  sweeping  evicted
/// 0.25x                      21.6%     21.6%        0       9.8%      9.8%        0
/// 0.50x                      42.1%     42.1%        0      36.4%     36.4%        0
/// 0.75x                      59.7%     59.7%        0      53.3%     53.3%        0
/// ```
///
/// Also identical, and for a reason rather than by luck — see `IDLE_SWEEPS_BEFORE_EVICTION`, and
/// `scopes_visited_in_alternation_do_not_evict_each_other`, which fails at a bound of 1.
///
/// **Three or more scopes is the hostile end, and this is the one row where the policy is worse
/// than refusal.** No real tree is needed to show it and no constant removes it; see
/// `EVICT_FREE_FRACTION` for the measurement and the mechanism. Capped, the rotation lands 0 to 4.5
/// points under refusal across three to ten scopes; uncapped it lands at zero. At twenty scopes the
/// loss widens to ~17 points, which is `QUIET_SWEEPS_BEFORE_RECLAIM`'s horizon being exceeded — but
/// that row reads the same with the #104 reclaim switched off (41.3% against 41.2%), so it belongs
/// to the capped sweep rather than to the reclaim.
///
/// The honest summary of both changes together: a large gain on the shape #40 is about, memory
/// returned to what the workload actually uses, no change at all on the overloaded single scope or
/// the two-scope alternation, and a small measured loss on a rotation over three or more.
///
/// Peak memory is bounded exactly as before, and the argument is mechanical rather than measured.
/// The sweep only ever frees, and frees in place — `retain` erases each bucket under its shard's
/// write lock, so no transient holds two generations, and a policy that only removes cannot raise
/// a peak. `mod adaptivity` cannot confirm the *intra-call* half of that on its own: `Snap` samples
/// `cached_bytes()` at call boundaries only, so what it shows is per-call endpoints (32.0 MB, then
/// 28.8 MB after one sweep, 30.7 MB once the subtree is admitted, and down to 1.9 MB across the
/// reclaim), never a state in between. What is not claimed at all is that process RSS falls with
/// the counter: the allocator is under no obligation to return freed pages to the OS.
///
/// Output is unaffected, as it must be and as it has been across #32 and #37: a filter is only ever
/// a pre-filter ahead of a real `memmem` check and a parse, so an evicted entry costs a rebuild and
/// never a wrong answer.
pub struct BloomFilterCache {
    filters: DashMap<PathBuf, CachedFilter>,
    /// Sum of `CachedFilter::bytes` over the map.
    ///
    /// Every read-modify-write of this happens while holding the `DashMap` shard lock for the
    /// key being changed, via `entry()`. That is load-bearing, not tidiness. The first version
    /// did the accounting outside the lock and claimed any race was "a transient overshoot
    /// bounded by one filter per thread"; it was neither. Two threads missing on the same path
    /// both built and both charged, so one entry was billed twice, permanently — and the window
    /// was the whole duration of `build_filter`, not a few instructions. Measured on two
    /// concurrent walks over 3000 shared files: 14.7 MB counted against 7.4 MB real, +98%.
    /// Four walks: +293%. That is reachable from a single `tilth_write` batch, whose
    /// `apply_batch` fans out with `into_par_iter` and reaches `find_callers_batch` per task,
    /// and it would pin the budget with a fraction of it real for the rest of the process —
    /// leaving the cache strictly worse than no cache, with nothing to show why.
    ///
    /// Races between *different* keys still overshoot transiently, by at most one filter per
    /// thread. That is the bound the old comment described, and for different keys it is true.
    bytes: std::sync::atomic::AtomicUsize,
    /// Byte ceiling. A field rather than a constant so tests can pick a small one — #13 asked
    /// for a *configurable* ceiling, and a 32 MB constant also made the growth test take 5s of
    /// a 5.7s suite because reaching the bound meant tokenising ~28M identifiers.
    ceiling: usize,
    /// Count of filters built, whether or not admission accepted them.
    ///
    /// Report-only — nothing reads it to decide control flow. It exists because "one build per
    /// file, not one per target" is the whole point of `contains_any` and the only way to assert
    /// it deterministically. A timing assertion would be flaky; a build count is exact.
    builds: std::sync::atomic::AtomicUsize,
    /// Probes answered from a cached filter with a matching mtime.
    ///
    /// With `builds`, this gives the hit rate. #40 needs it: the open question there is whether a
    /// full cache still *earns* its 32 MB, and "hit rate over a realistic query sequence" is the
    /// only way to answer that. Before this, the cache could be bounded, resident, and returning
    /// nothing, with no way to tell from outside.
    ///
    /// Report-only, like `builds`. Both are `Relaxed`: they are read after the walks they describe
    /// have joined, so no ordering is needed, and nothing branches on them.
    hits: std::sync::atomic::AtomicUsize,
    /// Admissions declined because the filter did not fit under `ceiling`.
    ///
    /// Distinguishes the two reasons a probe can miss — a cold entry, which the cache will serve
    /// next time, from a refused one, which it never will. A cache whose misses are nearly all
    /// refusals is full and useless, which is exactly the #40 regime; a cache whose misses are
    /// cold is simply warming up. The hit rate alone cannot tell those apart.
    refusals: std::sync::atomic::AtomicUsize,
    /// Consecutive misses with no intervening hit; zeroed by every hit.
    miss_run: std::sync::atomic::AtomicUsize,
    /// Longest value `miss_run` has reached.
    ///
    /// Instrumentation only — nothing in this file acts on it. It is here because the length of a
    /// miss run is the quantity an adaptive-eviction policy has to reason about, and #40's first
    /// attempt failed precisely by guessing at it. A fixed run-length trigger was measured against
    /// three fixtures on two trees and the natural run turned out to be
    /// `~(1 - ceiling/working_set) x probes_per_pass` — proportional to tree size, so any fixed
    /// bound is a per-tree tuning parameter in disguise. Recording the real distribution is what
    /// lets the next attempt normalise against something scale-free instead of guessing again.
    peak_miss_run: std::sync::atomic::AtomicUsize,
    /// Passes currently in flight. See `begin_pass`.
    ///
    /// A count rather than a flag because passes **overlap**: `edit::apply_batch` fans out with
    /// `into_par_iter` and every task can reach `find_callers_batch` at once, on different threads.
    /// (Not nesting — every other driver, `analyze_deps`, `grok`, `blast_radius`, `diff`, calls
    /// `find_callers_batch` sequentially. A count handles both, and only overlap is reachable.)
    /// Only the pass that takes this back to zero may sweep: an eviction decided while another walk
    /// is still probing would read "unprobed" for files that walk simply has not reached yet. So a
    /// write batch of N tasks adapts once rather than N times, which is the right trade — the cost
    /// of a missed sweep is a delayed adaptation, the cost of a wrong one is a live entry dropped.
    ///
    /// It closes the window rather than eliminating it: `end_pass` decrements to zero and *then*
    /// sweeps, so a new pass can start probing while that sweep is still running and be charged one
    /// spurious idle increment. Bounded — the sweep is one `retain` — and safe in the direction
    /// that only costs a rebuild.
    active_passes: std::sync::atomic::AtomicUsize,
    /// Refusals recorded while a pass was open, since the last pass closed.
    ///
    /// The gate on sweeping at all. No refusals means the ceiling was never in the way during this
    /// pass, so there is no demand the resident set is crowding out and nothing to gain by dropping
    /// any of it. Cleared by every last-in-flight `end_pass`, abandoned or not, and only written
    /// while `active_passes > 0` — both of those are corrections; see `end_pass` and `note_refusal`
    /// for what went wrong without them. Distinct from `refusals`, which is cumulative,
    /// unconditional and report-only.
    pass_refusals: std::sync::atomic::AtomicUsize,
    /// Set while a sweep is running, so a pass that opens and closes inside another sweep's window
    /// does not start a second one.
    ///
    /// Not "two passes ending together": only one thread can observe the `1 -> 0` decrement, so
    /// simultaneous closes cannot both reach `sweep`. The reachable overlap is the one above, and
    /// it is a sweep long. A second concurrent sweep would be *correct* — `retain` takes each
    /// shard's write lock and the accounting rides inside it — but it would double-count the idle
    /// increment, ageing every entry twice for one pass of elapsed work. Reasoned rather than
    /// measured: see `a_sweep_cannot_desynchronise_cached_bytes_from_the_map`, which covers
    /// sweep-against-admission and says plainly that it does not cover this.
    sweeping: std::sync::atomic::AtomicBool,
    /// Whether quiet passes should keep looking for bytes to reclaim.
    ///
    /// The whole of #104. Sweeping is gated on a pass having *refused* an admission, because a
    /// cache with room to spare has no reason to age anything — and that gate has an end state it
    /// cannot leave: once the live working set is resident, nothing is refused, no sweep runs, and
    /// whatever else is resident stays for the life of the process. On the #40 fixture that is
    /// ~28.8 MB of filters that will never be probed again, inside a 32 MB ceiling.
    ///
    /// **Armed by any pressured sweep, disarmed by a drain sweep that finds every resident entry
    /// probed.** Those two are the honest bracketing of "there might be something dead in here":
    /// a pressured sweep only happens when the budget was in the way, which is the only way a
    /// resident set the workload has moved off can arise; and a drain sweep that finds nothing
    /// unprobed has just proved the whole resident set is live, which is the only way to know
    /// there is nothing left to look for.
    ///
    /// A first version armed this from the *cap* instead — a sweep that hit `EVICT_FREE_FRACTION`
    /// with entries still in hand. That reads as the tidier receipt and does not work, because
    /// `QUIET_SWEEPS_BEFORE_RECLAIM` is long: the first drain sweep after it evicts nothing, the
    /// receipt clears, and the drain stops sixteen sweeps before it would have released a byte.
    /// Measured — the dead set sat at 200 of 200 entries for 45 passes.
    ///
    /// A workload that never refuses never arms it, so it never sweeps: `a_cache_with_room_to_spare
    /// _never_sweeps` holds unchanged, and so does the rotation whose union fits the ceiling.
    drain_armed: std::sync::atomic::AtomicBool,
    /// Sweeps run, and entries they dropped. Report-only, for the harness in `mod adaptivity`.
    sweeps: std::sync::atomic::AtomicUsize,
    evictions: std::sync::atomic::AtomicUsize,
}

struct CachedFilter {
    filter: BloomFilter,
    mtime: SystemTime,
    /// What this entry contributed to `bytes`, so replacing a stale entry can subtract it.
    bytes: usize,
    /// Probed since the last sweep. Set by every hit **and at admission** (`fresh_entry`, where the
    /// second half is load-bearing rather than symmetric), cleared by the sweep that reads it.
    ///
    /// Atomic because the hit path holds only the `DashMap` shard *read* guard — `contains_any`
    /// deliberately does not take the write lock to record a hit, since that would serialise
    /// every probe of a shard behind every other. The sweep reads it through `&mut V` under the
    /// write lock, so the two never overlap; the atomic is what makes probes from *different*
    /// walk threads sound.
    probed: std::sync::atomic::AtomicBool,
    /// The same evidence, read and cleared by **drain** sweeps instead.
    ///
    /// Two bits rather than one, because a single bit consumed by whichever sweep ran last makes
    /// `idle` count "pressured sweeps since the last sweep of any kind" rather than "since the last
    /// probe" — so every drain sweep silently shortens the pressured path's fuse, and a scope
    /// waiting its turn in a rotation dies sooner. That is not a hypothetical: it was measured at
    /// 34 hits against 50 on `a_rotation_over_three_or_more_scopes_does_not_collapse` before the
    /// second bit was added, with the separate clocks already in place. The clocks have to be
    /// independent all the way down, and one shared bit is the leak.
    probed_since_drain: std::sync::atomic::AtomicBool,
    /// Consecutive **pressured** sweeps that found this entry unprobed. Only ever touched under the
    /// shard write lock `retain` holds, so a plain integer is enough.
    idle: u8,
    /// The same for **drain** sweeps, which run when no admission was refused. Separate because the
    /// two questions have different answers: under pressure, two idle sweeps is enough to say an
    /// entry is not in the current working set, because something is waiting for its bytes. With
    /// nothing waiting there is no hurry and no cost to being sure, so the drain uses a horizon
    /// long enough to see a scope come round again — see `QUIET_SWEEPS_BEFORE_RECLAIM`.
    quiet_idle: u8,
}

impl Default for BloomFilterCache {
    fn default() -> Self {
        Self::new()
    }
}

impl BloomFilterCache {
    /// Create an empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::with_ceiling(MAX_CACHE_BYTES)
    }

    /// Create an empty cache with an explicit byte ceiling.
    #[must_use]
    pub fn with_ceiling(ceiling: usize) -> Self {
        Self {
            filters: DashMap::new(),
            bytes: std::sync::atomic::AtomicUsize::new(0),
            ceiling,
            builds: std::sync::atomic::AtomicUsize::new(0),
            hits: std::sync::atomic::AtomicUsize::new(0),
            refusals: std::sync::atomic::AtomicUsize::new(0),
            miss_run: std::sync::atomic::AtomicUsize::new(0),
            peak_miss_run: std::sync::atomic::AtomicUsize::new(0),
            active_passes: std::sync::atomic::AtomicUsize::new(0),
            pass_refusals: std::sync::atomic::AtomicUsize::new(0),
            sweeping: std::sync::atomic::AtomicBool::new(false),
            drain_armed: std::sync::atomic::AtomicBool::new(false),
            sweeps: std::sync::atomic::AtomicUsize::new(0),
            evictions: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Bytes currently accounted for by cached filters. Test/diagnostic accessor.
    #[must_use]
    pub fn cached_bytes(&self) -> usize {
        self.bytes.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Filters built since construction, admitted or not. Test/diagnostic accessor.
    #[must_use]
    pub fn filters_built(&self) -> usize {
        self.builds.load(Relaxed)
    }

    /// Probes served from a cached filter. Test/diagnostic accessor.
    ///
    /// `#[cfg(test)] pub(crate)`, unlike the older `cached_bytes` and `filters_built` above. Those
    /// are already published and narrowing them would be a breaking change, but there is no reason
    /// to add semver surface for counters only test code reads — and `pub(crate)` alone would trip
    /// `dead_code` in a non-test build, which is why the older pair is `pub` at all. The fields
    /// themselves are written unconditionally, so the counting is never compiled out.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn cache_hits(&self) -> usize {
        self.hits.load(Relaxed)
    }

    /// Admissions refused for want of budget. Test/diagnostic accessor.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn admissions_refused(&self) -> usize {
        self.refusals.load(Relaxed)
    }

    /// Longest run of consecutive misses with no intervening hit. Test/diagnostic accessor.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn peak_miss_run(&self) -> usize {
        self.peak_miss_run.load(Relaxed)
    }

    /// Read the peak miss run and clear it, so the next interval can be measured on its own.
    ///
    /// The cumulative peak is dominated by the first walk over any scope, which has no hits
    /// available to interrupt it and therefore always runs the full probe count. That single number
    /// hides the steady-state run, and the two are what an adaptive trigger has to tell apart — so
    /// the harness samples per walk rather than reading a running maximum.
    ///
    /// Clears the in-flight run as well as the peak. Without that, a walk whose first probe misses
    /// would report `previous_tail + 1` and the intervals would not be independent. On the fixtures
    /// measured so far it makes no difference — each walk happens to hit before it misses, so the
    /// tail is already 0 — but that is luck, not isolation.
    #[cfg(test)]
    pub(crate) fn take_peak_miss_run(&self) -> usize {
        self.miss_run.store(0, Relaxed);
        self.peak_miss_run.swap(0, Relaxed)
    }

    /// Open a pass. Closing the last one is what reclaims the bytes of anything it did not touch.
    ///
    /// A **pass** is one walk over a scope — `find_callers_batch`, the only place a whole tree is
    /// enumerated against this cache. The guard closes it on drop, so no early return can leak it;
    /// `find_callers_batch` takes it after its one `?` for a different reason, that there is
    /// nothing to sweep for a walk that never started.
    ///
    /// The pass is the unit that makes eviction possible here at all. `DashMap` keeps no access
    /// order, so the cache cannot rank entries by recency, and #40 measured what the alternative —
    /// refusing every admission once full — costs: 0.0% hit rate against 80.0% achievable, holding
    /// 32 MB to serve a subtree that caches completely in 1.9 MB. But it also measured why the
    /// obvious repairs fail. Every trigger tried before this one thresholded a *count* — hit rate
    /// over a window, refused bytes against the ceiling, consecutive misses — and each one has a
    /// regime where a cache that is full and earning reads the same as a cache that is full and
    /// dead. Fullness is not the signal, because both are full.
    ///
    /// What separates them is not a count but a *set*: which resident entries the workload is
    /// still probing. An overloaded cache — working set genuinely larger than the ceiling — probes
    /// **every** resident entry on every pass; that is what makes refusal degrade gracefully there,
    /// since whatever got in keeps earning. The #40 cache probes **none** of them. So the trigger
    /// is per-entry and needs no threshold: an entry that survived a whole pass without being
    /// probed is not part of the current working set, and its bytes are better spent on something
    /// that is. (`IDLE_SWEEPS_BEFORE_EVICTION` requires two such sweeps rather than one, which is
    /// not a rate threshold in disguise — it is what makes a workload alternating between two
    /// scopes stable. See there.)
    ///
    /// For a workload that keeps walking **one** scope larger than the ceiling, that gives a
    /// guarantee rather than a measurement: every resident entry is probed on every pass, so every
    /// `probed` bit is set, so the sweep evicts nothing and the cache behaves exactly as refusal
    /// did. The thrash rows in `mod adaptivity` are noisy enough that a measured no-regression
    /// would have been weak evidence; that one is structural.
    ///
    /// **The guarantee is that narrow, and the boundary is worth stating precisely, because a first
    /// version of this comment claimed it for "the overloaded case" in general and was wrong.** A
    /// workload that *rotates* over three or more disjoint scopes is equally overloaded and does
    /// not probe every resident entry on every pass — each scope's entries go unprobed while its
    /// neighbours are walked. Left alone that collapses to a 0% hit rate, worse than refusal by up
    /// to 58 points; `EVICT_FREE_FRACTION` is what contains it, and contains rather than removes
    /// it. See there, and `a_rotation_over_three_or_more_scopes_does_not_collapse`.
    ///
    /// Overlap is why this is a counter and not a flag — see `active_passes`.
    pub(crate) fn begin_pass(&self) -> PassGuard<'_> {
        self.active_passes
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        PassGuard {
            cache: self,
            sweep: true,
        }
    }

    /// Close a pass and sweep if this was the last one in flight.
    ///
    /// The refusal count is cleared whenever this close is the last one, **before** the `sweep`
    /// flag is consulted. An earlier version returned first on the abandon path and left the count
    /// standing, so a walk that was cancelled mid-way handed its refusals to the next pass and
    /// armed a sweep that pass had not earned. Review found it; the window it opens is the whole
    /// gap between two requests.
    fn end_pass(&self, sweep: bool) {
        if self
            .active_passes
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel)
            != 1
        {
            // Another walk is still probing. Its files would read as unprobed only because it has
            // not reached them. Leave the refusal count for whichever pass closes last.
            return;
        }
        let refused = self.pass_refusals.swap(0, Relaxed);
        if !sweep {
            return;
        }
        if refused == 0 && !self.drain_armed.load(Relaxed) {
            // Nothing was refused, so the ceiling never got in the way and no admission is waiting
            // on these bytes. Ageing entries here would evict a cache with room to spare. The one
            // exception is #104's drain, below, and it can only be armed by a pass that *did* find
            // the budget in the way.
            return;
        }
        self.sweep(if refused > 0 {
            Sweep::Pressured
        } else {
            Sweep::Drain
        });
    }

    /// Age every entry, and drop those unprobed for `IDLE_SWEEPS_BEFORE_EVICTION` consecutive
    /// sweeps — up to `EVICT_FREE_FRACTION` of the ceiling in one go.
    ///
    /// `retain` takes one shard's write lock at a time and calls the closure while holding it, so
    /// the `sub_bytes` below is inside the lock for the key it is un-charging. That is the same
    /// discipline `admit` follows and for the same reason — see the note on `bytes`, where doing
    /// this arithmetic outside the lock double-billed permanently.
    ///
    /// Frees in place: `retain` erases each bucket as it goes, so the peak never holds two
    /// generations and the ceiling bounds resident bytes during a sweep exactly as it does outside
    /// one. That is the whole reason this is a `retain` rather than "build the survivor set, then
    /// swap the map".
    fn sweep(&self, kind: Sweep) {
        if self
            .sweeping
            .swap(true, std::sync::atomic::Ordering::Acquire)
        {
            return;
        }
        let mut evicted = 0usize;
        let mut freed = 0usize;
        let mut found_idle = false;
        let cap = self.ceiling / EVICT_FREE_FRACTION;
        self.filters.retain(|_, entry| {
            // Each kind of sweep reads its own evidence bit and advances its own clock, and neither
            // touches the other's. That separation is the whole reason the drain costs the rotation
            // regime nothing: a drain that ticked `idle`, or that consumed the bit `idle` is
            // computed from, would shorten the pressured path's fuse on every quiet pass, and a
            // scope merely waiting its turn would die a pass or two sooner. Measured at 4 scopes /
            // 12 entries — 60 hits refusing, 50 with no drain, 40 with a drain that aged, 34 with
            // separate clocks but a shared bit.
            //
            // Ageing is not capped even though the release is. Stalling it too would mean a sweep
            // that hit its cap made no progress on the rest of a dead set, and the next sweep would
            // start it from scratch.
            let (clock, bound) = match kind {
                Sweep::Pressured => {
                    if entry.probed.swap(false, Relaxed) {
                        entry.idle = 0;
                        return true;
                    }
                    entry.idle = entry.idle.saturating_add(1);
                    (entry.idle, IDLE_SWEEPS_BEFORE_EVICTION)
                }
                Sweep::Drain => {
                    if entry.probed_since_drain.swap(false, Relaxed) {
                        entry.quiet_idle = 0;
                        return true;
                    }
                    // Something here is not being probed. Whether or not it is old enough to
                    // release yet, there is still a reason to come back.
                    found_idle = true;
                    entry.quiet_idle = entry.quiet_idle.saturating_add(1);
                    (entry.quiet_idle, QUIET_SWEEPS_BEFORE_RECLAIM)
                }
            };
            if clock < bound {
                return true;
            }
            if freed >= cap {
                // Over budget for this sweep. It stays at or above its bound, so the next sweep
                // reconsiders it first-class rather than restarting its clock.
                return true;
            }
            self.sub_bytes(entry.bytes);
            freed += entry.bytes;
            evicted += 1;
            false
        });
        match kind {
            // Any pressured sweep means the budget was in the way, which is the only way a resident
            // set the workload has moved off can come about. Look for one.
            Sweep::Pressured => self.drain_armed.store(true, Relaxed),
            // A drain sweep that found every resident entry probed has proved there is nothing to
            // reclaim. Stop scanning until pressure says otherwise.
            Sweep::Drain => self.drain_armed.store(found_idle, Relaxed),
        }
        self.sweeps.fetch_add(1, Relaxed);
        self.evictions.fetch_add(evicted, Relaxed);
        self.sweeping
            .store(false, std::sync::atomic::Ordering::Release);
    }

    /// Sweeps run since construction. Test/diagnostic accessor.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn sweeps_run(&self) -> usize {
        self.sweeps.load(Relaxed)
    }

    /// Entries dropped by sweeps since construction. Test/diagnostic accessor.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn entries_evicted(&self) -> usize {
        self.evictions.load(Relaxed)
    }

    /// Filters currently resident. Test/diagnostic accessor.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn resident_entries(&self) -> usize {
        self.filters.len()
    }

    /// What the map actually holds, summed by walking it. Test/diagnostic accessor.
    ///
    /// `cached_bytes` is the running counter; this is ground truth. The two must agree, and the
    /// only reason to have both is to assert that they do — the counter has been wrong before, in
    /// a way that pinned the budget permanently with a fraction of it real.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn summed_entry_bytes(&self) -> usize {
        self.filters.iter().map(|e| e.value().bytes).sum()
    }

    /// Extend the current miss run and record it if it is the longest so far.
    ///
    /// `fetch_max` rather than load-compare-store: the peak is written from every walk thread, and
    /// a read-modify-write outside a lock would lose updates and under-report exactly the long runs
    /// this exists to catch.
    fn note_miss(&self) {
        let run = self.miss_run.fetch_add(1, Relaxed) + 1;
        self.peak_miss_run.fetch_max(run, Relaxed);
    }

    /// Check if `symbol` might appear in the file at `path`.
    ///
    /// - If a cached filter exists with matching `mtime`, queries it directly.
    /// - Otherwise, builds a new filter from `content`, caches it, then queries.
    ///
    /// Returns `true` if the symbol MIGHT be in the file (possible false positive),
    /// `false` if it is DEFINITELY absent.
    #[must_use]
    pub fn contains(&self, path: &Path, mtime: SystemTime, content: &str, symbol: &str) -> bool {
        self.contains_any(path, mtime, content, [symbol])
    }

    /// Check if **any** of `targets` might appear in the file at `path`.
    ///
    /// Semantically `targets.any(|t| self.contains(path, mtime, content, t))`, but it builds at
    /// most one filter for the file rather than one per target. That distinction is the whole
    /// reason this exists: with the cache bounded, a refused admission is the steady state once
    /// the budget is full, and per-target `contains` then rebuilt the identical filter N times.
    /// Callers pass whole target sets — `find_callers_batch` every target of a `callers` query,
    /// `blast_radius` every symbol a `tilth_write` batch touched — so N is 5 or 20, not 1, and
    /// the result was strictly worse than having no cache at all. Keeping the fan-out here
    /// rather than in each caller makes "one build per file" structural.
    ///
    /// Returns `true` if some target MIGHT be in the file (possible false positive), `false` if
    /// every target is DEFINITELY absent. Empty `targets` is vacuously `false`, and costs no
    /// build — matching `any` on an empty iterator.
    ///
    /// Measured on a ~147k-file C++ tree, `kind: "callers"` with 5 targets, as a 2x2 over
    /// {pre-fix, post-fix} x {32 MB ceiling, unbounded}. Six sessions per arm, three reps each,
    /// arm order shuffled per session. The reported unit is the **session mean**: reps share a
    /// process and therefore a warm cache, so they are not independent observations, and treating
    /// them as such is how the first version of this comment claimed a p-value ~100x too small.
    /// `p2` is an exact two-sided permutation test on session means; 0.0022 is the 6-vs-6 floor.
    ///
    /// ```text
    ///                  pre-fix   post-fix    delta
    /// 32 MB ceiling    10638ms    9361ms    -12.0%   p2=0.0022  fully separated
    /// unbounded         9142ms    9052ms     -1.0%   p2=0.21    overlapping
    /// ```
    ///
    /// The second row is the evidence. Unbounded, the cache admits, so target 1 built and 2..N
    /// hit — there was no fan-out to remove, and the fix does nothing measurable. The gain appears
    /// only in the configuration where the bug existed. A before/after pair on one configuration
    /// could not separate this from a general speedup; the interaction can.
    ///
    /// Read down the columns and it answers #34's other question — what the ceiling costs now:
    ///
    /// ```text
    /// ceiling cost, pre-fix    +16.4%   p2=0.0022  fully separated
    /// ceiling cost, post-fix    +3.4%   p2=0.0043  overlapping
    /// ```
    ///
    /// So most of the ceiling's throughput penalty was this bug, but not all of it — ~3.4% remains
    /// and is still resolvable. "Almost entirely" would be too strong.
    ///
    /// Peak RSS: the ceiling still halves it (241 MB unbounded vs 120 MB bounded, p2=0.0022, fully
    /// separated). The fix itself does not move it (118 -> 120 MB, p2=0.47, overlapping), which is
    /// what should happen — admission logic is untouched, both arms fill the same 32 MB, and the
    /// fix removes redundant *transient* builds rather than retained bytes.
    ///
    /// The history of this measurement is worth recording, because three of its four earlier
    /// readings were wrong and two of them survived a review round:
    ///
    /// * A 2-session before/after gave -11.5% and a p-value of "~0.1%", computed by treating six
    ///   reps as six independent observations. On the session as the unit that p was 17%.
    /// * A 5-session rerun gave +4.3% (p=0.27) — no effect at all — plus an apparent "pre-fix
    ///   variance is 10x post-fix". Both were artifacts of fixed arm order and an unsettled
    ///   machine, with pre-fix session means spanning 7901-11255ms (CV 14.4%, against 1.4% here).
    ///   The variance finding does not survive randomised order and is claimed nowhere.
    /// * The same 2x2 run against the pre-#36 base showed the fix *raising* bounded peak RSS by
    ///   ~12 MB, fully separated. It does not reproduce here. Treat it as unexplained noise in
    ///   that run rather than a property of the fix.
    ///
    /// What has replicated, on both sides of #36 and across three separate runs, is the -12% on
    /// the bounded arm together with no effect on the unbounded arm. That pairing is the result;
    /// the point estimate itself has moved between 11.3% and 12.0% and should not be read as
    /// tighter than that.
    ///
    /// Output is byte-identical across all four arms — two response digests over the whole run,
    /// one per query shape. It must be: a filter is only ever a pre-filter ahead of a real
    /// `memmem` check and a parse, so a miss costs work and never a wrong answer.
    ///
    /// **`targets` must not touch this cache.** The cached-filter arm holds the `DashMap` shard
    /// read lock across the whole target loop, so a target iterator that re-entered the cache
    /// would deadlock — and not only on the write side: the shard guard is a `parking_lot`-style
    /// `RwLock`, where a re-entrant *read* also deadlocks if a writer is already queued. Under
    /// `find_callers_batch` a queued writer is the normal state, since every other walk thread is
    /// calling `admit`. No current caller does this: they pass slices, `HashSet`s, or (in
    /// `callees`) a lazy `Copied<Iter>` whose `next` only walks its own set. Left as a contract
    /// rather than enforced by draining into a `Vec`, because the drain would cost an allocation
    /// per candidate file on the cached-hit path — the path with no build to amortise it.
    #[must_use]
    pub fn contains_any<I, S>(
        &self,
        path: &Path,
        mtime: SystemTime,
        content: &str,
        targets: I,
    ) -> bool
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut targets = targets.into_iter().peekable();
        // No target means no question to answer, and building a filter to answer it would charge
        // the budget for a probe that cannot return true.
        if targets.peek().is_none() {
            return false;
        }

        // Fast path: check existing cached entry. One lookup, then every target queried against
        // the filter we already hold.
        if let Some(entry) = self.filters.get(path) {
            if entry.mtime == mtime {
                self.hits.fetch_add(1, Relaxed);
                self.miss_run.store(0, Relaxed);
                // This entry is part of the working set of whatever pass is running. `sweep`
                // reads and clears the bit; see `begin_pass`.
                entry.probed.store(true, Relaxed);
                entry.probed_since_drain.store(true, Relaxed);
                return targets.any(|t| entry.filter.contains(t.as_ref()));
            }
        }

        // Cache miss or stale: build outside any lock, answer from the fresh filter, then admit.
        // Admission may refuse; the answer is already in hand either way.
        let filter = build_filter(content, code_lang(path));
        self.builds.fetch_add(1, Relaxed);
        self.note_miss();
        let result = targets.any(|t| filter.contains(t.as_ref()));
        self.admit(path, mtime, filter);
        result
    }

    /// Offer `filter` to the cache, charging the budget if it fits.
    ///
    /// Every read-modify-write of `bytes` for this key happens inside `DashMap::entry()`, which
    /// holds the shard write lock for the whole match. That is what makes the accounting sound:
    /// with the arithmetic outside the lock, two threads missing on the same path both charged
    /// for one entry and the over-count was permanent. See the note on `bytes`.
    fn admit(&self, path: &Path, mtime: SystemTime, filter: BloomFilter) {
        let cost = entry_bytes(&filter);

        match self.filters.entry(path.to_path_buf()) {
            dashmap::mapref::entry::Entry::Occupied(mut occupied) => {
                if occupied.get().mtime == mtime {
                    // Another thread built and admitted the same version while we were
                    // building. Charging again is exactly the double-billing this design
                    // exists to prevent, so drop ours.
                    return;
                }
                // Stale: reclaim its budget before considering the replacement, so a file
                // edited repeatedly cannot consume the ceiling one revision at a time.
                let stale = occupied.get().bytes;
                self.sub_bytes(stale);
                if self.fits(cost) {
                    self.bytes.fetch_add(cost, Relaxed);
                    occupied.insert(fresh_entry(filter, mtime, cost));
                } else {
                    // Its budget is already reclaimed and its mtime can never match again, so
                    // keeping it would be resident memory the counter no longer knows about.
                    self.note_refusal();
                    occupied.remove();
                }
            }
            dashmap::mapref::entry::Entry::Vacant(vacant) => {
                if self.fits(cost) {
                    self.bytes.fetch_add(cost, Relaxed);
                    vacant.insert(fresh_entry(filter, mtime, cost));
                } else {
                    self.note_refusal();
                }
            }
        }
    }

    /// Record a refused admission, cumulatively and — if a pass is open — against the sweep window.
    ///
    /// The `active_passes` check is not bookkeeping tidiness. `admit` is reachable with no pass
    /// open at all: `callees::resolve_callees` probes this cache through `read_with_bloom_check`
    /// for each of a file's imports and never opens one, because a handful of imports is not a walk
    /// over a scope and must not be allowed to age anything. Without the check those refusals sat
    /// in the counter until some later callers walk closed, and armed a sweep on a pass whose own
    /// budget was never under pressure — so `a_cache_with_room_to_spare_never_sweeps` held in its
    /// fixture and not in production. Review found that; it is the same class of mistake as
    /// counting a cancelled pass's refusals, and both are fixed here and in `end_pass`.
    ///
    /// A refusal racing a pass close can still land on either side of it. That is a one-count
    /// difference in a gate that only asks "was anything refused", and either answer is defensible
    /// for a probe that overlapped the boundary.
    fn note_refusal(&self) {
        self.refusals.fetch_add(1, Relaxed);
        if self.active_passes.load(Relaxed) > 0 {
            self.pass_refusals.fetch_add(1, Relaxed);
        }
    }

    /// Would `cost` more bytes stay inside the ceiling? `saturating_add` so a corrupted counter
    /// refuses admission rather than overflowing.
    fn fits(&self, cost: usize) -> bool {
        self.bytes.load(Relaxed).saturating_add(cost) <= self.ceiling
    }

    /// `saturating_sub`, so an accounting slip can never wrap the counter to `usize::MAX` and
    /// permanently disable caching.
    fn sub_bytes(&self, amount: usize) {
        let _ = self
            .bytes
            .fetch_update(Relaxed, Relaxed, |b| Some(b.saturating_sub(amount)));
    }
}

/// A newly built entry, born already probed.
///
/// `probed: true` is not bookkeeping symmetry — it is a grace period, and without it the fix
/// eats its own work. Admission happens *during* a pass, so the sweep at the end of that same
/// pass would find every entry admitted after the last hit on it still unprobed, and start ageing
/// a filter the walk had just paid to build. Marking it probed says what is already true: the
/// probe that built it was a probe.
fn fresh_entry(filter: BloomFilter, mtime: SystemTime, cost: usize) -> CachedFilter {
    CachedFilter {
        filter,
        mtime,
        bytes: cost,
        probed: std::sync::atomic::AtomicBool::new(true),
        // `false`, unlike `probed`. The grace period that bit gives a freshly built filter is for
        // the pressured path, whose bound is two sweeps — short enough that admitting during a pass
        // and being aged at the end of it would eat the work. The drain's horizon is
        // `QUIET_SWEEPS_BEFORE_RECLAIM`, so grace buys nothing, and starting `true` actively breaks
        // it: the first drain sweep after a workload settles would find every entry "probed", read
        // that as proof the resident set is live, and disarm sixteen sweeps before it could release
        // anything. Measured — the dead set sat at 200 of 200 entries for 45 passes.
        probed_since_drain: std::sync::atomic::AtomicBool::new(false),
        idle: 0,
        quiet_idle: 0,
    }
}

/// Open pass marker. Closing it is what may trigger a sweep; see `BloomFilterCache::begin_pass`.
///
/// RAII rather than a paired call so an early `?` inside a walk driver cannot leave the count
/// raised — a leaked pass would suppress every future sweep for the life of the process, which is
/// exactly the "bounded but useless" state #40 is about, reintroduced by a bookkeeping slip.
pub(crate) struct PassGuard<'a> {
    cache: &'a BloomFilterCache,
    sweep: bool,
}

impl PassGuard<'_> {
    /// Close this pass without sweeping, because the walk did not finish.
    ///
    /// A walk that quits early — a cancelled request, a per-request timeout — enumerated only part
    /// of its scope, so "unprobed" for everything past the cut means "not reached", not "no longer
    /// wanted". Ageing on that would let two consecutive timeouts evict a live working set. The
    /// cost of skipping is nothing: the next complete pass sweeps, and the bits it reads are
    /// conservative in the safe direction — a file the truncated walk *did* probe stays marked, so
    /// the skipped sweep can only retain entries, never drop one it should have kept.
    pub(crate) fn abandon(mut self) {
        self.sweep = false;
    }
}

impl Drop for PassGuard<'_> {
    fn drop(&mut self) {
        self.cache.end_pass(self.sweep);
    }
}

/// The source language of `path`, or `None` when it is not a known code file.
fn code_lang(path: &Path) -> Option<Lang> {
    match detect_file_type(path) {
        FileType::Code(lang) => Some(lang),
        FileType::Markdown
        | FileType::StructuredData
        | FileType::Tabular
        | FileType::Log
        | FileType::Other => None,
    }
}

/// Build a Bloom filter from file content by extracting all identifiers.
///
fn build_filter(content: &str, lang: Option<Lang>) -> BloomFilter {
    let idents: Vec<&str> = extract_identifiers(content, lang).collect();
    // Sized for total token count, not unique identifiers -- duplicates over-allocate
    // the filter, so the achieved FPR is well below the 0.01 target in practice.
    let expected = idents.len().max(1);

    let mut filter = BloomFilter::with_false_pos(0.01).expected_items(expected);
    for ident in idents {
        filter.insert(ident);
    }
    filter
}

// ---------------------------------------------------------------------------
// Identifier extraction (byte-level state machine)
// ---------------------------------------------------------------------------

/// Extract identifier tokens from source code using a simple byte-level
/// state machine. Skips string literals and block/line comments.
///
/// An identifier is `[a-zA-Z_][a-zA-Z0-9_]*`.
///
/// This is intentionally approximate -- it does not understand all language
/// syntaxes perfectly, but is fast and good enough for Bloom filter population.
///
/// `lang` gates language-specific lexing: the Rust lifetime heuristic only
/// applies when `lang` is `Some(Lang::Rust)`. For every other language a `'`
/// opens a single-quoted string, matching their actual syntax.
fn extract_identifiers(content: &str, lang: Option<Lang>) -> impl Iterator<Item = &str> {
    IdentifierIter::new(content, lang)
}

/// States for the identifier extraction state machine.
#[derive(Clone, Copy)]
enum ScanState {
    /// Normal code scanning.
    Code,
    /// Inside a double-quoted string.
    StringDouble,
    /// Inside a single-quoted string/char.
    StringSingle,
    /// Inside a backtick string (JS template literals, Go raw strings).
    StringBacktick,
    /// Inside a line comment (// ...).
    LineComment,
    /// Inside a block comment (/* ... */).
    BlockComment,
}

struct IdentifierIter<'a> {
    bytes: &'a [u8],
    src: &'a str,
    pos: usize,
    state: ScanState,
    lang: Option<Lang>,
}

impl<'a> IdentifierIter<'a> {
    fn new(content: &'a str, lang: Option<Lang>) -> Self {
        Self {
            bytes: content.as_bytes(),
            src: content,
            pos: 0,
            state: ScanState::Code,
            lang,
        }
    }
}

impl<'a> Iterator for IdentifierIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        let bytes = self.bytes;
        let len = bytes.len();

        while self.pos < len {
            let i = self.pos;
            let b = bytes[i];

            match self.state {
                ScanState::Code => {
                    // Check for start of string literals
                    if b == b'"' {
                        self.state = ScanState::StringDouble;
                        self.pos += 1;
                        continue;
                    }
                    if b == b'\'' {
                        // Distinguish a Rust lifetime (`'a`, `'static`) from a char
                        // literal (`'a'`, `'\n'`). A char literal has a closing quote
                        // right after a single char/escape; a lifetime is a tick
                        // followed by an identifier with no closing quote. Treating a
                        // lifetime as a string opener would swallow every following
                        // identifier up to the next tick, dropping them from the filter
                        // and producing a false negative (the one thing Bloom forbids).
                        // Lifetimes are Rust-only; in other languages a `'` opens a
                        // single-quoted string, so the heuristic is gated on
                        // `has_lifetimes` to avoid swallowing identifiers after a
                        // `'foo'` string there.
                        let is_lifetime = self.lang.is_some_and(Lang::has_lifetimes)
                            && i + 1 < len
                            && is_ident_start(bytes[i + 1])
                            && !(i + 2 < len && bytes[i + 2] == b'\'');
                        if is_lifetime {
                            self.pos += 1;
                            continue;
                        }
                        self.state = ScanState::StringSingle;
                        self.pos += 1;
                        continue;
                    }
                    if b == b'`' {
                        self.state = ScanState::StringBacktick;
                        self.pos += 1;
                        continue;
                    }

                    // Check for comments
                    if b == b'/' && i + 1 < len {
                        if bytes[i + 1] == b'/' {
                            self.state = ScanState::LineComment;
                            self.pos += 2;
                            continue;
                        }
                        if bytes[i + 1] == b'*' {
                            self.state = ScanState::BlockComment;
                            self.pos += 2;
                            continue;
                        }
                    }

                    // Check for start of identifier
                    if is_ident_start(b) {
                        let start = i;
                        self.pos += 1;
                        while self.pos < len && is_ident_continue(bytes[self.pos]) {
                            self.pos += 1;
                        }
                        // Safety: identifiers are pure ASCII, so byte slicing is valid UTF-8
                        return Some(&self.src[start..self.pos]);
                    }

                    self.pos += 1;
                }

                ScanState::StringDouble => {
                    if b == b'\\' && i + 1 < len {
                        self.pos += 2; // skip escaped character
                    } else if b == b'"' {
                        self.state = ScanState::Code;
                        self.pos += 1;
                    } else {
                        self.pos += 1;
                    }
                }

                ScanState::StringSingle => {
                    if b == b'\\' && i + 1 < len {
                        self.pos += 2; // skip escaped character
                    } else if b == b'\'' {
                        self.state = ScanState::Code;
                        self.pos += 1;
                    } else {
                        self.pos += 1;
                    }
                }

                ScanState::StringBacktick => {
                    if b == b'\\' && i + 1 < len {
                        self.pos += 2;
                    } else if b == b'`' {
                        self.state = ScanState::Code;
                        self.pos += 1;
                    } else {
                        self.pos += 1;
                    }
                }

                ScanState::LineComment => {
                    if b == b'\n' {
                        self.state = ScanState::Code;
                    }
                    self.pos += 1;
                }

                ScanState::BlockComment => {
                    if b == b'*' && i + 1 < len && bytes[i + 1] == b'/' {
                        self.state = ScanState::Code;
                        self.pos += 2;
                    } else {
                        self.pos += 1;
                    }
                }
            }
        }

        None
    }
}

#[inline]
fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

#[inline]
fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Measurement harness for #40 — does a full cache still earn its ceiling?
///
/// `#[ignore]`d and driven by environment variables, because the question only has an answer on a
/// tree large enough to fill 32 MB, and no such tree can live in this repository. Run it as:
///
/// ```text
/// TILTH_ADAPTIVITY_ROOT=<large tree> TILTH_ADAPTIVITY_SUBTREE=<a subdir of it> \
///   TILTH_ADAPTIVITY_SUBTREE_B=<a second, disjoint subdir> \
///   cargo test --release adaptivity -- --ignored --nocapture
/// ```
///
/// `TILTH_ADAPTIVITY_SUBTREE_B` is needed only by `adaptivity_alternating_scopes`; the other two
/// tests skip it. Each test that has a before/after runs both arms **within one execution**, with
/// sweeping suppressed on one of them by `no_sweeps` — see there for why that is done with the
/// shipping code path rather than a test-only switch, and see the noise table below for why a
/// cross-run comparison would not have been usable.
///
/// It exists as a committed test rather than a throwaway script so the numbers in #40 can be
/// reproduced, and so the *shape* of the measurement is reviewable. It asserts almost nothing —
/// its output is the deliverable.
///
/// **Run every arm several times before believing any difference.** The thrash rows are noisy, and
/// badly so at the middle ceilings. Refuse-forever behaviour, one fixture, six observations per
/// row, no code change between them:
///
/// ```text
/// ceiling / working set   observed hit rate   spread
/// 0.25x                       14.1 - 26.1%    12.0 pts
/// 0.50x                       33.7 - 49.6%    15.9 pts
/// 0.75x                       58.3 - 64.7%     6.4 pts
/// 1.00x                       80.0 - 80.0%     0
/// ```
///
/// **This is a lower bound on the noise, from one fixture on one machine at its default thread
/// count — not a floor to compare against.** An earlier version of this table quoted 0.25x as the
/// quiet row at 2.7 points, from six samples. Nineteen samples put it at 12.0 points, with values
/// nine points below that entire range. The spread did not converge as samples were added; it kept
/// widening. Treat every row as "at least this noisy", and re-measure per fixture rather than
/// reusing these numbers.
///
/// The mechanism is the parallel walk, and that is demonstrated rather than assumed: `TILTH_THREADS=1`
/// makes every row reproduce bit-for-bit across reps, hit rates and per-walk miss runs alike. Once
/// the ceiling is reached, *which* files won the race into the cache decides what hits afterwards,
/// and thread scheduling decides that. **Set `TILTH_THREADS=1` for anything compared across builds.**
///
/// The 1.00x row's zero spread is arithmetic, not evidence: when the ceiling fits the working set,
/// walk 0 is all misses and walks 1-4 are all hits, so the rate is exactly 4/5 by construction. It
/// is a useful control precisely because it cannot vary, but it demonstrates nothing about the
/// mechanism — the deterministic-thread result above is what does.
///
/// This is why the first attempt at #40 reported a "worst regression of 4.0 points" that did not
/// survive contact with a second fixture: every delta it quoted was a single run of one build
/// against a single run of another, and all of them were inside the band above. No numeric
/// "is it evidence" rule is offered here, because an earlier version of this comment gave one and it
/// was derived from the same too-narrow sample it was meant to protect against.
#[cfg(test)]
mod adaptivity {
    use super::*;
    use std::collections::HashSet;

    /// Counter snapshot, so each phase can be reported as a delta rather than a running total.
    #[derive(Clone, Copy)]
    struct Snap {
        hits: usize,
        builds: usize,
        refusals: usize,
        bytes: usize,
        peak_run: usize,
        sweeps: usize,
        evictions: usize,
    }

    impl Snap {
        fn of(c: &BloomFilterCache) -> Self {
            Self {
                hits: c.cache_hits(),
                builds: c.filters_built(),
                refusals: c.admissions_refused(),
                bytes: c.cached_bytes(),
                peak_run: c.peak_miss_run(),
                sweeps: c.sweeps_run(),
                evictions: c.entries_evicted(),
            }
        }

        fn since(self, before: Self) -> (usize, usize, usize) {
            (
                self.hits - before.hits,
                self.builds - before.builds,
                self.refusals - before.refusals,
            )
        }
    }

    /// Ratios here are printed for a human to read, never compared or asserted, so the precision
    /// loss on the casts does not matter.
    #[allow(clippy::cast_precision_loss, reason = "diagnostic output only")]
    fn report(label: &str, before: Snap, after: Snap) {
        let (hits, builds, refusals) = after.since(before);
        let probes = hits + builds;
        let rate = if probes == 0 {
            f64::NAN
        } else {
            hits as f64 / probes as f64 * 100.0
        };
        let refused_share = if builds == 0 {
            f64::NAN
        } else {
            refusals as f64 / builds as f64 * 100.0
        };
        let resident_mb = after.bytes as f64 / (1024.0 * 1024.0);
        // `peak_run` is whatever the per-walk sampling has not already drained, so it reads 0 in the
        // thrash arm and the per-walk list is the number to look at. Printed regardless, because the
        // first attempt at #40 justified a threshold against a run length it never measured at all,
        // and adding this one line falsified three separate claims immediately.
        println!(
            "{label:28} probes={probes:>8}  hits={hits:>8}  builds={builds:>8}  \
             hit_rate={rate:>5.1}%  refused/build={refused_share:>5.1}%  \
             resident={resident_mb:.1}MB  peak_miss_run={:>7}  sweeps={:>4}  evicted={:>8}",
            after.peak_run,
            after.sweeps - before.sweeps,
            after.evictions - before.evictions
        );
    }

    /// The walker thread count these numbers were produced under.
    ///
    /// Printed with every run because miss-run length is a function of it — `miss_run` is one
    /// counter for the whole walk, so any thread's hit ends the run for all of them. Two people
    /// measuring the same cache behaviour on different machines get different miss runs, and the
    /// default is machine-dependent. An unlabelled miss-run number is not a measurement.
    ///
    /// Mirrors the resolution in `search::walker`; set `TILTH_THREADS=1` for reproducible runs.
    fn walk_threads() -> usize {
        std::env::var("TILTH_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(|| {
                std::thread::available_parallelism().map_or(4, |n| (n.get() / 2).clamp(2, 6))
            })
    }

    fn targets() -> HashSet<String> {
        ["IsValid", "Initialize", "Reset", "Update", "Serialize"]
            .into_iter()
            .map(String::from)
            .collect()
    }

    fn env_tree() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
        let root = std::env::var_os("TILTH_ADAPTIVITY_ROOT")?;
        let sub = std::env::var_os("TILTH_ADAPTIVITY_SUBTREE")?;
        Some((
            std::path::PathBuf::from(root),
            std::path::PathBuf::from(sub),
        ))
    }

    /// The #40 regime: one broad query fills the budget, then the agent narrows to one subtree for
    /// the next N calls.
    ///
    /// The **control** is the point. A low hit rate in the narrow phase means nothing on its own —
    /// it could just as easily mean the subtree has few files probed more than once. So the same
    /// narrow sequence also runs against a *fresh* cache, which shows what hit rate the sequence
    /// can reach when the budget has not already been spent on unrelated files. The gap between
    /// them is the cost of the cache not adapting; if there is no gap, #40 is not a real problem
    /// and the honest outcome is to document it and close.
    ///
    /// Since the fix there is a second control: the same poisoned sequence with sweeping
    /// **suppressed**, which is what this cache did before it could evict. See `no_sweeps` for how
    /// that is arranged without a test-only switch. Three arms, then: refusal, adaptive, and the
    /// fresh-cache upper bound.
    #[test]
    #[ignore = "needs a tree large enough to fill the 32 MB ceiling; see module docs"]
    #[allow(clippy::cast_precision_loss, reason = "diagnostic output only")]
    fn adaptivity_broad_then_narrow_versus_fresh() {
        let Some((root, sub)) = env_tree() else {
            println!(
                "SKIPPED (no TILTH_ADAPTIVITY_ROOT / TILTH_ADAPTIVITY_SUBTREE) — measured nothing"
            );
            return;
        };
        println!("walk threads: {}", walk_threads());
        let narrow_calls: usize = std::env::var("TILTH_ADAPTIVITY_CALLS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);
        let t = targets();

        // --- Arm R: the pre-fix behaviour — poisoned, and never allowed to sweep ---
        let refusing = BloomFilterCache::new();
        let suppress = no_sweeps(&refusing);
        let r0 = Snap::of(&refusing);
        let _ = crate::search::callers::find_callers_batch(&t, &root, &refusing, None);
        let r1 = Snap::of(&refusing);
        report("R broad (fills budget)", r0, r1);
        for i in 0..narrow_calls {
            let before = Snap::of(&refusing);
            let _ = crate::search::callers::find_callers_batch(&t, &sub, &refusing, None);
            report(&format!("R narrow #{i}"), before, Snap::of(&refusing));
        }
        let r_end = Snap::of(&refusing);
        report("R narrow total", r1, r_end);
        assert_eq!(
            r_end.sweeps, 0,
            "the suppressed arm swept, so it is not measuring refusal"
        );
        drop(suppress);

        // --- Arm A: the same shape, sweeping live ---
        let poisoned = BloomFilterCache::new();
        let a0 = Snap::of(&poisoned);
        let _ = crate::search::callers::find_callers_batch(&t, &root, &poisoned, None);
        let a1 = Snap::of(&poisoned);
        report("A broad (fills budget)", a0, a1);

        for i in 0..narrow_calls {
            let before = Snap::of(&poisoned);
            let _ = crate::search::callers::find_callers_batch(&t, &sub, &poisoned, None);
            let after = Snap::of(&poisoned);
            report(&format!("A narrow #{i}"), before, after);
        }
        let a_end = Snap::of(&poisoned);
        report("A narrow total", a1, a_end);

        // --- Arm B: the same narrow sequence against a cache that was never poisoned ---
        let fresh = BloomFilterCache::new();
        let b0 = Snap::of(&fresh);
        for i in 0..narrow_calls {
            let before = Snap::of(&fresh);
            let _ = crate::search::callers::find_callers_batch(&t, &sub, &fresh, None);
            let after = Snap::of(&fresh);
            report(&format!("B narrow #{i}"), before, after);
        }
        let b_end = Snap::of(&fresh);
        report("B narrow total", b0, b_end);

        let rr = hit_rate(r1, r_end);
        let ar = hit_rate(a1, a_end);
        let br = hit_rate(b0, b_end);
        println!(
            "\nnarrow-phase hit rate: refusing {rr:.1}%  adaptive {ar:.1}%  fresh {br:.1}%  \
             (adaptive recovers {:.1} of the {:.1} points refusal gave up)",
            ar - rr,
            br - rr
        );
        println!(
            "if the refusing/fresh gap is ~0, a full cache costs nothing here and #40 should be \
             documented, not fixed"
        );
        println!(
            "note: at the default 5 calls the resident column is mid-adaptation. #104's reclaim \
             needs QUIET_SWEEPS_BEFORE_RECLAIM quiet passes before it releases a byte — set \
             TILTH_ADAPTIVITY_CALLS=30 to see resident return to the fresh-cache figure"
        );
    }

    /// Hit rate over the interval between two snapshots, as a percentage.
    #[allow(clippy::cast_precision_loss, reason = "diagnostic output only")]
    fn hit_rate(before: Snap, after: Snap) -> f64 {
        let (hits, builds, _) = after.since(before);
        hits as f64 / (hits + builds).max(1) as f64 * 100.0
    }

    /// Hold sweeping off for as long as the returned guard lives.
    ///
    /// No test-only switch, and deliberately so: the arm that is supposed to reproduce the old
    /// behaviour has to run the *shipping* code path, or it measures a second implementation
    /// instead of the one under test. An outer pass that never closes is exactly the state
    /// `end_pass` refuses to sweep in — another walk is still in flight — so the whole sequence
    /// runs with admission-refusal as its only policy and every other line of the cache unchanged.
    fn no_sweeps(cache: &BloomFilterCache) -> PassGuard<'_> {
        cache.begin_pass()
    }

    /// The case that constrains the fix: a working set genuinely larger than the ceiling.
    ///
    /// Refusal degrades gracefully here — whatever got in keeps hitting, so the hit rate lands
    /// somewhere between 0% and the ceiling's share of the working set. A naive "clear when full"
    /// reset does **not**: it throws away a cache that was working, and if the clear lands near the
    /// end of each walk it can score worse than refusing ever did. So this number is the floor any
    /// reset policy has to beat, and the reason #40 leans toward hysteresis rather than a bare
    /// reset.
    ///
    /// Uses `with_ceiling` to make the working set exceed the budget cheaply, rather than needing a
    /// tree big enough to overflow 32 MB several times over.
    #[test]
    #[ignore = "needs TILTH_ADAPTIVITY_SUBTREE; see module docs"]
    #[allow(clippy::cast_precision_loss, reason = "diagnostic output only")]
    fn adaptivity_thrash_working_set_larger_than_ceiling() {
        // Only the subtree is used here, so do not demand the root as well — the earlier version
        // required both and then ignored one, which contradicted its own `#[ignore]` reason.
        let Some(sub) = std::env::var_os("TILTH_ADAPTIVITY_SUBTREE").map(std::path::PathBuf::from)
        else {
            println!("SKIPPED (no TILTH_ADAPTIVITY_SUBTREE set) — this run measured nothing");
            return;
        };
        let t = targets();

        // First find what this subtree actually needs, so the ceilings below are fractions of a
        // measured number rather than guesses.
        //
        // Effectively unbounded, not `new()`. With the default 32 MB ceiling this run would refuse
        // admissions on any subtree needing more than that, `needed` would come back clamped to
        // 32 MB, and every row below would silently be a fraction of the ceiling rather than of the
        // working set — including the "1.00x" row, which the type doc leans on as the case where
        // nothing is ever refused. The assertion makes the clamp impossible to miss if this is ever
        // pointed at a subtree that overflows `usize`-worth of filters.
        let sizing = BloomFilterCache::with_ceiling(usize::MAX / 2);
        let _ = crate::search::callers::find_callers_batch(&t, &sub, &sizing, None);
        let needed = sizing.cached_bytes();
        assert_eq!(
            sizing.admissions_refused(),
            0,
            "sizing run hit its ceiling, so `needed` is clamped and every row below is mislabelled"
        );
        println!(
            "subtree needs {:.2}MB to cache fully  (walk threads: {})",
            needed as f64 / (1024.0 * 1024.0),
            walk_threads()
        );

        // Ceiling as a fraction of the working set. 1.0 is the adequate case for reference.
        //
        // Each row runs twice — sweeping suppressed and live — because the noise on these rows is
        // wide enough (up to 15.9 points at 0.50x on one fixture) that a live number compared
        // against a table in a comment proves nothing. Paired within one run, against the same
        // tree in the same state, they answer the only question that matters here: did the policy
        // move this row at all?
        for frac in [0.25_f64, 0.5, 0.75, 1.0] {
            #[allow(clippy::cast_sign_loss, reason = "frac and needed are both positive")]
            #[allow(clippy::cast_possible_truncation, reason = "byte count fits usize")]
            let ceiling = (needed as f64 * frac) as usize;
            for sweeping in [false, true] {
                let cache = BloomFilterCache::with_ceiling(ceiling);
                let suppress = (!sweeping).then(|| no_sweeps(&cache));
                let b0 = Snap::of(&cache);
                // Sample the peak miss run per walk, not cumulatively. Walk 0 over a fresh cache
                // has no hits available to interrupt it, so its run is the whole probe count at
                // *every* ceiling — including a ceiling that fits the working set entirely.
                // Reading a running maximum therefore reports that constant and hides the
                // steady-state run, which is the number an adaptive trigger would actually have to
                // threshold on. Reporting both is what reconciles two measurements of "the miss
                // run" that disagreed by 6x.
                let mut runs = Vec::new();
                for _ in 0..5 {
                    let _ = crate::search::callers::find_callers_batch(&t, &sub, &cache, None);
                    runs.push(cache.take_peak_miss_run());
                }
                let b1 = Snap::of(&cache);
                let policy = if sweeping { "sweeping" } else { "refusing" };
                report(&format!("ceiling={frac:.2}x  {policy}"), b0, b1);
                println!("{:30} per-walk peak miss run: {runs:?}", "");
                drop(suppress);
            }
        }
        println!(
            "\nthe refusing rows are what refusal already achieves; a policy that scores below \
             them here is a regression, however well it does on the narrow case. Zero evictions \
             on the sweeping rows is the expected result, not a lucky one — every resident entry \
             is probed on every walk, so no sweep has anything to age."
        );
    }

    /// The regression case this policy invents, and the one the earlier attempts did not have:
    /// a workload that **alternates** between two scopes.
    ///
    /// The thrash test above oversubscribes the ceiling with a single repeated walk, where every
    /// resident entry is probed every time and no sweep can drop one. Alternation is the shape that
    /// can defeat "unprobed for a whole sweep", because each scope genuinely goes unprobed while
    /// the other is walked. `IDLE_SWEEPS_BEFORE_EVICTION` is what holds the line, and this is where
    /// the claim is measured on real trees rather than argued from a synthetic fixture.
    ///
    /// Needs a second subtree, disjoint from the first: `TILTH_ADAPTIVITY_SUBTREE_B`.
    #[test]
    #[ignore = "needs TILTH_ADAPTIVITY_SUBTREE and TILTH_ADAPTIVITY_SUBTREE_B; see module docs"]
    #[allow(clippy::cast_precision_loss, reason = "diagnostic output only")]
    fn adaptivity_alternating_scopes() {
        let (Some(a), Some(b)) = (
            std::env::var_os("TILTH_ADAPTIVITY_SUBTREE").map(std::path::PathBuf::from),
            std::env::var_os("TILTH_ADAPTIVITY_SUBTREE_B").map(std::path::PathBuf::from),
        ) else {
            println!(
                "SKIPPED (need TILTH_ADAPTIVITY_SUBTREE and TILTH_ADAPTIVITY_SUBTREE_B) — \
                 measured nothing"
            );
            return;
        };
        let t = targets();
        println!("walk threads: {}", walk_threads());

        // Size the pair together, then squeeze. At 0.5x neither scope can be fully resident, which
        // is the regime where an over-eager trigger ping-pongs.
        let sizing = BloomFilterCache::with_ceiling(usize::MAX / 2);
        let _ = crate::search::callers::find_callers_batch(&t, &a, &sizing, None);
        let _ = crate::search::callers::find_callers_batch(&t, &b, &sizing, None);
        let needed = sizing.cached_bytes();
        assert_eq!(
            sizing.admissions_refused(),
            0,
            "sizing run hit its ceiling, so every row below is mislabelled"
        );
        println!(
            "both scopes need {:.2}MB to cache fully",
            needed as f64 / (1024.0 * 1024.0)
        );

        for frac in [0.25_f64, 0.5, 0.75] {
            #[allow(clippy::cast_sign_loss, reason = "frac and needed are both positive")]
            #[allow(clippy::cast_possible_truncation, reason = "byte count fits usize")]
            let ceiling = (needed as f64 * frac) as usize;
            for sweeping in [false, true] {
                let cache = BloomFilterCache::with_ceiling(ceiling);
                let suppress = (!sweeping).then(|| no_sweeps(&cache));
                let start = Snap::of(&cache);
                for _ in 0..4 {
                    for scope in [&a, &b] {
                        let _ = crate::search::callers::find_callers_batch(&t, scope, &cache, None);
                    }
                }
                let policy = if sweeping { "sweeping" } else { "refusing" };
                report(
                    &format!("alternating {frac:.2}x  {policy}"),
                    start,
                    Snap::of(&cache),
                );
                drop(suppress);
            }
        }
        println!(
            "\nthe sweeping rows must not score below the refusing ones. If they do, the idle \
             bound is too tight for this alternation and the two scopes are clearing each other."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Content with roughly `n` distinct identifiers.
    fn ident_content(n: usize) -> String {
        (0..n).map(|i| format!("ident_{i} ")).collect()
    }

    /// The cache must stop growing at its ceiling, and must keep answering correctly once it
    /// does — a refused insert costs a rebuild, never an answer.
    ///
    /// Uses `with_ceiling` rather than the shipped 32 MB constant. That is not just speed:
    /// reaching 32 MB means tokenising ~28M identifiers, which made this one test 5.0s of a 5.7s
    /// suite. A small injected ceiling exercises identical logic in milliseconds.
    #[test]
    fn cache_stops_growing_at_the_ceiling_and_stays_correct() {
        let ceiling = 64 * 1024;
        let cache = BloomFilterCache::with_ceiling(ceiling);
        let mtime = SystemTime::UNIX_EPOCH;
        let content = ident_content(200);

        let mut admitted_at_least_one = false;
        for f in 0..400 {
            let path = PathBuf::from(format!("/synthetic/f{f}.rs"));
            // The only guarantee a Bloom filter makes is no false *negatives*, so this is the
            // only per-file assertion available. Asserting the absent case fails legitimately —
            // the filters target a 1% false-positive rate, and an earlier version of this test
            // did exactly that and tripped on the sixth file.
            assert!(
                cache.contains(&path, mtime, &content, "ident_7"),
                "a present symbol must be found regardless of cache admission (file {f})"
            );
            admitted_at_least_one |= cache.cached_bytes() > 0;
        }

        assert!(admitted_at_least_one, "nothing was ever admitted");
        let bytes = cache.cached_bytes();
        assert!(
            bytes <= ceiling,
            "cache exceeded its ceiling: {bytes} > {ceiling}"
        );
        // Tight, not `ceiling / 2`: the fixture overruns the bound many times over, so anything
        // materially below the ceiling means admission stopped early. A loose floor would miss a
        // cost computed several times too large.
        let one_entry = ceiling / 8;
        assert!(
            bytes + one_entry > ceiling,
            "cache filled only to {bytes} of {ceiling}; admission stopped too early"
        );
    }

    /// The refusal branch — a stale entry whose replacement does not fit — must leave the counter
    /// consistent with the map, and must not keep an entry it has already un-charged.
    ///
    /// Untestable before the ceiling was injectable, which is why it had no coverage.
    #[test]
    fn stale_entry_that_cannot_be_replaced_is_dropped_and_uncharged() {
        let content = ident_content(200);
        // A ceiling that admits exactly one entry, so a second version cannot fit alongside it.
        let cache = BloomFilterCache::with_ceiling(64 * 1024);
        let filler = ident_content(200);
        // Fill the budget with other files first.
        for f in 0..400 {
            let _ = cache.contains(
                &PathBuf::from(format!("/synthetic/filler{f}.rs")),
                SystemTime::UNIX_EPOCH,
                &filler,
                "ident_1",
            );
        }
        let full = cache.cached_bytes();

        // Now churn a file that is already cached, so the stale path runs with no room.
        let path = PathBuf::from("/synthetic/filler0.rs");
        let newer = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1);
        assert!(cache.contains(&path, newer, &content, "ident_7"));

        assert!(
            cache.cached_bytes() <= full,
            "refusing a replacement must not increase the charge"
        );
        // Whatever happened, the ceiling still holds and answers are still correct.
        assert!(cache.cached_bytes() <= 64 * 1024);
        assert!(cache.contains(&path, newer, &content, "ident_7"));
    }

    /// A probe with N targets must build one filter per file, not one per target — including when
    /// the cache is full and refuses to admit it.
    ///
    /// This is the #34 regression guard. Before `contains_any`, `read_with_bloom_check` looped
    /// `contains` per target: with an unbounded cache target 1 built and 2..N hit, but once the
    /// ceiling landed in #32 a refused admission became the steady state and all N built the
    /// identical filter. The build count is asserted rather than the wall time because the
    /// difference is a constant factor, and a timing assertion at that resolution is flaky.
    #[test]
    fn a_refused_admission_builds_one_filter_per_file_not_per_target() {
        // Ceiling of 0 refuses every admission, which is exactly the full-cache steady state the
        // multiplier lived in — and it makes the test independent of how big a filter happens to
        // be. `cache_stops_growing_at_the_ceiling_and_stays_correct` covers the partial case.
        let cache = BloomFilterCache::with_ceiling(0);
        let mtime = SystemTime::UNIX_EPOCH;
        let content = ident_content(200);
        let path = PathBuf::from("/synthetic/refused.rs");

        // Five targets, none present, so `any` cannot short-circuit and every target is queried.
        // A present target would let `any` return on the first one and hide the fan-out.
        let targets = ["absent_a", "absent_b", "absent_c", "absent_d", "absent_e"];
        // The answer is not assertable either way: at a 1% target FPR over five probes a `true`
        // here is legitimate. Correctness of the answer is covered by the present-symbol tests;
        // what this test pins down is that refusing admission did not turn one build into five.
        let _ = cache.contains_any(&path, mtime, &content, targets);

        assert_eq!(
            cache.filters_built(),
            1,
            "a {}-target probe built {} filters for one file; the build is back inside the \
             per-target loop",
            targets.len(),
            cache.filters_built()
        );
        assert_eq!(
            cache.cached_bytes(),
            0,
            "a 0-byte ceiling admitted an entry"
        );

        // And the count still does not scale with N across repeated refused probes: each probe
        // costs exactly one more build, not one per target.
        for _ in 0..3 {
            let _ = cache.contains_any(&path, mtime, &content, targets);
        }
        assert_eq!(
            cache.filters_built(),
            4,
            "four refused probes should cost four builds"
        );
    }

    /// A cached filter answers every target with no build at all — the property that made the
    /// per-target loop harmless before the cache was bounded, and which must survive the change.
    #[test]
    fn a_cached_filter_answers_many_targets_without_rebuilding() {
        let cache = BloomFilterCache::with_ceiling(16 * 1024 * 1024);
        let mtime = SystemTime::UNIX_EPOCH;
        let content = ident_content(200);
        let path = PathBuf::from("/synthetic/admitted.rs");

        assert!(cache.contains_any(&path, mtime, &content, ["ident_7"]));
        assert_eq!(cache.filters_built(), 1);
        assert!(cache.cached_bytes() > 0, "nothing was admitted");

        // A present target last, so the whole set is walked before `any` returns.
        assert!(cache.contains_any(
            &path,
            mtime,
            &content,
            ["absent_a", "absent_b", "absent_c", "absent_d", "ident_7"]
        ));
        assert_eq!(
            cache.filters_built(),
            1,
            "querying a cached filter rebuilt it"
        );
    }

    /// An empty target set is vacuously false and must not pay for a filter, matching `any` on an
    /// empty iterator.
    ///
    /// Not reachable from any caller today — every batch entry point guards an empty set upstream
    /// (`resolve_callees` returns early, and its import loop breaks on `remaining.is_empty()`
    /// *before* the prefilter call; `blast_radius`, `diff` and `grok` likewise). So this is purely
    /// defensive, kept so `contains_any` stays faithful to `any` on an empty iterator, and
    /// asserted so it stays that way.
    #[test]
    fn no_targets_costs_no_build() {
        let cache = BloomFilterCache::with_ceiling(16 * 1024 * 1024);
        let empty: [&str; 0] = [];
        assert!(!cache.contains_any(
            Path::new("/synthetic/empty.rs"),
            SystemTime::UNIX_EPOCH,
            &ident_content(200),
            empty
        ));
        assert_eq!(cache.filters_built(), 0);
        assert_eq!(cache.cached_bytes(), 0);
    }

    /// `contains_any` must agree with per-target `contains` on the same filter, target for target.
    /// The optimisation is only sound if it changes cost and nothing else.
    ///
    /// Both runs share **one** cache, and `contains_any` goes first so it builds and admits. The
    /// per-target run then queries that same admitted filter, which is what makes the comparison
    /// about loop logic and nothing else. Two caches would compare two *differently seeded*
    /// filters: `fastbloom`'s `with_false_pos` derives a fresh SipHash key per filter instance, so
    /// filters built from identical content have different false-positive sets. Measured on this
    /// fixture — a 64-bit filter over 3 identifiers — two independent filters disagree on ~6 of
    /// 20000 absent probes, so the two-absent-target case would flake, and it would look like
    /// nondeterminism in a Bloom filter, the worst possible false alarm.
    #[test]
    fn contains_any_agrees_with_per_target_contains() {
        let content = "fn alpha() { beta(); }";
        let mtime = SystemTime::UNIX_EPOCH;
        let path = Path::new("/synthetic/agree.rs");
        let cache = BloomFilterCache::new();

        for targets in [
            vec!["alpha"],
            vec!["beta"],
            vec!["alpha", "beta"],
            vec!["nope_xyzzy", "alpha"],
            vec!["alpha", "nope_xyzzy"],
            vec!["nope_xyzzy", "nope_plugh"],
        ] {
            let via_any = cache.contains_any(path, mtime, content, targets.iter().copied());
            let via_each = targets
                .iter()
                .any(|t| cache.contains(path, mtime, content, t.as_ref()));
            assert_eq!(via_any, via_each, "disagreement on {targets:?}");
        }
        // One filter served every case: the first probe admitted it and nothing invalidated it.
        assert_eq!(cache.filters_built(), 1);
    }

    /// Two threads missing on the **same** path must charge the budget once, not twice.
    ///
    /// This is the test whose absence let a real bug ship in review: the accounting used to sit
    /// outside the `DashMap` shard lock, with the window spanning the whole of `build_filter`, so
    /// concurrent probes of one path each charged for it and the over-count was permanent. It is
    /// reachable in production — `edit::apply_batch` fans out with `into_par_iter` and each task
    /// reaches `find_callers_batch`, which runs a parallel walk against the shared cache.
    #[test]
    fn concurrent_probes_of_one_path_charge_the_budget_once() {
        use std::sync::{Arc, Barrier};

        let cache = Arc::new(BloomFilterCache::with_ceiling(16 * 1024 * 1024));
        let content = Arc::new(ident_content(4000));
        let path = PathBuf::from("/synthetic/contended.rs");
        let mtime = SystemTime::UNIX_EPOCH;

        let threads = 8;
        let barrier = Arc::new(Barrier::new(threads));
        let mut handles = Vec::new();
        for _ in 0..threads {
            let cache = Arc::clone(&cache);
            let content = Arc::clone(&content);
            let barrier = Arc::clone(&barrier);
            let path = path.clone();
            handles.push(std::thread::spawn(move || {
                // Align the threads so they all miss before any of them admits.
                barrier.wait();
                assert!(cache.contains(&path, mtime, &content, "ident_7"));
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        let charged = cache.cached_bytes();
        // Charge for exactly one entry. The old code charged up to `threads` times.
        let single = entry_bytes(&build_filter(&content, None));
        assert_eq!(
            charged, single,
            "{threads} concurrent probes of one path charged {charged} bytes for a {single}-byte              entry — the accounting is outside the shard lock again"
        );
    }

    // -----------------------------------------------------------------------
    // Pass-scoped eviction (#40)
    // -----------------------------------------------------------------------

    /// One probe of one synthetic file, as a walk would make it.
    fn probe(cache: &BloomFilterCache, path: &str, content: &str) {
        assert!(
            cache.contains(
                &PathBuf::from(path),
                SystemTime::UNIX_EPOCH,
                content,
                "ident_7"
            ),
            "a present symbol must be found regardless of admission ({path})"
        );
    }

    /// Bytes one entry of this content costs, so ceilings below are exact multiples of an entry
    /// rather than guesses that happen to admit "about" the intended number.
    fn cost_of(content: &str) -> usize {
        entry_bytes(&build_filter(content, None))
    }

    /// The #40 regime, in miniature: a broad pass fills the budget, then the workload narrows to a
    /// set that is disjoint from it. The resident set has to be released, or the narrow set is
    /// refused forever.
    ///
    /// The assertion is on the **last** narrow pass costing no builds at all. Under refusal it
    /// costs one per file, for ever, which is the 0.0% hit rate #40 measured on a real tree.
    ///
    /// The narrow scope is a **tenth** of the ceiling, which is the ratio that matters rather than
    /// an arbitrary small number: `EVICT_FREE_FRACTION` releases a tenth of the budget per sweep,
    /// so this is the widest incoming scope that still clears in one. It is also generous against
    /// production, where #40's own fixture wants 1.9 MB of a 32 MB ceiling — 6%. A first version
    /// used a scope half the ceiling and failed once the cap landed, which is the honest signal
    /// that a scope that large takes several sweeps rather than one.
    #[test]
    fn a_resident_set_the_workload_stopped_probing_is_released() {
        let content = ident_content(200);
        // Room for the broad set alone. The narrow set cannot fit alongside it, so it can only be
        // admitted if the broad set goes.
        let cache = BloomFilterCache::with_ceiling(cost_of(&content) * 200);

        {
            let _pass = cache.begin_pass();
            for f in 0..400 {
                probe(&cache, &format!("/broad/{f}.rs"), &content);
            }
        }
        assert!(
            cache.admissions_refused() > 0,
            "the broad pass never filled the budget, so nothing below is about a full cache"
        );
        assert_eq!(cache.resident_entries(), 200);

        // The agent narrows. Passes 0 and 1 age the broad set out, pass 2 admits the narrow set,
        // pass 3 is served entirely from cache.
        for _ in 0..3 {
            let _pass = cache.begin_pass();
            for f in 0..20 {
                probe(&cache, &format!("/narrow/{f}.rs"), &content);
            }
        }

        let before = (cache.filters_built(), cache.cache_hits());
        {
            let _pass = cache.begin_pass();
            for f in 0..20 {
                probe(&cache, &format!("/narrow/{f}.rs"), &content);
            }
        }
        let built = cache.filters_built() - before.0;
        let hit = cache.cache_hits() - before.1;
        assert_eq!(
            (built, hit),
            (0, 20),
            "the narrow set is still not resident: {built} builds and {hit} hits in a pass over \
             20 files the workload has probed four times"
        );
        assert!(
            cache.entries_evicted() >= 20,
            "only {} entries were released, so the narrow set cannot have displaced the broad one",
            cache.entries_evicted()
        );
        // What is *not* asserted here: that the broad set is gone. At this point it is not —
        // `EVICT_FREE_FRACTION` releases a tenth per sweep, the narrow set takes that tenth, and
        // from then on the pass refuses nothing. This test is about the hit rate, which is restored
        // in four passes. The rest of the dead set comes back later and on a different clock;
        // `a_dead_resident_set_is_reclaimed_once_the_workload_settles` is where that is asserted.
        assert_eq!(
            cache.cached_bytes(),
            cache.summed_entry_bytes(),
            "eviction lost track of the budget"
        );
    }

    /// The case the fix must **not** touch: a working set genuinely larger than the ceiling.
    ///
    /// Refusal degrades gracefully here — whatever got in keeps earning — and #40 ruled out three
    /// earlier triggers precisely because they cleared this cache. Every resident entry is probed
    /// on every pass, so no sweep may drop one. This is a property, not a measurement: the
    /// assertion is exactly zero evictions, not "not many".
    ///
    /// Neuter it by making the sweep clear unconditionally — the "generational reset" #40's body
    /// proposed. That is the policy this test exists to reject.
    #[test]
    fn a_working_set_larger_than_the_ceiling_is_never_evicted() {
        let content = ident_content(200);
        // Ceiling at a quarter of the working set — the worst row in #40's thrash table.
        let cache = BloomFilterCache::with_ceiling(cost_of(&content) * 10);

        for _ in 0..5 {
            let _pass = cache.begin_pass();
            for f in 0..40 {
                probe(&cache, &format!("/tree/{f}.rs"), &content);
            }
        }

        assert!(
            cache.sweeps_run() >= 4,
            "no sweep ran ({}), so this asserts nothing about sweeping",
            cache.sweeps_run()
        );
        assert_eq!(
            cache.entries_evicted(),
            0,
            "an overloaded cache was cleared; refusal already scores ~25% here and this policy \
             has to beat that, not reset it"
        );
        assert_eq!(cache.resident_entries(), 10);
        assert!(
            cache.cache_hits() >= 40,
            "the resident tenth stopped earning: {} hits over five passes",
            cache.cache_hits()
        );
    }

    /// Two scopes visited in alternation must not evict each other.
    ///
    /// This is what `IDLE_SWEEPS_BEFORE_EVICTION` is for, and the test fails at 1: a scope goes
    /// unprobed for exactly the pass its neighbour is being walked, so a one-pass trigger clears
    /// each of them just before it is needed. At 2 the idle count returns to 0 on the scope's own
    /// pass and never reaches the bound.
    ///
    /// The ceiling is below **each** scope, not merely below the two together. That is what makes
    /// the test discriminate, and a first version that only oversubscribed the pair did not: the
    /// idle count advances per *sweep*, and a sweep only runs on a pass that refused something, so
    /// with room to spare on one scope's pass the resident set's `probed` bits survived the round
    /// and even a one-pass trigger evicted nothing. Refusing on every pass is what puts a sweep at
    /// the end of every pass.
    #[test]
    fn scopes_visited_in_alternation_do_not_evict_each_other() {
        let content = ident_content(200);
        let cache = BloomFilterCache::with_ceiling(cost_of(&content) * 6);

        for round in 0..4 {
            for scope in ["a", "b"] {
                let _pass = cache.begin_pass();
                for f in 0..10 {
                    probe(&cache, &format!("/{scope}/{f}.rs"), &content);
                }
            }
            assert_eq!(
                cache.entries_evicted(),
                0,
                "round {round}: alternating scopes evicted each other's filters"
            );
        }
        assert!(
            cache.sweeps_run() >= 8,
            "only {} sweeps over 8 passes; the fixture stopped refusing and the alternation was \
             never tested",
            cache.sweeps_run()
        );
        assert!(
            cache.cache_hits() >= 18,
            "the resident scope stopped earning across the alternation: {} hits",
            cache.cache_hits()
        );
    }

    /// Hits and builds a rotation over `scopes` disjoint scopes produces, with sweeping on or off.
    ///
    /// The suppressed arm holds one extra pass open for the whole run, which is what `end_pass`
    /// refuses to sweep in — so it is the pre-#40 refuse-forever policy running the shipping code
    /// path rather than a second implementation of it. Deterministic: one thread, fixed order.
    fn rotation_hit_rate(scopes: usize, ceiling_entries: usize, sweeping: bool) -> (usize, usize) {
        let content = ident_content(200);
        let cache = BloomFilterCache::with_ceiling(cost_of(&content) * ceiling_entries);
        let suppress = (!sweeping).then(|| cache.begin_pass());
        for _ in 0..6 {
            for s in 0..scopes {
                let _pass = cache.begin_pass();
                for f in 0..10 {
                    probe(&cache, &format!("/s{s}/{f}.rs"), &content);
                }
            }
        }
        drop(suppress);
        (cache.cache_hits(), cache.filters_built())
    }

    /// A rotation over three or more disjoint scopes must not collapse.
    ///
    /// This is the regression an adversarial review found, and it was a cliff rather than a slope:
    /// with the sweep releasing everything that aged out at once, every scope's entries age
    /// together and each is dropped a pass or two before its turn, so **no probe ever hits**. Four
    /// scopes at a 12-entry ceiling scored 0.0% against refusal's 25.0%, and a ten-scope rotation
    /// at 0.70x the union — the same ceiling fraction the thrash table calls harmless — scored 0.0%
    /// against 58.3%.
    ///
    /// `EVICT_FREE_FRACTION` is what holds the line, by turning the clear into partial retention.
    /// The bound asserted here is three quarters of refusal, against a measured 20.8% versus 25.0%
    /// (0.83) on this fixture — loose enough not to pin an exact ratio, tight enough that the
    /// uncapped policy's 0.0% cannot pass. Remove the cap in `sweep` and this test fails.
    ///
    /// Two scopes are covered separately by `scopes_visited_in_alternation_do_not_evict_each_other`,
    /// which is the case that needs no cap at all.
    #[test]
    fn a_rotation_over_three_or_more_scopes_does_not_collapse() {
        for (scopes, ceiling) in [(3usize, 8usize), (4, 12), (4, 15), (5, 25), (10, 70)] {
            let (rh, rb) = rotation_hit_rate(scopes, ceiling, false);
            let (sh, sb) = rotation_hit_rate(scopes, ceiling, true);
            assert_eq!(
                rh + rb,
                sh + sb,
                "{scopes} scopes / {ceiling} entries: the two arms did not make the same probes"
            );
            assert!(
                rh > 0,
                "{scopes} scopes / {ceiling} entries: refusal itself scored 0, so there is no \
                 baseline to regress against"
            );
            assert!(
                sh * 4 >= rh * 3,
                "{scopes} scopes / {ceiling} entries: rotation collapsed — {sh} hits sweeping \
                 against {rh} refusing, of {} probes",
                rh + rb
            );
        }
    }

    /// #104: once the workload settles, the bytes it has stopped using must come back.
    ///
    /// #103 fixed the hit rate half of #40 and left this one. Its own test says so in as many
    /// words: the narrow set displaces a tenth of the broad set, every call after that hits and
    /// refuses nothing, so no further sweep runs and the rest of the dead set stays for the life of
    /// the process. On the real fixture that was ~28.8 MB of a 32 MB ceiling.
    ///
    /// Both halves are asserted, and the second is the one that makes this a fix rather than a
    /// trade: the resident set must fall to exactly the live scope, **and** the live scope must
    /// still be answering every probe from cache while that happens.
    #[test]
    fn a_dead_resident_set_is_reclaimed_once_the_workload_settles() {
        let content = ident_content(200);
        let cache = BloomFilterCache::with_ceiling(cost_of(&content) * 200);

        {
            let _pass = cache.begin_pass();
            for f in 0..400 {
                probe(&cache, &format!("/broad/{f}.rs"), &content);
            }
        }
        assert_eq!(cache.resident_entries(), 200, "the broad pass did not fill");

        // Long enough for `QUIET_SWEEPS_BEFORE_RECLAIM` plus `EVICT_FREE_FRACTION`'s release rate:
        // 16 sweeps before the first byte moves, then a tenth of the ceiling per sweep.
        let mut settled_at = None;
        for pass in 0..40 {
            {
                let _pass = cache.begin_pass();
                for f in 0..20 {
                    probe(&cache, &format!("/narrow/{f}.rs"), &content);
                }
            }
            if cache.resident_entries() == 20 && settled_at.is_none() {
                settled_at = Some(pass);
            }
        }
        assert!(
            settled_at.is_some(),
            "the dead set never drained: {} entries still resident after 40 settled passes",
            cache.resident_entries()
        );

        // And the live scope never stopped being served while that happened.
        let before = (cache.filters_built(), cache.cache_hits());
        {
            let _pass = cache.begin_pass();
            for f in 0..20 {
                probe(&cache, &format!("/narrow/{f}.rs"), &content);
            }
        }
        assert_eq!(
            (
                cache.filters_built() - before.0,
                cache.cache_hits() - before.1
            ),
            (0, 20),
            "the drain reclaimed the live scope along with the dead one"
        );
    }

    /// The drain must stop scanning once it has proved there is nothing left to reclaim.
    ///
    /// Otherwise every tool call for the rest of the process pays a full `retain` over the map for
    /// nothing. `drain_armed` is disarmed by a sweep that finds every resident entry probed, and
    /// only a pressured sweep can arm it again.
    #[test]
    fn a_drained_cache_stops_scanning() {
        let content = ident_content(200);
        let cache = BloomFilterCache::with_ceiling(cost_of(&content) * 200);
        {
            let _pass = cache.begin_pass();
            for f in 0..400 {
                probe(&cache, &format!("/broad/{f}.rs"), &content);
            }
        }
        for _ in 0..40 {
            let _pass = cache.begin_pass();
            for f in 0..20 {
                probe(&cache, &format!("/narrow/{f}.rs"), &content);
            }
        }
        assert_eq!(cache.resident_entries(), 20, "the fixture did not drain");

        let swept = cache.sweeps_run();
        for _ in 0..10 {
            let _pass = cache.begin_pass();
            for f in 0..20 {
                probe(&cache, &format!("/narrow/{f}.rs"), &content);
            }
        }
        assert_eq!(
            cache.sweeps_run(),
            swept,
            "a fully drained cache is still scanning the map on every pass"
        );
    }

    /// A cache with room to spare must never sweep, however long the session runs.
    ///
    /// The gate is `pass_refusals`: with no refusal there is no admission waiting on these bytes,
    /// so ageing an entry buys nothing and costs a rebuild. Without the gate this session — three
    /// passes over three disjoint file sets, all of which fit — would age out and evict the first
    /// set for no reason at all.
    #[test]
    fn a_cache_with_room_to_spare_never_sweeps() {
        let content = ident_content(200);
        let cache = BloomFilterCache::with_ceiling(cost_of(&content) * 100);

        for set in 0..3 {
            let _pass = cache.begin_pass();
            for f in 0..10 {
                probe(&cache, &format!("/set{set}/{f}.rs"), &content);
            }
        }

        assert_eq!(cache.admissions_refused(), 0, "the fixture did not fit");
        assert_eq!(cache.sweeps_run(), 0, "a cache with room to spare swept");
        assert_eq!(cache.resident_entries(), 30);
    }

    /// Refusals from an **abandoned** pass must not arm the next pass's sweep.
    ///
    /// Review found this: `end_pass` used to return on the abandon path before clearing the count,
    /// so a cancelled walk handed its refusals forward and the next pass swept whether or not its
    /// own budget was ever under pressure. The window is the whole gap between two requests.
    #[test]
    fn an_abandoned_pass_does_not_arm_the_next_ones_sweep() {
        let content = ident_content(200);
        let cache = BloomFilterCache::with_ceiling(cost_of(&content) * 5);

        // A pass that refuses, then abandons — a cancelled walk.
        let pass = cache.begin_pass();
        for f in 0..10 {
            probe(&cache, &format!("/f{f}.rs"), &content);
        }
        assert!(cache.admissions_refused() > 0, "the fixture did not refuse");
        pass.abandon();
        assert_eq!(cache.sweeps_run(), 0, "an abandoned pass swept");

        // A pass with nothing to admit. It is owed no sweep of its own.
        {
            let _pass = cache.begin_pass();
            for f in 0..5 {
                probe(&cache, &format!("/f{f}.rs"), &content);
            }
        }
        assert_eq!(
            cache.sweeps_run(),
            0,
            "the abandoned pass's refusals armed a sweep on a pass that refused nothing"
        );
    }

    /// Refusals recorded with **no pass open** must not arm a sweep either.
    ///
    /// Also from review. `admit` is reachable outside any pass — `callees::resolve_callees` probes
    /// a file's imports through the same prefilter and opens none — so without the guard in
    /// `note_refusal` a session that had ever refused a callees probe swept on the next callers
    /// walk regardless of that walk's own budget. `a_cache_with_room_to_spare_never_sweeps` held in
    /// its fixture and did not hold in production.
    #[test]
    fn refusals_outside_a_pass_do_not_arm_a_sweep() {
        let content = ident_content(200);
        let cache = BloomFilterCache::with_ceiling(cost_of(&content) * 5);

        // No pass: the shape `resolve_callees` produces.
        for f in 0..10 {
            probe(&cache, &format!("/f{f}.rs"), &content);
        }
        assert!(cache.admissions_refused() > 0, "the fixture did not refuse");

        {
            let _pass = cache.begin_pass();
            for f in 0..5 {
                probe(&cache, &format!("/f{f}.rs"), &content);
            }
        }
        assert_eq!(
            cache.sweeps_run(),
            0,
            "refusals from outside any pass armed a sweep on a pass that refused nothing"
        );
    }

    /// A nested pass must not sweep — the outer walk is still probing, so its untouched files are
    /// untouched only because it has not reached them yet.
    ///
    /// Nesting itself is not reachable in production — every driver (`analyze_deps`, `grok`,
    /// `blast_radius`, `diff`) calls `find_callers_batch` sequentially. **Overlap** is:
    /// `edit::apply_batch` fans out with `into_par_iter` and several tasks reach the walk at once
    /// against the one process-lifetime cache. `active_passes` is one counter serving both, so this
    /// pins it in the shape a single-threaded test can express deterministically; the concurrent
    /// shape is covered by `a_sweep_cannot_desynchronise_cached_bytes_from_the_map`.
    #[test]
    fn a_nested_pass_does_not_sweep_before_the_outer_one_closes() {
        // Every admission refused, so a sweep is always due on the refusal gate and the count is
        // about pass nesting and nothing else.
        let cache = BloomFilterCache::with_ceiling(0);
        let content = ident_content(200);

        let outer = cache.begin_pass();
        {
            let _inner = cache.begin_pass();
            probe(&cache, "/nested/a.rs", &content);
        }
        assert_eq!(
            cache.sweeps_run(),
            0,
            "the inner pass swept while the outer walk was still running"
        );
        drop(outer);
        assert_eq!(cache.sweeps_run(), 1, "the outer close did not sweep");
    }

    /// A sweep running against concurrent admissions must leave `cached_bytes` equal to what the
    /// map actually holds.
    ///
    /// The shape asked for by `concurrent_probes_of_one_path_charge_the_budget_once`, applied to
    /// the other direction of the same hazard. Eviction un-charges, so it can desynchronise the
    /// counter the way double-billing did — and the failure mode is the same: a permanently wrong
    /// budget rather than a crash, leaving the cache strictly worse than no cache with nothing to
    /// show why. `retain` calls its closure under the shard write lock, which is what keeps the
    /// arithmetic on the right side of the lock; this is the assertion that it stays there.
    #[test]
    fn a_sweep_cannot_desynchronise_cached_bytes_from_the_map() {
        use std::sync::{Arc, Barrier};

        let content = Arc::new(ident_content(200));
        // Small enough that admissions are refused throughout, so every pass close is due a sweep.
        let cache = Arc::new(BloomFilterCache::with_ceiling(cost_of(&content) * 12));

        // Two roles, and exactly **one** walker, because a fleet of identical pass-holders tests
        // nothing: only the close that takes `active_passes` to zero may sweep, so overlapping
        // passes suppress each other and the run ends with one sweep that raced nothing. Measured
        // — two walkers produced 2 sweeps and 0 evictions across 400 rounds, and the test failed
        // its own precondition about one run in eight. One walker sweeps on every close, so the
        // race under test — a sweep un-charging while six other threads admit — runs hundreds of
        // times per execution.
        //
        // What this does *not* cover is two sweeps at once. That needs thread X to observe the
        // 1 -> 0 transition and begin sweeping while thread Y opens and closes a whole pass inside
        // that window; only one thread can observe any given decrement, so the window is a sweep
        // long and there is no way to hold it open from a test. The `sweeping` CAS is what makes
        // it safe, and it is reasoned about rather than measured — stated because an untested
        // guard should be labelled as one.
        let probers = 6;
        let walkers = 1;
        let barrier = Arc::new(Barrier::new(probers + walkers));
        let mut handles = Vec::new();

        for t in 0..probers {
            let cache = Arc::clone(&cache);
            let content = Arc::clone(&content);
            let barrier = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                for round in 0..200 {
                    // Two shared paths, which contend on one shard entry and stay hot, and eight
                    // one-shot paths, which keep the budget moving and are exactly what a sweep
                    // has to age out.
                    probe(&cache, "/shared/a.rs", &content);
                    probe(&cache, "/shared/b.rs", &content);
                    for f in 0..8 {
                        probe(&cache, &format!("/t{t}/r{round}/{f}.rs"), &content);
                    }
                }
            }));
        }

        for w in 0..walkers {
            let cache = Arc::clone(&cache);
            let content = Arc::clone(&content);
            let barrier = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                for round in 0..200 {
                    let _pass = cache.begin_pass();
                    probe(&cache, &format!("/w{w}/r{round}.rs"), &content);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert!(
            cache.sweeps_run() > 0 && cache.entries_evicted() > 0,
            "no sweep evicted anything ({} sweeps, {} evictions), so the race was never run",
            cache.sweeps_run(),
            cache.entries_evicted()
        );
        assert_eq!(
            cache.cached_bytes(),
            cache.summed_entry_bytes(),
            "the counter and the map disagree after concurrent sweeps and admissions"
        );
    }

    #[test]
    fn test_basic_membership() {
        let mut bf = BloomFilter::with_false_pos(0.01).expected_items(100);
        bf.insert("foo");
        bf.insert("bar");
        bf.insert("baz");

        assert!(bf.contains("foo"));
        assert!(bf.contains("bar"));
        assert!(bf.contains("baz"));
    }

    #[test]
    fn extracts_identifiers_across_rust_lifetimes() {
        let src = "fn longest<'a>(x: &'a str, y: &'a str) -> &'a str { x }";
        let idents: Vec<&str> = extract_identifiers(src, Some(Lang::Rust)).collect();
        for want in ["fn", "longest", "x", "y", "str"] {
            assert!(
                idents.contains(&want),
                "lifetime tick swallowed identifier {want:?}; got {idents:?}"
            );
        }
    }

    #[test]
    fn char_literal_is_still_skipped() {
        let src = "let c = 'a'; let d = '\\n'; fn target() {}";
        let idents: Vec<&str> = extract_identifiers(src, Some(Lang::Rust)).collect();
        assert!(idents.contains(&"target"), "got {idents:?}");
        assert!(
            !idents.contains(&"a"),
            "char-literal body leaked: {idents:?}"
        );
    }

    #[test]
    fn non_rust_single_quote_string_does_not_swallow_following_idents() {
        // In JS/Python/Ruby/PHP a `'...'` is a string, not a Rust lifetime. The
        // lifetime heuristic must stay off for them: if it fired, the closing
        // quote of `'foo'` would open a spurious string that swallows every
        // identifier up to the next quote -- a Bloom false negative.
        let src = "let x = 'foo'; bar();";
        let idents: Vec<&str> = extract_identifiers(src, Some(Lang::JavaScript)).collect();
        assert!(
            idents.contains(&"bar"),
            "closing quote opened a swallowing string: {idents:?}"
        );
        assert!(idents.contains(&"let"), "got {idents:?}");
        assert!(idents.contains(&"x"), "got {idents:?}");
    }

    #[test]
    fn test_definitely_not_present() {
        let mut bf = BloomFilter::with_false_pos(0.01).expected_items(10);
        bf.insert("alpha");
        bf.insert("beta");
        bf.insert("gamma");

        // With only 3 items in a filter sized for 10 at 1% FPR,
        // these should almost certainly return false.
        let mut false_positives = 0;
        let test_items = [
            "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu",
            "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
        ];
        for item in &test_items {
            if bf.contains(item) {
                false_positives += 1;
            }
        }
        // At most 1 false positive out of 21 items is generous
        assert!(
            false_positives <= 1,
            "too many false positives: {false_positives}/{}",
            test_items.len()
        );
    }

    #[test]
    fn test_false_positive_rate() {
        let n = 500;
        let mut bf = BloomFilter::with_false_pos(0.01).expected_items(n);

        // Insert N items
        for i in 0..n {
            bf.insert(&format!("item_{i}"));
        }

        // Verify all inserted items are found
        for i in 0..n {
            assert!(bf.contains(&format!("item_{i}")), "missing item_{i}");
        }

        // Test M random items that were NOT inserted
        let m = 10_000;
        let mut false_positives = 0;
        for i in 0..m {
            if bf.contains(&format!("notinserted_{i}")) {
                false_positives += 1;
            }
        }

        let fpr = f64::from(false_positives) / f64::from(m);
        // Target is 1%, allow up to 5% for statistical variance
        assert!(
            fpr < 0.05,
            "false positive rate too high: {fpr:.4} ({false_positives}/{m})"
        );
    }

    #[test]
    fn test_identifier_extraction() {
        let code = "fn foo(bar: Baz) { qux() }";
        let idents: Vec<&str> = extract_identifiers(code, Some(Lang::Rust)).collect();
        assert_eq!(idents, vec!["fn", "foo", "bar", "Baz", "qux"]);
    }

    #[test]
    fn test_identifier_extraction_skips_strings() {
        let code = r#"let x = "hello world"; let y = 42;"#;
        let idents: Vec<&str> = extract_identifiers(code, Some(Lang::Rust)).collect();
        assert!(idents.contains(&"let"));
        assert!(idents.contains(&"x"));
        assert!(idents.contains(&"y"));
        // "hello" and "world" are inside a string -- should be skipped
        assert!(!idents.contains(&"hello"));
        assert!(!idents.contains(&"world"));
    }

    #[test]
    fn test_identifier_extraction_skips_comments() {
        let code = "fn real() // fn fake()\n/* fn also_fake() */\nfn another()";
        let idents: Vec<&str> = extract_identifiers(code, Some(Lang::Rust)).collect();
        assert!(idents.contains(&"real"));
        assert!(idents.contains(&"another"));
        assert!(!idents.contains(&"fake"));
        assert!(!idents.contains(&"also_fake"));
    }

    #[test]
    fn test_identifier_extraction_underscores_and_numbers() {
        let code = "_private __dunder var_123 _0 a1b2c3";
        let idents: Vec<&str> = extract_identifiers(code, Some(Lang::Rust)).collect();
        assert_eq!(
            idents,
            vec!["_private", "__dunder", "var_123", "_0", "a1b2c3"]
        );
    }

    #[test]
    fn test_identifier_extraction_empty() {
        let idents: Vec<&str> = extract_identifiers("", Some(Lang::Rust)).collect();
        assert!(idents.is_empty());
    }

    #[test]
    fn test_identifier_extraction_no_identifiers() {
        let idents: Vec<&str> = extract_identifiers("123 + 456 = 789", Some(Lang::Rust)).collect();
        assert!(idents.is_empty());
    }

    #[test]
    fn test_cache_mtime_invalidation() {
        let cache = BloomFilterCache::new();
        let path = Path::new("/tmp/test_bloom.rs");

        let old_content = "fn old_function() {}";
        let new_content = "fn new_function() {}";

        let mtime_old = SystemTime::UNIX_EPOCH;
        let mtime_new = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1);

        // Cache with old content
        assert!(cache.contains(path, mtime_old, old_content, "old_function"));
        assert!(!cache.contains(path, mtime_old, old_content, "new_function"));

        // Same mtime: should use cached filter (old content), even though
        // we pass new content -- the cache trusts the mtime.
        assert!(cache.contains(path, mtime_old, new_content, "old_function"));

        // Different mtime: should rebuild from new content
        assert!(cache.contains(path, mtime_new, new_content, "new_function"));
        assert!(!cache.contains(path, mtime_new, new_content, "old_function"));
    }

    #[test]
    fn test_identifier_extraction_escaped_strings() {
        let code = r#"let s = "escaped \"quote\" inside"; let t = 1;"#;
        let idents: Vec<&str> = extract_identifiers(code, Some(Lang::Rust)).collect();
        assert!(idents.contains(&"s"));
        assert!(idents.contains(&"t"));
        // "quote" and "inside" are inside the string -- should be skipped
        assert!(!idents.contains(&"quote"));
        assert!(!idents.contains(&"inside"));
    }

    #[test]
    fn test_identifier_extraction_single_quotes() {
        let code = "let c = 'a'; let d = 'b';";
        let idents: Vec<&str> = extract_identifiers(code, Some(Lang::Rust)).collect();
        assert!(idents.contains(&"let"));
        assert!(idents.contains(&"c"));
        assert!(idents.contains(&"d"));
    }

    #[test]
    fn test_build_filter_integration() {
        let content = "pub fn search(query: &str) -> Vec<Match> { find(query) }";
        let filter = build_filter(content, Some(Lang::Rust));

        assert!(filter.contains("search"));
        assert!(filter.contains("query"));
        assert!(filter.contains("Vec"));
        assert!(filter.contains("Match"));
        assert!(filter.contains("find"));
        assert!(!filter.contains("nonexistent_symbol_xyz"));
    }
}
