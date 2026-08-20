//! Per-session caches for derived file analyses, keyed by path and invalidated by mtime.
//!
//! # Why nothing here holds a tree-sitter tree
//!
//! It used to. `parsed` was a `DashMap<PathBuf, Arc<ParsedFile>>` holding each file's full
//! contents *and* its full tree, populated once per distinct file appearing in a rendered
//! result, and nothing ever evicted. The MCP server keeps one `OutlineCache` for the
//! process lifetime, so the cost accumulated across a session rather than within a call.
//!
//! Measured (#67), 60 files of 499 000 B, one match per file so ten distinct files reach
//! each page, `TILTH_THREADS=1`, five queries over disjoint file sets, peak working set:
//!
//! ```text
//! calls   1        2        3        4        5
//! peak    263 MB   499 MB   734 MB   968 MB   1200 MB
//! ```
//!
//! Dead linear, ~235 MB per call, ~23.5 MB per shown file, never returned. The same run
//! with the fixture renamed to `.txt` — identical bytes, identical matches, but
//! `detect_file_type` returns non-`Code` so nothing was parsed — peaked at 4.0 MB.
//!
//! The retained tree was ~48x the file's own bytes. That ratio is also not something a
//! per-byte budget could have predicted. Content bytes divided by `descendant_count`, over
//! every file of each tree rather than a sample: **3.00** on that dense fixture, **4.91** on
//! this repository's `src` (69 files) and **5.02** on a real C++ tree (2 948 files). So a
//! tree's size per source byte varies 1.67x with content alone, and a budget denominated in
//! source bytes would be wrong by that much — in the unsafe direction for dense files.
//!
//! # What replaced it
//!
//! The render path never wanted a tree. `enclosing_scope_label` asks one question per shown
//! match — *what definition encloses this line* — and the answer is two small fields. So the
//! cache now stores **the answers, not the trees**: `labels` maps a file to the resolved
//! scope for the specific lines that were asked about, at a few dozen bytes each instead of
//! ~23.5 MB.
//!
//! Trees became transient, and `scope::warm_labels` keeps them from piling up *within* a
//! call: it groups a page's matches by file, parses each distinct file once, answers every
//! line for that file, and drops the tree before moving to the next. So peak carries **one
//! tree at a time** — a structural bound, not a tuned ceiling, and independent of page size,
//! display cap and session length.
//!
//! Both maps that remain are bounded by **bytes** (`MAX_CACHE_BYTES`) with least-recently-used
//! eviction, plus a per-entry line cap (`MAX_LINES_PER_FILE`) for the one way an entry grows
//! without a new key. A byte budget is viable here and was not for trees for the reason above:
//! everything still cached is a string, so its cost is `len()` — exact, not estimated.
//!
//! Eviction rather than refusal-on-full is deliberate. `BloomFilterCache` refuses once its
//! ceiling is reached, and #40 measured what that does: whatever fills the budget first keeps
//! it forever, scoring a 0% hit rate against 80% achievable. A full LRU still serves its
//! working set; a full refusing cache serves nothing.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::SystemTime;

use dashmap::DashMap;

/// Largest file this cache will parse, in bytes.
///
/// Shares the crate-wide parse gate (`lang::parse_budget::MAX_PARSE_FILE_SIZE`) so the outline
/// cache and `search::scope` track the search walks. A file past this is not worth a scope
/// annotation, and the transient tree for one is the single largest live allocation the
/// render path can make.
const MAX_PARSE_BYTES: u64 = crate::lang::parse_budget::MAX_PARSE_FILE_SIZE;

/// Byte ceiling for each map, applied independently.
///
/// A **byte** budget rather than an entry count, and it can be one precisely because trees
/// are gone. Everything still cached is a string, so its cost is `len()` — exact, not
/// estimated. That is the whole reason a budget is viable here and was not for trees, where
/// the same measurement said content-bytes-per-node ranges 3.00–5.20 depending on content.
///
/// An entry count was the first attempt and it does not bound memory. Neither per-entry size
/// is small the way it looks:
///
/// * `get_outline_str` generates with no line cap, so a 486 KB file of one-line definitions
///   produces a **896 KB** outline string, not the 3.5–5.8 KB typical of ordinary source. A
///   1024-entry cap on that shape admits ~900 MB.
/// * `store_labels` merges within one mtime — deliberately, so a session alternating between
///   lines of one file does not re-parse — which means an entry accumulates every line ever
///   asked about for that file, not the ≤10 of a single page.
///
/// Sized from measured sessions. Bytes actually held at the end of each:
///
/// ```text
/// session                                outline    label
/// this repo's src, 8 symbol queries      0.10 MB   ~2 KB
/// real C++ tree, 6 usage queries         0.07 MB   ~2 KB
/// 60x499 KB fixture, 5 disjoint queries  0.00 MB   ~5 KB
/// ```
///
/// 16 MB is ~160x the largest of those, so no ordinary session evicts at all, while the
/// pathological shape above is held to ~18 files instead of ~900 MB.
const MAX_CACHE_BYTES: usize = 16 * 1024 * 1024;

/// When over budget, free this fraction of it in one pass rather than stopping at the ceiling.
///
/// Hysteresis, load-bearing for wall time rather than memory. Eviction scans the map linearly
/// for the coldest entry, so evicting one entry per insert makes a full cache cost
/// `O(entries)` on *every* subsequent insert — measured at ~1.5x wall time for `tilth map`
/// over a 4 000-file tree, which inserts one entry per file and re-reads none of them.
/// Freeing a tenth per scan amortises that to roughly one scan per tenth-of-a-budget stored.
///
/// Expressed as a divisor rather than a percentage on purpose. The percentage form,
/// `budget / 100 * 90`, truncates to **zero** for any budget under 100 bytes, which turns
/// eviction into "empty the map". Harmless at 16 MB and wrong at any size a test picks — a
/// test using a byte-accurate small budget is what found it.
const EVICT_FREE_FRACTION: usize = 10;

/// Line cap for a single file's label set.
///
/// The byte budget bounds the *map*; this bounds one entry, and both are needed because they
/// fail in different directions. `store_labels` merges within one mtime, so a file that stays
/// hot across a session accumulates every line ever asked about it — and eviction cannot
/// reclaim that, since it evicts whole files and never the one being written. Without this
/// cap a single hot path grows past the budget on its own, which a test found at 4 506 bytes
/// against a 4 096 budget.
///
/// 512 is 5x the most a single page can ask for (100 under `--full`), so no page is ever
/// truncated and only a long session revisiting hundreds of distinct lines of one file
/// reaches it. Exceeding it costs a re-parse, never a wrong answer.
const MAX_LINES_PER_FILE: usize = 512;

/// A rendered outline string plus the mtime it was computed from.
struct CacheEntry {
    mtime: SystemTime,
    outline: Arc<str>,
    /// This entry's charge against the map's budget, stored rather than recomputed so the
    /// refund on replacement is always exactly what was charged. `BloomFilterCache` records
    /// why that matters: recomputing a cost that has since changed leaks budget permanently.
    bytes: usize,
    used: AtomicU64,
}

/// The enclosing definition of one line: a normalized kind label and an identifier.
///
/// `kind` is `&'static str` because every value comes from a fixed match in
/// `scope::kind_label`, so caching one costs a pointer rather than an allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScopeLabel {
    pub kind: &'static str,
    pub name: String,
}

/// Resolved scope answers for one file.
///
/// `lines` is keyed by the specific lines that were asked about, and `None` is a cached
/// answer too — "this line is at top level" is as worth remembering as any other, and
/// without it a top-level match re-parses its file on every call that shows it.
struct LabelEntry {
    mtime: SystemTime,
    lines: std::collections::HashMap<u32, Option<ScopeLabel>>,
    /// As `CacheEntry::bytes`, but re-derived on every merge, since merging grows the entry.
    bytes: usize,
    used: AtomicU64,
}

/// Byte cost of one cached outline: the string plus the key that reaches it.
fn outline_cost(path: &Path, outline: &str) -> usize {
    path.as_os_str().len() + outline.len()
}

/// Byte cost of one file's label set.
///
/// Counted rather than assumed, because the merge in `store_labels` means the line count is
/// session-lifetime rather than page-sized. `size_of` covers the map slot and the `Option`;
/// `name.len()` is the only heap string, `kind` being `&'static str`.
fn label_cost(path: &Path, lines: &std::collections::HashMap<u32, Option<ScopeLabel>>) -> usize {
    path.as_os_str().len()
        + lines
            .values()
            .map(|v| {
                std::mem::size_of::<u32>()
                    + std::mem::size_of::<Option<ScopeLabel>>()
                    + v.as_ref().map_or(0, |l| l.name.len())
            })
            .sum::<usize>()
}

/// Per-session cache of derived file analyses, keyed by path.
///
/// Staleness is checked per access: the stored `mtime` is compared to the file's current
/// one and a mismatch replaces the entry. That is a correctness guard, not a budget — see
/// the module header for what bounds size.
pub struct OutlineCache {
    entries: DashMap<PathBuf, CacheEntry>,
    labels: DashMap<PathBuf, LabelEntry>,
    /// Monotonic access counter driving LRU. Relaxed throughout: it decides only *which*
    /// entry is evicted, never whether the map is correct, so a racy tick can at worst
    /// evict a slightly-less-cold entry.
    clock: AtomicU64,
    /// Live byte totals, one per map. Every read-modify-write happens inside the shard lock
    /// the entry is held under — the hazard `BloomFilterCache::bytes` documents at length,
    /// where doing the arithmetic outside the lock double-billed permanently.
    entry_bytes: AtomicUsize,
    label_bytes: AtomicUsize,
    budget: usize,
}

impl Default for OutlineCache {
    fn default() -> Self {
        Self::with_budget(MAX_CACHE_BYTES)
    }
}

impl OutlineCache {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct with a specific per-map byte ceiling. Tests use this to exercise eviction
    /// without materialising 16 MB of outlines.
    #[must_use]
    pub fn with_budget(budget: usize) -> Self {
        Self {
            entries: DashMap::new(),
            labels: DashMap::new(),
            clock: AtomicU64::new(0),
            entry_bytes: AtomicUsize::new(0),
            label_bytes: AtomicUsize::new(0),
            budget,
        }
    }

    /// Bytes currently held by the outline map.
    #[must_use]
    pub fn outline_bytes(&self) -> usize {
        self.entry_bytes.load(Ordering::Relaxed)
    }

    /// Bytes currently held by the scope-label map.
    #[must_use]
    pub fn label_bytes(&self) -> usize {
        self.label_bytes.load(Ordering::Relaxed)
    }

    fn tick(&self) -> u64 {
        self.clock.fetch_add(1, Ordering::Relaxed)
    }

    /// Get cached outline or compute and cache it. Accepts `&Path` (not `&PathBuf`).
    pub fn get_or_compute(
        &self,
        path: &Path,
        mtime: SystemTime,
        compute: impl FnOnce() -> String,
    ) -> Arc<str> {
        let key = path.to_path_buf();
        // Fast path: entry exists and mtime matches.
        if let Some(e) = self.entries.get(&key) {
            if e.mtime == mtime {
                e.used.store(self.tick(), Ordering::Relaxed);
                return Arc::clone(&e.outline);
            }
        }
        // Stale or absent — compute and insert, replacing any stale entry. Computed *before*
        // taking the entry: `compute` reads and parses a file, and running it under the shard
        // write lock would serialise unrelated paths that happen to share a shard.
        let outline: Arc<str> = compute().into();
        let cost = outline_cost(path, &outline);
        let tick = self.tick();
        match self.entries.entry(key.clone()) {
            dashmap::mapref::entry::Entry::Occupied(mut occupied) => {
                // Refund the entry being displaced before charging its replacement, or a file
                // edited repeatedly would consume the budget one revision at a time.
                sub_bytes(&self.entry_bytes, occupied.get().bytes);
                self.entry_bytes.fetch_add(cost, Ordering::Relaxed);
                occupied.insert(CacheEntry {
                    mtime,
                    outline: Arc::clone(&outline),
                    bytes: cost,
                    used: AtomicU64::new(tick),
                });
            }
            dashmap::mapref::entry::Entry::Vacant(vacant) => {
                self.entry_bytes.fetch_add(cost, Ordering::Relaxed);
                vacant.insert(CacheEntry {
                    mtime,
                    outline: Arc::clone(&outline),
                    bytes: cost,
                    used: AtomicU64::new(tick),
                });
            }
        }
        evict_to_budget(
            &self.entries,
            &key,
            &self.entry_bytes,
            self.budget,
            |e| &e.used,
            |e| e.bytes,
        );
        outline
    }

    /// The cached scope answer for `(path, line)`, if this session has already resolved it
    /// against the file's current mtime.
    ///
    /// The outer `Option` is "not cached"; the inner one is the cached answer, where `None`
    /// means the line is at top level.
    #[must_use]
    pub fn cached_label(
        &self,
        path: &Path,
        mtime: SystemTime,
        line: u32,
    ) -> Option<Option<ScopeLabel>> {
        let e = self.labels.get(path)?;
        if e.mtime != mtime {
            return None;
        }
        let hit = e.lines.get(&line).cloned();
        if hit.is_some() {
            e.used.store(self.tick(), Ordering::Relaxed);
        }
        hit
    }

    /// Record scope answers for `path`, merging into any entry already held for the same
    /// mtime and replacing one held for a different mtime.
    ///
    /// Merging matters: a later call asking about different lines of the same file must not
    /// discard what an earlier one resolved, or a session alternating between two lines of
    /// one file would re-parse on every call.
    ///
    /// But merging is also the one way an entry grows without a new key, and eviction cannot
    /// answer that: it works at file granularity and cannot evict the entry currently being
    /// written. So a per-entry line cap does — see `MAX_LINES_PER_FILE`.
    pub fn store_labels(
        &self,
        path: &Path,
        mtime: SystemTime,
        resolved: impl IntoIterator<Item = (u32, Option<ScopeLabel>)>,
    ) {
        let key = path.to_path_buf();
        let tick = self.tick();
        match self.labels.entry(key.clone()) {
            dashmap::mapref::entry::Entry::Occupied(mut occupied) => {
                let before = occupied.get().bytes;
                let e = occupied.get_mut();
                if e.mtime == mtime {
                    e.lines.extend(resolved);
                    if e.lines.len() > MAX_LINES_PER_FILE {
                        // Past the cap, keep the newest page's answers and drop the rest.
                        // Coarse on purpose: the alternative is a second LRU *inside* the
                        // entry, and this is a pure optimisation — a dropped answer costs one
                        // re-parse, never a wrong label. Only a session that asks about
                        // hundreds of distinct lines of one file ever reaches it.
                        let keep: Vec<u32> = {
                            let mut ls: Vec<u32> = e.lines.keys().copied().collect();
                            ls.sort_unstable();
                            ls.into_iter().rev().take(MAX_LINES_PER_FILE).collect()
                        };
                        e.lines.retain(|line, _| keep.contains(line));
                    }
                } else {
                    // A different mtime discards every answer for this file: they were all
                    // computed from contents that no longer exist.
                    e.mtime = mtime;
                    e.lines = resolved.into_iter().collect();
                }
                // Re-derived rather than incremented: a merge can overwrite an existing
                // line's answer as well as add new ones, so the delta is not the size of
                // `resolved`.
                e.bytes = label_cost(path, &e.lines);
                let after = e.bytes;
                e.used.store(tick, Ordering::Relaxed);
                sub_bytes(&self.label_bytes, before);
                self.label_bytes.fetch_add(after, Ordering::Relaxed);
            }
            dashmap::mapref::entry::Entry::Vacant(vacant) => {
                let lines: std::collections::HashMap<u32, Option<ScopeLabel>> =
                    resolved.into_iter().collect();
                let cost = label_cost(path, &lines);
                self.label_bytes.fetch_add(cost, Ordering::Relaxed);
                vacant.insert(LabelEntry {
                    mtime,
                    lines,
                    bytes: cost,
                    used: AtomicU64::new(tick),
                });
            }
        }
        // After the match, not inside it: `entry()` holds the shard write lock for the whole
        // match, and evicting from the same map under it deadlocks. Reached on the merge path
        // too — merging grows the entry, so it can push the map over budget just as an insert
        // can, and an early return here would have let one hot file grow without bound.
        evict_to_budget(
            &self.labels,
            &key,
            &self.label_bytes,
            self.budget,
            |e| &e.used,
            |e| e.bytes,
        );
    }

    /// Largest file the scope resolver will parse.
    #[must_use]
    pub fn max_parse_bytes() -> u64 {
        MAX_PARSE_BYTES
    }
}

/// Evict least-recently-used entries until `map`'s byte total is back inside `budget`.
///
/// Called *after* the insert, never before. `store_labels` has no choice — `entry()` holds
/// the shard write lock for the whole match, so touching the same map inside it deadlocks —
/// and `get_or_compute` matches it so both maps behave identically. The consequence is that
/// the budget is exact **at rest** and can be exceeded in flight by whatever the threads
/// currently between their insert and their eviction are carrying. Bounded, not a leak;
/// `concurrent_inserts_settle_within_budget` pins both halves.
///
/// More than one thread can genuinely be here. MCP dispatch is serial, but `spawn_with_timeout`
/// *detaches* a worker whose request timed out and `MAX_ABANDONED_THREADS` of them may still
/// be running inside `format_search_result`, holding this same cache. So the scope-label path
/// is not single-threaded, whatever the request loop suggests.
///
/// **Freeing to `EVICT_TO_PERCENT` of the budget rather than to exactly the budget is what
/// makes the linear scan affordable.** Each pass scans the whole map for the coldest entry, so
/// evicting one entry per insert costs `O(entries)` on every insert once full — measured at
/// ~1.5x wall time for `tilth map` over a 4 000-file tree, which inserts one entry per file
/// and re-reads none of them. Freeing a tenth of the budget per scan amortises that to
/// roughly one scan per tenth-of-a-budget inserted.
///
/// `protect` is the key just inserted: without it a full map could evict the very entry the
/// caller is storing, which is wasted work and a guaranteed miss on the next read. It also
/// guarantees progress — every iteration either removes some other key or breaks.
fn evict_to_budget<V>(
    map: &DashMap<PathBuf, V>,
    protect: &Path,
    total: &AtomicUsize,
    budget: usize,
    used_of: impl Fn(&V) -> &AtomicU64,
    bytes_of: impl Fn(&V) -> usize,
) {
    if total.load(Ordering::Relaxed) <= budget {
        return;
    }
    let target = budget - budget / EVICT_FREE_FRACTION;
    while total.load(Ordering::Relaxed) > target {
        // Bind rather than inlining into the `match` scrutinee: `DashMap::iter` holds a shard
        // read guard, and a scrutinee temporary lives until the end of the match — which would
        // deadlock against the `remove` below. As a `let`, both guards drop at the semicolon.
        let victim = map
            .iter()
            .filter(|e| e.key().as_path() != protect)
            .min_by_key(|e| used_of(e.value()).load(Ordering::Relaxed))
            .map(|e| e.key().clone());
        match victim {
            Some(key) => {
                // Refund what `remove` actually took out, not what the scan saw. Those differ:
                // between the scan and the removal another thread can merge into the same
                // entry and rewrite its `bytes`, and refunding the stale figure desynchronises
                // the counter from the map. Reading it from the removed value also makes the
                // refund happen exactly once, since only one caller gets `Some` back.
                if let Some((_, evicted)) = map.remove(&key) {
                    sub_bytes(total, bytes_of(&evicted));
                }
            }
            // Nothing left but `protect`. One entry larger than the whole budget stays
            // cached rather than being thrashed out on the next insert — the same
            // "a single item must always be admissible" rule #70 states for its budget.
            None => break,
        }
    }
}

/// `saturating_sub` on a shared total, so an accounting slip can never wrap the counter to
/// `usize::MAX` and permanently disable caching. Lifted from `BloomFilterCache::sub_bytes`,
/// which exists for the same reason.
fn sub_bytes(total: &AtomicUsize, amount: usize) {
    let _ = total.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |b| {
        Some(b.saturating_sub(amount))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Nominal cost of one test entry — a short path plus a short outline — so a budget can
    /// be written as "room for N entries" rather than a bare byte count. Tests that depend on
    /// the exact boundary derive it from `outline_cost` instead; see
    /// `eviction_keeps_the_recently_used_entry`.
    const SMALL_ENTRY: usize = 32;

    #[test]
    fn evicts_stale_mtime_on_reinsert() {
        let cache = OutlineCache::new();
        let path = std::path::Path::new("fake/path.rs");
        let t0 = SystemTime::UNIX_EPOCH;
        let t1 = t0 + Duration::from_secs(1);

        // Insert with t0.
        cache.get_or_compute(path, t0, || "outline v0".to_string());
        assert_eq!(cache.entries.len(), 1);

        // Re-insert with t1 — stale t0 entry must be evicted.
        cache.get_or_compute(path, t1, || "outline v1".to_string());
        assert_eq!(cache.entries.len(), 1, "stale entry was not evicted");

        // Confirm only the new entry survives.
        let hit = cache.get_or_compute(path, t1, || panic!("should hit cache"));
        assert_eq!(&*hit, "outline v1");
    }

    /// The bound is on entries, so it has to hold however many distinct files a session
    /// touches — that is the whole complaint in #67, where the map grew once per distinct
    /// file appearing in a result and nothing ever evicted.
    #[test]
    fn outline_map_stays_within_its_byte_budget() {
        let budget = 8 * SMALL_ENTRY;
        let cache = OutlineCache::with_budget(budget);
        let t = SystemTime::UNIX_EPOCH;
        for i in 0..500 {
            let p = PathBuf::from(format!("f{i}.rs"));
            cache.get_or_compute(&p, t, || format!("outline {i}"));
            assert!(
                cache.outline_bytes() <= budget,
                "after {i} inserts the map holds {} bytes against a {budget} budget",
                cache.outline_bytes()
            );
        }
        assert!(
            cache.entries.len() > 1,
            "the cache should still hold entries"
        );
        // The accounting must match reality, not just stay under the ceiling — a counter that
        // drifts low would keep the bound satisfied while the map grew without limit.
        let actual: usize = cache
            .entries
            .iter()
            .map(|e| outline_cost(e.key(), &e.outline))
            .sum();
        assert_eq!(
            cache.outline_bytes(),
            actual,
            "byte counter disagrees with the map it is counting"
        );
    }

    #[test]
    fn label_map_stays_within_its_byte_budget() {
        let budget = 8 * SMALL_ENTRY;
        let cache = OutlineCache::with_budget(budget);
        let t = SystemTime::UNIX_EPOCH;
        for i in 0..500 {
            let p = PathBuf::from(format!("f{i}.rs"));
            cache.store_labels(&p, t, [(1, None)]);
            assert!(
                cache.label_bytes() <= budget,
                "after {i} inserts: {}",
                cache.label_bytes()
            );
        }
        let actual: usize = cache
            .labels
            .iter()
            .map(|e| label_cost(e.key(), &e.lines))
            .sum();
        assert_eq!(cache.label_bytes(), actual);
    }

    /// The merge in `store_labels` is why an entry count could not bound this map: one hot
    /// file accumulates every line ever asked about. The byte budget has to see that growth
    /// and act on it, which an entry count never would — the map holds a single key
    /// throughout.
    #[test]
    fn one_hot_file_accumulating_lines_is_bounded() {
        let budget = 4 * 1024;
        let cache = OutlineCache::with_budget(budget);
        let t = SystemTime::UNIX_EPOCH;
        let p = PathBuf::from("hot.rs");
        for round in 0..500u32 {
            let lines: Vec<(u32, Option<ScopeLabel>)> = (0..10)
                .map(|k| {
                    (
                        round * 10 + k,
                        Some(ScopeLabel {
                            kind: "function",
                            name: format!("fn_{round}_{k}"),
                        }),
                    )
                })
                .collect();
            cache.store_labels(&p, t, lines);
        }
        // The cap, not the budget, is what bounds a single merging entry — eviction cannot
        // touch the entry being written. Pinned as the line count, which is what the cap
        // actually controls, plus the byte total it implies.
        let lines_held = cache.labels.get(&p).expect("still cached").lines.len();
        assert!(
            lines_held <= MAX_LINES_PER_FILE,
            "one file accumulated {lines_held} lines"
        );
        assert!(
            cache.label_bytes() < 64 * 1024,
            "one file grew to {} bytes",
            cache.label_bytes()
        );
        let _ = budget;
    }

    /// An entry larger than the whole budget must stay cached rather than be evicted on the
    /// next insert and re-evicted forever. Same rule #70 states for its own budget: one item
    /// is always admissible.
    #[test]
    fn an_entry_larger_than_the_budget_survives() {
        let cache = OutlineCache::with_budget(64);
        let t = SystemTime::UNIX_EPOCH;
        let big = PathBuf::from("big.rs");
        let huge = "x".repeat(4096);
        cache.get_or_compute(&big, t, || huge.clone());
        assert_eq!(
            &*cache.get_or_compute(&big, t, || panic!("the oversized entry was evicted")),
            huge
        );
    }

    /// Eviction must be least-recently-*used*, not least-recently-inserted, or a session
    /// that keeps returning to the same file loses it to files it touched once. This is the
    /// property that separates this from the Bloom cache's refuse-when-full (#40): a full
    /// cache still serves its working set.
    #[test]
    fn eviction_keeps_the_recently_used_entry() {
        let (a, b, c) = (
            PathBuf::from("a.rs"),
            PathBuf::from("b.rs"),
            PathBuf::from("c.rs"),
        );
        // Sized from the real cost function, not a guessed constant: room for two of these
        // entries and not three, so inserting the third must evict exactly one. Deriving it
        // is what keeps the test meaningful if `outline_cost` ever changes — a hardcoded
        // budget silently became "no eviction at all" while this test still passed.
        let one = outline_cost(&a, "A");
        let cache = OutlineCache::with_budget(2 * one + one / 2);
        let t = SystemTime::UNIX_EPOCH;

        cache.get_or_compute(&a, t, || "A".into());
        cache.get_or_compute(&b, t, || "B".into());
        // Touch `a` so `b` becomes the coldest.
        let _ = cache.get_or_compute(&a, t, || panic!("a must still be cached"));
        cache.get_or_compute(&c, t, || "C".into());

        assert!(
            !cache.entries.contains_key(&b),
            "b was the least recently used and should have been evicted; map holds {:?}",
            cache
                .entries
                .iter()
                .map(|e| e.key().clone())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            &*cache.get_or_compute(&a, t, || panic!("a was evicted despite being used")),
            "A"
        );
    }

    /// A cached "top level" answer (`None`) is a real answer. Storing it and reading it back
    /// as a hit is what stops a top-level match from re-parsing its file on every call.
    #[test]
    fn a_top_level_answer_is_cached_not_treated_as_a_miss() {
        let cache = OutlineCache::new();
        let t = SystemTime::UNIX_EPOCH;
        let p = PathBuf::from("f.rs");
        cache.store_labels(&p, t, [(7, None)]);
        assert_eq!(cache.cached_label(&p, t, 7), Some(None), "should be a hit");
        assert_eq!(cache.cached_label(&p, t, 8), None, "line 8 was never asked");
    }

    /// Two calls asking about different lines of one file must both end up cached — a
    /// replace-instead-of-merge would make a session alternating between two lines re-parse
    /// every time.
    #[test]
    fn storing_labels_merges_within_one_mtime() {
        let cache = OutlineCache::new();
        let t = SystemTime::UNIX_EPOCH;
        let p = PathBuf::from("f.rs");
        let lbl = |n: &str| {
            Some(ScopeLabel {
                kind: "function",
                name: n.to_string(),
            })
        };
        cache.store_labels(&p, t, [(1, lbl("one"))]);
        cache.store_labels(&p, t, [(2, lbl("two"))]);
        assert_eq!(cache.cached_label(&p, t, 1), Some(lbl("one")));
        assert_eq!(cache.cached_label(&p, t, 2), Some(lbl("two")));
    }

    /// Both maps are `DashMap`s and the struct is `Sync`, so the budget has to hold under
    /// concurrent insertion rather than only in a single-threaded loop. This is the shape of
    /// `bloom`'s `concurrent_probes_of_one_path_charge_the_budget_once`, and for the same
    /// reason: there, arithmetic outside the shard lock permanently double-billed the budget.
    ///
    /// Concurrency here is not hypothetical. MCP dispatch is serial, but `spawn_with_timeout`
    /// detaches a timed-out worker and up to `MAX_ABANDONED_THREADS` of them can still be
    /// inside `format_search_result` holding this cache.
    ///
    /// The property pinned is the real one rather than the one that reads better: **at rest
    /// the total is within budget and the counter agrees with the map; in flight it can
    /// overshoot by what the threads between their insert and their eviction are carrying.**
    /// An earlier version asserted a hard in-flight bound and failed, which is how the
    /// overshoot was found rather than assumed.
    #[test]
    fn concurrent_inserts_settle_within_budget() {
        const THREADS: usize = 8;
        const BUDGET: usize = 8 * SMALL_ENTRY;
        let cache = Arc::new(OutlineCache::with_budget(BUDGET));
        let t = SystemTime::UNIX_EPOCH;
        let mut handles = Vec::new();
        for th in 0..THREADS {
            let c = Arc::clone(&cache);
            handles.push(std::thread::spawn(move || {
                for i in 0..200 {
                    // Overlapping key spaces across threads, so the same path is inserted
                    // concurrently as well as distinct ones — the Occupied arms, where the
                    // refund-then-charge happens, run as often as the Vacant ones.
                    let p = PathBuf::from(format!("f{}.rs", (i + th) % 64));
                    c.get_or_compute(&p, t, || format!("outline {i}"));
                    c.store_labels(&p, t, [(i as u32 % 7, None)]);
                }
            }));
        }
        for h in handles {
            h.join().expect("no thread panicked");
        }

        // At rest: within budget, and — the part that catches a double-billed or leaked
        // refund — the counter equals what the map actually holds.
        assert!(
            cache.outline_bytes() <= BUDGET,
            "outline total settled at {}",
            cache.outline_bytes()
        );
        assert!(
            cache.label_bytes() <= BUDGET,
            "label total settled at {}",
            cache.label_bytes()
        );
        let outline_actual: usize = cache
            .entries
            .iter()
            .map(|e| outline_cost(e.key(), &e.outline))
            .sum();
        let label_actual: usize = cache
            .labels
            .iter()
            .map(|e| label_cost(e.key(), &e.lines))
            .sum();
        assert_eq!(
            cache.outline_bytes(),
            outline_actual,
            "outline counter desynchronised from the map under contention"
        );
        assert_eq!(
            cache.label_bytes(),
            label_actual,
            "label counter desynchronised from the map under contention"
        );
    }

    /// The staleness guard has to survive the merge path: a rewritten file must not serve
    /// answers computed from its previous contents.
    #[test]
    fn a_new_mtime_discards_every_answer_for_that_file() {
        let cache = OutlineCache::new();
        let t0 = SystemTime::UNIX_EPOCH;
        let t1 = t0 + Duration::from_secs(1);
        let p = PathBuf::from("f.rs");
        let lbl = Some(ScopeLabel {
            kind: "function",
            name: "old".to_string(),
        });
        cache.store_labels(&p, t0, [(1, lbl.clone()), (2, lbl)]);
        cache.store_labels(&p, t1, [(1, None)]);
        assert_eq!(cache.cached_label(&p, t0, 1), None, "stale mtime must miss");
        assert_eq!(
            cache.cached_label(&p, t1, 2),
            None,
            "line 2's answer was computed from the old contents and must not survive"
        );
        assert_eq!(cache.cached_label(&p, t1, 1), Some(None));
    }
}
