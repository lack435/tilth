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
//! The retained tree was ~48x the file's own bytes. That is not a number a per-byte budget
//! can predict either: measured content-bytes-per-node is 3.00 on that dense fixture, 4.58
//! on this repository's Rust and 5.20 on real C++, so "tree size" varies by 1.7x per source
//! byte depending on content.
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
//! Both maps that remain are bounded by entry count (`MAX_CACHED_FILES`) with
//! least-recently-used eviction. Eviction rather than refusal-on-full is deliberate: the
//! `BloomFilterCache` refuses once its ceiling is reached, and #40 measured what that does —
//! whatever fills the budget first keeps it forever, scoring a 0% hit rate against 80%
//! achievable. A full LRU still serves the working set; a full refusing cache serves nothing.

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::SystemTime;

use dashmap::DashMap;

/// Largest file this cache will parse, in bytes.
///
/// Unchanged from when `get_or_parse` owned it. A file past this is not worth a scope
/// annotation, and the transient tree for one is the single largest live allocation the
/// render path can make.
const MAX_PARSE_BYTES: u64 = 500_000;

/// Entry-count ceiling for each map.
///
/// A count rather than a byte budget, because what is stored is now small and *bounded per
/// entry*: an outline string measured 3.5–5.8 KB across the sessions below, and a label set
/// is a handful of short strings for the ≤10 lines one page can ask about (≤100 under
/// `--full`). The 70x content-dependence that made a byte budget necessary for trees does
/// not exist for either.
///
/// Sized from measured sessions rather than picked. Distinct files touched, and the bytes
/// they cost, over whole sessions:
///
/// ```text
///                                      outline   label
/// session                              files     files   outline bytes
/// this repo's src, 8 symbol queries        20      19          0.10 MB
/// real C++ tree, 6 usage queries           12      19          0.07 MB
/// 60x499 KB fixture, 5 disjoint queries    50      50          0.00 MB
/// ```
///
/// 1024 is ~20x the largest observed working set, so no realistic session evicts at all,
/// and it caps the two maps at roughly 6 MB and 1 MB respectively for a session that walks
/// far more files than any of these.
const MAX_CACHED_FILES: usize = 1024;

/// A rendered outline string plus the mtime it was computed from.
struct CacheEntry {
    mtime: SystemTime,
    outline: Arc<str>,
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
    used: AtomicU64,
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
    capacity: NonZeroUsize,
}

impl Default for OutlineCache {
    fn default() -> Self {
        Self::with_capacity(NonZeroUsize::new(MAX_CACHED_FILES).expect("nonzero"))
    }
}

impl OutlineCache {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct with a specific per-map entry ceiling. Tests use this to exercise eviction
    /// without materialising a thousand files.
    #[must_use]
    pub fn with_capacity(capacity: NonZeroUsize) -> Self {
        Self {
            entries: DashMap::new(),
            labels: DashMap::new(),
            clock: AtomicU64::new(0),
            capacity,
        }
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
        // Stale or absent — compute and insert, replacing any stale entry.
        let outline: Arc<str> = compute().into();
        let used = AtomicU64::new(self.tick());
        self.entries.insert(
            key.clone(),
            CacheEntry {
                mtime,
                outline: Arc::clone(&outline),
                used,
            },
        );
        evict_to_capacity(&self.entries, &key, self.capacity, |e| &e.used);
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
    pub fn store_labels(
        &self,
        path: &Path,
        mtime: SystemTime,
        resolved: impl IntoIterator<Item = (u32, Option<ScopeLabel>)>,
    ) {
        let key = path.to_path_buf();
        let used = AtomicU64::new(self.tick());
        match self.labels.entry(key.clone()) {
            dashmap::mapref::entry::Entry::Occupied(mut occupied) => {
                if occupied.get().mtime == mtime {
                    let e = occupied.get_mut();
                    e.lines.extend(resolved);
                    e.used
                        .store(used.load(Ordering::Relaxed), Ordering::Relaxed);
                    return;
                }
                occupied.insert(LabelEntry {
                    mtime,
                    lines: resolved.into_iter().collect(),
                    used,
                });
            }
            dashmap::mapref::entry::Entry::Vacant(vacant) => {
                vacant.insert(LabelEntry {
                    mtime,
                    lines: resolved.into_iter().collect(),
                    used,
                });
            }
        }
        // After insert, not before: `entry()` above holds the shard lock, and evicting from
        // the same map while holding it deadlocks.
        evict_to_capacity(&self.labels, &key, self.capacity, |e| &e.used);
    }

    /// Largest file the scope resolver will parse.
    #[must_use]
    pub fn max_parse_bytes() -> u64 {
        MAX_PARSE_BYTES
    }

    #[cfg(test)]
    fn label_files(&self) -> usize {
        self.labels.len()
    }
}

/// Evict least-recently-used entries until `map` is back within `capacity`.
///
/// Called *after* the insert, never before, because the two maps insert through different
/// APIs — `DashMap::insert` and `DashMap::entry` — and only the after position is uniformly
/// safe: `entry()` holds the shard write lock for the whole match, so touching the same map
/// inside it deadlocks.
///
/// `protect` is the key just inserted. Without it a full map would be free to evict the very
/// entry the caller is storing, which is both useless work and a silent cache miss on the
/// next read. It also means the loop can only terminate by removing something else, so the
/// `None` arm breaks rather than spinning.
///
/// Evicting after the insert means the bound is exact **at rest** and overshoots by at most
/// one entry per thread that is mid-insert. That is bounded, not a leak, and the concurrency
/// this sees is minimal — the only caller that can run in parallel is outline computation,
/// and the scope-label path is single-threaded formatting. `concurrent_inserts_stay_within_a_bounded_overshoot`
/// pins both halves.
///
/// A linear scan for the minimum tick, affordable precisely because it only runs on growth
/// past a full map: the scan is bounded by `capacity`, and a session that never fills the map
/// never pays it. The alternative — an access-ordered structure maintained alongside a
/// `DashMap` — is the machinery #40 records as the reason the Bloom cache refuses instead of
/// evicting, and it is not worth it for maps this small.
fn evict_to_capacity<V>(
    map: &DashMap<PathBuf, V>,
    protect: &Path,
    capacity: NonZeroUsize,
    used_of: impl Fn(&V) -> &AtomicU64,
) {
    while map.len() > capacity.get() {
        let victim = map
            .iter()
            .filter(|e| e.key().as_path() != protect)
            .min_by_key(|e| used_of(e.value()).load(Ordering::Relaxed))
            .map(|e| e.key().clone());
        match victim {
            Some(v) => {
                map.remove(&v);
            }
            None => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn cap(n: usize) -> OutlineCache {
        OutlineCache::with_capacity(NonZeroUsize::new(n).unwrap())
    }

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
    fn outline_map_never_exceeds_capacity() {
        let cache = cap(4);
        let t = SystemTime::UNIX_EPOCH;
        for i in 0..50 {
            let p = PathBuf::from(format!("f{i}.rs"));
            cache.get_or_compute(&p, t, || format!("outline {i}"));
            assert!(
                cache.entries.len() <= 4,
                "after {i} inserts the map holds {}",
                cache.entries.len()
            );
        }
        assert_eq!(cache.entries.len(), 4);
    }

    #[test]
    fn label_map_never_exceeds_capacity() {
        let cache = cap(4);
        let t = SystemTime::UNIX_EPOCH;
        for i in 0..50 {
            let p = PathBuf::from(format!("f{i}.rs"));
            cache.store_labels(&p, t, [(1, None)]);
            assert!(cache.label_files() <= 4);
        }
        assert_eq!(cache.label_files(), 4);
    }

    /// Eviction must be least-recently-*used*, not least-recently-inserted, or a session
    /// that keeps returning to the same file loses it to files it touched once. This is the
    /// property that separates this from the Bloom cache's refuse-when-full (#40): a full
    /// cache still serves its working set.
    #[test]
    fn eviction_keeps_the_recently_used_entry() {
        let cache = cap(2);
        let t = SystemTime::UNIX_EPOCH;
        let (a, b, c) = (
            PathBuf::from("a.rs"),
            PathBuf::from("b.rs"),
            PathBuf::from("c.rs"),
        );
        cache.get_or_compute(&a, t, || "A".into());
        cache.get_or_compute(&b, t, || "B".into());
        // Touch `a` so `b` becomes the coldest.
        let _ = cache.get_or_compute(&a, t, || panic!("a must still be cached"));
        cache.get_or_compute(&c, t, || "C".into());

        assert_eq!(
            &*cache.get_or_compute(&a, t, || panic!("a was evicted despite being used")),
            "A"
        );
        assert!(
            !cache.entries.contains_key(&b),
            "b should have been evicted"
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

    /// Both maps are `DashMap`s and the struct is `Sync`, so the bound has to hold under
    /// concurrent insertion rather than only in a single-threaded loop. This is the shape of
    /// `bloom`'s `concurrent_probes_of_one_path_charge_the_budget_once`, for the same reason:
    /// there, arithmetic outside the shard lock permanently double-billed the budget.
    ///
    /// It cannot desynchronise the way the Bloom budget did, because `len()` is the map's own
    /// count rather than a number maintained beside it — a large part of why this bound is an
    /// entry count. What it *can* do is overshoot transiently, and the test pins the real
    /// property rather than the one that reads better: **at rest the map is within capacity,
    /// and in flight it exceeds it by at most one entry per thread mid-insert**.
    ///
    /// That is inherent to evicting after the insert, which `store_labels` has to do because
    /// `entry()` holds the shard lock. Bounded overshoot is the honest claim; asserting a hard
    /// in-flight bound would fail, and it did — the first version of this test tripped at 9
    /// with capacity 8 and 8 threads.
    #[test]
    fn concurrent_inserts_stay_within_a_bounded_overshoot() {
        const CAPACITY: usize = 8;
        const THREADS: usize = 8;
        let cache = Arc::new(cap(CAPACITY));
        let t = SystemTime::UNIX_EPOCH;
        let mut handles = Vec::new();
        for th in 0..THREADS {
            let c = Arc::clone(&cache);
            handles.push(std::thread::spawn(move || {
                for i in 0..200 {
                    // Overlapping key spaces across threads, so the same path is inserted
                    // concurrently as well as distinct ones.
                    let p = PathBuf::from(format!("f{}.rs", (i + th) % 64));
                    c.get_or_compute(&p, t, || format!("outline {i}"));
                    c.store_labels(&p, t, [(i as u32 % 7, None)]);
                    // In flight: one thread can be between its insert and its eviction.
                    assert!(
                        c.entries.len() <= CAPACITY + THREADS,
                        "outline map grew to {} — overshoot is not bounded by thread count",
                        c.entries.len()
                    );
                    assert!(
                        c.label_files() <= CAPACITY + THREADS,
                        "label map grew to {} — overshoot is not bounded by thread count",
                        c.label_files()
                    );
                }
            }));
        }
        for h in handles {
            h.join().expect("no thread panicked");
        }
        // At rest every insert has run its eviction, so the steady-state bound is exact.
        assert_eq!(cache.entries.len(), CAPACITY);
        assert_eq!(cache.label_files(), CAPACITY);
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
