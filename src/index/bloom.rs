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
/// has a known mechanism — `bloom_walk::read_with_bloom_check` calls `contains` once per target,
/// so a refused admission makes every target rebuild the same filter instead of one building it
/// and the rest hitting. That is worth fixing on its own and is tracked separately.
///
/// Output is unaffected at every setting — verified byte-identical with the cache unbounded,
/// bounded and disabled, and it must be: a filter is only ever a pre-filter ahead of a real
/// `memmem` check and a parse, so a miss costs work and never a wrong answer.
const MAX_CACHE_BYTES: usize = 32 * 1024 * 1024;

/// Fixed cost of one cache entry, beyond its bit array.
///
/// `CachedFilter` embeds the `BloomFilter` struct, its mtime and its own byte count; the map
/// also stores a `PathBuf` key inline. The rest is the key's heap allocation plus hashbrown's
/// control byte and load-factor slack, which is where the 128 comes from — deliberately
/// generous, since over-counting the overhead only makes the ceiling more conservative.
const PER_ENTRY_OVERHEAD: usize =
    std::mem::size_of::<CachedFilter>() + std::mem::size_of::<PathBuf>() + 128;

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
/// Bounded by `ceiling`. Once the budget is reached the cache stops accepting new entries
/// rather than evicting, because eviction needs an access order `DashMap` does not keep and a
/// miss only costs a rebuild.
///
/// Known limitation, stated because the first version of this comment justified the design with
/// a claim that does not hold ("repeated walks visit files in a similar order, so what gets in
/// early is also what gets re-probed"). Successive tool calls routinely supply *different*
/// scopes, and the natural agent shape is adversarial to this: a broad query at the repo root
/// fills the budget with whatever the walk reached first, then the agent narrows to one subtree
/// for the next fifty calls and nothing it touches is ever admitted. In that regime the cache
/// is dead weight — bounded, but useless. A generational reset (clear and start over when the
/// ceiling is hit) would fix it without needing an access order; not done here.
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
}

struct CachedFilter {
    filter: BloomFilter,
    mtime: SystemTime,
    /// What this entry contributed to `bytes`, so replacing a stale entry can subtract it.
    bytes: usize,
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
        }
    }

    /// Bytes currently accounted for by cached filters. Test/diagnostic accessor.
    #[must_use]
    pub fn cached_bytes(&self) -> usize {
        self.bytes.load(std::sync::atomic::Ordering::Relaxed)
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
        // Fast path: check existing cached entry
        if let Some(entry) = self.filters.get(path) {
            if entry.mtime == mtime {
                return entry.filter.contains(symbol);
            }
        }

        // Cache miss or stale: build outside any lock, answer from the fresh filter, then admit.
        let filter = build_filter(content, code_lang(path));
        let result = filter.contains(symbol);
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
                    occupied.insert(CachedFilter {
                        filter,
                        mtime,
                        bytes: cost,
                    });
                } else {
                    // Its budget is already reclaimed and its mtime can never match again, so
                    // keeping it would be resident memory the counter no longer knows about.
                    occupied.remove();
                }
            }
            dashmap::mapref::entry::Entry::Vacant(vacant) => {
                if self.fits(cost) {
                    self.bytes.fetch_add(cost, Relaxed);
                    vacant.insert(CachedFilter {
                        filter,
                        mtime,
                        bytes: cost,
                    });
                }
            }
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
