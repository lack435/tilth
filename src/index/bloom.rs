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
/// The constant is calibrated against measured peak RSS, **not** against its own accounting —
/// which matters, because the accounting undercounts. On a 176k-file C++ tree, one
/// `kind: "callers"` query, three repetitions in a single MCP session:
///
/// ```text
/// ceiling        peak RSS      wall
/// unbounded      188-214 MB    2947-3572ms
/// 64 MB          136-147 MB    3064-3190ms
/// 8 MB            70-78 MB     3583-3670ms
/// disabled        44-50 MB     3097-3917ms
/// ```
///
/// Read that carefully, because two things in it are easy to get wrong:
///
/// * **The ceiling controls peak memory** — monotonic across four settings, which is what this
///   bound is for. But a nominal 64 MB produced ~94 MB of actual growth over the
///   cache-disabled baseline, so `estimated_filter_bytes` undercounts real cost by around 1.5x.
///   The constant is a calibrated knob, not a byte-accurate budget, and it should be re-measured
///   rather than reasoned about if it is ever retuned.
/// * **The time differences are inside the noise.** An earlier two-point reading suggested the
///   unbounded cache was worth ~15% of wall time; across four settings the ranges overlap and
///   the cache-disabled run was sometimes the fastest. So this bound is close to free, and any
///   claim that the cache buys a specific percentage is not supported by these numbers.
///
/// 32 MB is what shipped. Measured against `main` on the same tree, three reps each:
///
/// ```text
///                 unbounded                 32 MB ceiling
/// single-target   189-215 MB / 3001-3546ms  107-122 MB / 2973-3471ms
/// 5-target        206-241 MB / 10.1-10.8s   117-128 MB / 11.4-12.2s
/// ```
///
/// So roughly 45% off peak RSS, free on the single-walk path and about **12% slower** on the
/// multi-walk one — that second row is a real cost, outside the noise, and worth stating rather
/// than rounding away. Multi-target `callers` is where cross-walk reuse pays most, because it
/// runs one walk per target; that fan-out is itself tracked as a defect, and when it is fixed
/// this cost shrinks with it.
///
/// Output is unaffected at every setting — verified byte-identical with the cache unbounded,
/// bounded and disabled, and it must be: a filter is only ever a pre-filter ahead of a real
/// `memmem` check and a parse, so a miss costs work and never a wrong answer.
const MAX_CACHE_BYTES: usize = 32 * 1024 * 1024;

/// Bytes a filter of `expected_items` occupies, for budgeting only.
///
/// `BloomFilter` exposes no size accessor, so this reproduces its nominal sizing: a 1%
/// false-positive rate needs about 9.6 bits per item, hence ~1.2 bytes, plus a constant for the
/// struct. It is known to **undercount** — see `MAX_CACHE_BYTES` for the measurement — because
/// it counts neither the `PathBuf` key, nor `DashMap`'s per-entry overhead, nor allocator
/// rounding. That is tolerable for a guard against unbounded growth; it would not be tolerable
/// if anything depended on the number being right.
fn estimated_filter_bytes(expected_items: usize) -> usize {
    expected_items * 12 / 10 + 64
}

/// Thread-safe cache of per-file Bloom filters, keyed by path and validated
/// by mtime. Stale entries are automatically rebuilt on access.
///
/// Bounded by `MAX_CACHE_BYTES`. Once the budget is reached the cache stops accepting new
/// entries rather than evicting: eviction would need an access order that `DashMap` does not
/// keep, and for a cache whose miss penalty is a rebuild the simpler rule is enough. Repeated
/// walks visit files in a similar order, so what gets in early is also what gets re-probed.
pub struct BloomFilterCache {
    filters: DashMap<PathBuf, CachedFilter>,
    /// Sum of `CachedFilter::bytes` over the map. Only ever compared against the ceiling, so
    /// a transient overshoot from two threads inserting at once is acceptable and bounded by
    /// one filter per thread.
    bytes: std::sync::atomic::AtomicUsize,
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
        Self {
            filters: DashMap::new(),
            bytes: std::sync::atomic::AtomicUsize::new(0),
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

        // Cache miss or stale: build and cache a new filter
        let (filter, expected_items) = build_filter(content, code_lang(path));
        let result = filter.contains(symbol);
        let cost = estimated_filter_bytes(expected_items);

        // Replacing a stale entry frees its budget first, so a file edited repeatedly cannot
        // consume the ceiling one revision at a time.
        let reclaimed = self
            .filters
            .get(path)
            .filter(|e| e.mtime != mtime)
            .map_or(0, |e| e.bytes);
        if reclaimed > 0 {
            self.bytes.fetch_sub(reclaimed, Relaxed);
        }

        if self.bytes.load(Relaxed) + cost <= MAX_CACHE_BYTES {
            self.bytes.fetch_add(cost, Relaxed);
            self.filters.insert(
                path.to_path_buf(),
                CachedFilter {
                    filter,
                    mtime,
                    bytes: cost,
                },
            );
        } else if reclaimed > 0 {
            // Budget reclaimed but the replacement does not fit: drop the stale entry rather
            // than leave one whose mtime can never match again.
            self.filters.remove(path);
        }
        result
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
/// Returns the filter and the `expected_items` it was sized for, so the caller can budget
/// against it without a size accessor `BloomFilter` does not provide.
fn build_filter(content: &str, lang: Option<Lang>) -> (BloomFilter, usize) {
    let idents: Vec<&str> = extract_identifiers(content, lang).collect();
    // Sized for total token count, not unique identifiers -- duplicates over-allocate
    // the filter, so the achieved FPR is well below the 0.01 target in practice.
    let expected = idents.len().max(1);

    let mut filter = BloomFilter::with_false_pos(0.01).expected_items(expected);
    for ident in idents {
        filter.insert(ident);
    }
    (filter, expected)
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

    /// The cache must stop growing at `MAX_CACHE_BYTES`, and must keep answering correctly
    /// once it does — a filter is only ever a pre-filter, so a refused insert costs a rebuild
    /// and never a wrong answer.
    ///
    /// Uses `estimated_filter_bytes` to size the fixture rather than a magic number, so raising
    /// the ceiling retunes this instead of breaking it.
    #[test]
    fn cache_stops_growing_at_the_ceiling_and_stays_correct() {
        let cache = BloomFilterCache::new();
        let mtime = SystemTime::UNIX_EPOCH;

        // Each file carries enough distinct identifiers to make its filter non-trivial.
        let content: String = (0..2000).map(|i| format!("ident_{i} ")).collect();
        let per_file = estimated_filter_bytes(2001);
        // Enough files to overrun the ceiling several times over.
        let files = MAX_CACHE_BYTES / per_file + 50;

        for f in 0..files {
            let path = PathBuf::from(format!("/synthetic/f{f}.rs"));
            // Correctness under the bound: a symbol that is present must still be reported
            // present whether or not this file got cached.
            // The only guarantee a Bloom filter makes is no false *negatives*, so this is the
            // only thing that can be asserted per file. Asserting the absent case fails
            // legitimately — the filters are built for a 1% false-positive rate, and over this
            // many files a false positive is near-certain (it first fired on file 6).
            assert!(
                cache.contains(&path, mtime, &content, "ident_7"),
                "a present symbol must be found regardless of cache admission (file {f})"
            );
        }

        let bytes = cache.cached_bytes();
        assert!(
            bytes <= MAX_CACHE_BYTES,
            "cache exceeded its ceiling: {bytes} > {MAX_CACHE_BYTES}"
        );
        // And it must actually have filled up, or the ceiling was never exercised.
        assert!(
            bytes > MAX_CACHE_BYTES / 2,
            "fixture did not reach the ceiling ({bytes} bytes); the bound is untested"
        );
    }

    /// Re-caching a modified file must not consume the ceiling one revision at a time.
    #[test]
    fn restating_a_stale_entry_reclaims_its_budget() {
        let cache = BloomFilterCache::new();
        let path = PathBuf::from("/synthetic/churn.rs");
        let content: String = (0..2000).map(|i| format!("ident_{i} ")).collect();

        let mut last = 0;
        for rev in 0..50u64 {
            let mtime = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(rev);
            assert!(cache.contains(&path, mtime, &content, "ident_7"));
            let now = cache.cached_bytes();
            if rev > 0 {
                assert_eq!(
                    now, last,
                    "budget grew on revision {rev} — a stale entry was not reclaimed"
                );
            }
            last = now;
        }
        assert!(last > 0, "nothing was cached, so this proves nothing");
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
        let (filter, _) = build_filter(content, Some(Lang::Rust));

        assert!(filter.contains("search"));
        assert!(filter.contains("query"));
        assert!(filter.contains("Vec"));
        assert!(filter.contains("Match"));
        assert!(filter.contains("find"));
        assert!(!filter.contains("nonexistent_symbol_xyz"));
    }
}
