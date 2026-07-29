//! Shared file-prefilter helper for relational queries (callers, callees,
//! deps). Reads a file, gates on size, and runs the per-file bloom prefilter
//! against any of the supplied target symbols. Returns content + mtime when
//! the file is worth deeper inspection (tree-sitter parse, outline scan).

use std::path::Path;
use std::time::SystemTime;

use crate::index::bloom::BloomFilterCache;

/// Skip files larger than this; tree-sitter parses on huge files dominate
/// query latency without surfacing useful matches.
pub(super) const MAX_FILE_SIZE: u64 = 500_000;

/// Read `path`, validate size, and pass through only when at least one
/// target is bloom-positive. Returns `(content, mtime)` for the next stage,
/// or `None` to skip the file.
///
/// Bloom is probabilistic: a positive may be a false positive. Callers that
/// need a tighter pre-AST filter (e.g. memchr) should run it on the returned
/// content before paying for tree-sitter.
///
/// The whole target set goes to `contains_any` in one call rather than being looped over
/// `contains`. That is load-bearing, not style: the cache is byte-bounded, so once the budget is
/// full admission is refused and a per-target loop rebuilt the same file's filter once per
/// target. See `BloomFilterCache::contains_any`.
pub(super) fn read_with_bloom_check<I, S>(
    path: &Path,
    targets: I,
    bloom: &BloomFilterCache,
    max_size: u64,
) -> Option<(String, SystemTime)>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let meta = std::fs::metadata(path).ok()?;
    if meta.len() > max_size {
        return None;
    }
    let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
    let content = std::fs::read_to_string(path).ok()?;

    if !bloom.contains_any(path, mtime, &content, targets) {
        return None;
    }

    Some((content, mtime))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::fs;

    #[test]
    fn returns_none_for_oversized_file() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("big.rs");
        // Fill past max_size
        let payload = "fn foo() {}\n".repeat(2);
        fs::write(&p, &payload).unwrap();
        let bloom = BloomFilterCache::new();
        let targets: HashSet<String> = ["foo".to_string()].into_iter().collect();
        // max_size below file len → skip
        assert!(read_with_bloom_check(&p, &targets, &bloom, 1).is_none());
    }

    #[test]
    fn returns_none_when_no_target_is_bloom_positive() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.rs");
        fs::write(&p, "fn alpha() {}\n").unwrap();
        let bloom = BloomFilterCache::new();
        let targets: HashSet<String> = ["beta".to_string()].into_iter().collect();
        assert!(read_with_bloom_check(&p, &targets, &bloom, MAX_FILE_SIZE).is_none());
    }

    #[test]
    fn returns_content_when_target_present() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.rs");
        fs::write(&p, "fn alpha() {}\n").unwrap();
        let bloom = BloomFilterCache::new();
        let targets: HashSet<String> = ["alpha".to_string()].into_iter().collect();
        let (content, _) = read_with_bloom_check(&p, &targets, &bloom, MAX_FILE_SIZE).unwrap();
        assert!(content.contains("alpha"));
    }

    /// The prefilter must cost one filter build per file however many targets it is given, even
    /// when the cache is full and refuses to admit. This is the #34 path: `find_callers_batch`
    /// hands the whole target set to this helper for every candidate file, so a per-target build
    /// is an N x multiplier on the walk.
    #[test]
    fn a_full_cache_costs_one_build_per_file_not_per_target() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.rs");
        fs::write(&p, "fn alpha() { beta(); }\n").unwrap();

        // Ceiling of 0 refuses every admission — the steady state once the real 32 MB budget is
        // full, and the regime where the old per-target loop rebuilt the same filter N times.
        let bloom = BloomFilterCache::with_ceiling(0);
        // Ordered, and with the only present target last, so nothing short-circuits early: a
        // `HashSet` iterates arbitrarily and could hit `alpha` first, hiding the fan-out.
        let targets = ["absent_a", "absent_b", "absent_c", "absent_d", "alpha"];

        assert!(read_with_bloom_check(&p, targets, &bloom, MAX_FILE_SIZE).is_some());
        assert_eq!(
            bloom.filters_built(),
            1,
            "{} targets cost {} builds for one file; the prefilter is looping `contains` per \
             target again",
            targets.len(),
            bloom.filters_built()
        );
    }

    #[test]
    fn accepts_borrowed_str_targets() {
        // callees.rs holds HashSet<&str>; the helper must accept that shape
        // without forcing a String allocation per call.
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.rs");
        fs::write(&p, "fn alpha() {}\n").unwrap();
        let bloom = BloomFilterCache::new();
        let targets: HashSet<&str> = ["alpha"].into_iter().collect();
        assert!(read_with_bloom_check(&p, &targets, &bloom, MAX_FILE_SIZE).is_some());
    }
}
