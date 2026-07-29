use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::types::{FacetTotals, Match};

/// Faceted search results grouped by match type and location.
pub struct FacetedResult {
    pub definitions: Vec<Match>,
    pub implementations: Vec<Match>,
    pub tests: Vec<Match>,
    pub usages_local: Vec<Match>,
    pub usages_cross: Vec<Match>,
}

/// Which facet a single match belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Facet {
    Definition,
    Implementation,
    Test,
    UsageLocal,
    UsageCross,
}

/// The package root used to decide local-vs-cross, derived from the primary definition.
fn primary_package(matches: &[Match]) -> Option<PathBuf> {
    matches
        .iter()
        .find(|m| m.is_definition)
        .and_then(|m| m.path.parent())
        .and_then(crate::lang::package_root)
        .map(std::path::Path::to_path_buf)
}

/// Memoises "is this file's directory in the primary package?" per parent directory.
///
/// `package_root` stats up to ten manifest names per directory level and walks to the
/// filesystem root, so calling it once per match is an unbounded syscall storm now that
/// the search walks complete — the old ~80-entry early-quit bound is what previously
/// made it free. `rank::sort` keeps an equivalent `pkg_cache` for exactly this reason.
/// The answer depends only on the parent directory, so that is the key.
type PkgCache = HashMap<PathBuf, bool>;

fn same_package_cached(path: &Path, primary_pkg: Option<&PathBuf>, cache: &mut PkgCache) -> bool {
    // No primary definition means no local/cross distinction to make, and
    // `is_same_package` would return false without touching the filesystem.
    if primary_pkg.is_none() {
        return false;
    }
    let Some(parent) = path.parent() else {
        return false;
    };
    if let Some(hit) = cache.get(parent) {
        return *hit;
    }
    let same = is_same_package(path, primary_pkg);
    cache.insert(parent.to_path_buf(), same);
    same
}

/// Single source of truth for facet assignment. `facet_matches` and `facet_totals`
/// both route through this so a count can never disagree with what gets grouped —
/// they are rendered side by side as `shown/total`, so a drift between them would
/// read as a truncation that did not happen.
fn facet_of(m: &Match, primary_pkg: Option<&PathBuf>, cache: &mut PkgCache) -> Facet {
    if m.is_definition && m.impl_target.is_some() {
        Facet::Implementation
    } else if m.is_definition {
        Facet::Definition
    } else if is_test_match(m) {
        Facet::Test
    } else if same_package_cached(&m.path, primary_pkg, cache) {
        Facet::UsageLocal
    } else {
        Facet::UsageCross
    }
}

/// Group matches into facets when there are many results (>5).
/// Partitions by definition type, test status, and package locality.
pub fn facet_matches(matches: Vec<Match>, _scope: &Path) -> FacetedResult {
    // Find primary definition's package root for local/cross determination
    let primary_pkg = primary_package(&matches);
    let mut cache = PkgCache::new();

    let mut definitions = Vec::new();
    let mut implementations = Vec::new();
    let mut tests = Vec::new();
    let mut usages_local = Vec::new();
    let mut usages_cross = Vec::new();

    for m in matches {
        match facet_of(&m, primary_pkg.as_ref(), &mut cache) {
            Facet::Implementation => implementations.push(m),
            Facet::Definition => definitions.push(m),
            Facet::Test => tests.push(m),
            Facet::UsageLocal => usages_local.push(m),
            Facet::UsageCross => usages_cross.push(m),
        }
    }

    FacetedResult {
        definitions,
        implementations,
        tests,
        usages_local,
        usages_cross,
    }
}

/// Per-facet totals over the *pre-cap* match set, counted in place.
///
/// This used to be done by cloning the whole set and calling `facet_matches` on the
/// copy, which was justified by a comment reading "bounded by the early-quit
/// thresholds (~80 entries), so the clone is cheap". Removing those thresholds (so
/// that search results stop varying run to run) removed that bound: a common literal
/// on a large tree now reaches tens of thousands of matches, where the clone is a
/// second copy of every `Match` — each carrying a `PathBuf` and the matched line.
/// Counting borrows instead, so peak memory is unchanged by the totals.
pub fn facet_totals(matches: &[Match], _scope: &Path) -> FacetTotals {
    let primary_pkg = primary_package(matches);
    let mut cache = PkgCache::new();
    let mut totals = FacetTotals::default();

    for m in matches {
        let slot = match facet_of(m, primary_pkg.as_ref(), &mut cache) {
            Facet::Implementation => &mut totals.implementations,
            Facet::Definition => &mut totals.definitions,
            Facet::Test => &mut totals.tests,
            Facet::UsageLocal => &mut totals.usages_local,
            Facet::UsageCross => &mut totals.usages_cross,
        };
        *slot += 1;
    }

    totals
}

/// Test-ness of a single match, for callers computing facet totals during a walk.
///
/// `content::search` needs this to keep its `tests` count exact while retaining only a
/// bounded number of matches. Exposed rather than duplicated so the count cannot drift from
/// the bucket the renderer puts the match in.
pub(crate) fn is_test_match_for_totals(m: &Match) -> bool {
    is_test_match(m)
}

/// Check if a match is in a test file or contains test markers.
fn is_test_match(m: &Match) -> bool {
    // Path-based detection
    let path_str = m.path.to_string_lossy();
    if path_str.contains("_test.")
        || path_str.contains("/test/")
        || path_str.contains("/tests/")
        || path_str.contains("_spec.")
        || path_str.contains("/spec/")
    {
        return true;
    }

    // Content-based detection
    let text = &m.text;
    text.contains("#[test]")
        || text.contains("#[cfg(test)]")
        || text.contains("@Test")
        || text.contains("def test_")
        || text.contains("it(\"")
        || text.contains("it('")
        || text.contains("describe(\"")
        || text.contains("describe('")
        || text.contains("func Test")
}

/// Check if path is in the same package as the primary definition.
fn is_same_package(path: &Path, primary_pkg: Option<&PathBuf>) -> bool {
    let Some(pkg_root) = primary_pkg else {
        return false;
    };

    path.parent()
        .and_then(crate::lang::package_root)
        .is_some_and(|p| p == pkg_root.as_path())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    fn m(
        path: &str,
        line: u32,
        text: &str,
        is_definition: bool,
        impl_target: Option<&str>,
    ) -> Match {
        Match {
            path: PathBuf::from(path),
            line,
            text: text.to_string(),
            is_definition,
            exact: true,
            file_lines: 100,
            mtime: SystemTime::UNIX_EPOCH,
            def_range: None,
            def_name: None,
            def_weight: if is_definition { 60 } else { 0 },
            impl_target: impl_target.map(str::to_string),
        }
    }

    /// `facet_totals` and `facet_matches` are rendered side by side as `shown/total`, so a
    /// disagreement between them would read as a truncation that never happened. They share
    /// `facet_of` for exactly that reason; this asserts the sharing actually holds for every
    /// facet, so reintroducing a second copy of the classification fails here.
    ///
    /// The fixture is written to **disk**, with a real manifest. An earlier version used
    /// synthetic paths that existed only in memory, which made `package_root` return `None`
    /// for every match on both sides: the `usages_local` assertion was `0 == 0`, and the
    /// local-vs-cross split — the one part of the classification that touches the
    /// filesystem, and the only place the two functions derive `primary_pkg` differently
    /// (one from a consumed `Vec`, one from a borrowed slice) — was never exercised at all.
    #[test]
    fn facet_totals_agrees_with_facet_matches_bucket_for_bucket() {
        let root = tempfile::tempdir().unwrap();
        let scope = root.path();

        // Two packages, each with its own manifest, so `package_root` resolves differently
        // for `inside/` than for `outside/`.
        let inside = scope.join("inside");
        let outside = scope.join("outside");
        std::fs::create_dir_all(&inside).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(inside.join("Cargo.toml"), "[package]\nname = \"inside\"\n").unwrap();
        std::fs::write(
            outside.join("Cargo.toml"),
            "[package]\nname = \"outside\"\n",
        )
        .unwrap();
        for f in ["lib.rs", "lib_test.rs", "other.rs"] {
            std::fs::write(inside.join(f), "fn thing() {}\n").unwrap();
        }
        std::fs::write(outside.join("far.rs"), "fn thing() {}\n").unwrap();

        let lib = inside.join("lib.rs");
        let matches = vec![
            m(lib.to_str().unwrap(), 1, "pub fn thing() {}", true, None),
            m(
                lib.to_str().unwrap(),
                9,
                "impl Trait for Thing {}",
                true,
                Some("Trait"),
            ),
            m(
                inside.join("lib_test.rs").to_str().unwrap(),
                3,
                "thing();",
                false,
                None,
            ),
            m(
                inside.join("other.rs").to_str().unwrap(),
                4,
                "    thing();",
                false,
                None,
            ),
            m(
                outside.join("far.rs").to_str().unwrap(),
                7,
                "thing();",
                false,
                None,
            ),
            m(lib.to_str().unwrap(), 12, "    thing();", false, None),
        ];

        let totals = facet_totals(&matches, scope);
        let grouped = facet_matches(matches, scope);

        assert_eq!(totals.definitions, grouped.definitions.len(), "definitions");
        assert_eq!(
            totals.implementations,
            grouped.implementations.len(),
            "implementations"
        );
        assert_eq!(totals.tests, grouped.tests.len(), "tests");
        assert_eq!(
            totals.usages_local,
            grouped.usages_local.len(),
            "usages_local"
        );
        assert_eq!(
            totals.usages_cross,
            grouped.usages_cross.len(),
            "usages_cross"
        );

        // Every bucket must be non-empty, or an assertion above is `0 == 0`. Counted
        // separately per usage bucket — summing them would mask the local/cross split,
        // which is the one this test exists to cover.
        for (label, n) in [
            ("definitions", totals.definitions),
            ("implementations", totals.implementations),
            ("tests", totals.tests),
            ("usages_local", totals.usages_local),
            ("usages_cross", totals.usages_cross),
        ] {
            assert!(
                n > 0,
                "{label} bucket is empty — assertion would be vacuous"
            );
        }
    }
}
