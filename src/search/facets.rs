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

/// Single source of truth for facet assignment. `facet_matches` and `facet_totals`
/// both route through this so a count can never disagree with what gets grouped —
/// they are rendered side by side as `shown/total`, so a drift between them would
/// read as a truncation that did not happen.
fn facet_of(m: &Match, primary_pkg: Option<&PathBuf>) -> Facet {
    if m.is_definition && m.impl_target.is_some() {
        Facet::Implementation
    } else if m.is_definition {
        Facet::Definition
    } else if is_test_match(m) {
        Facet::Test
    } else if is_same_package(&m.path, primary_pkg) {
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

    let mut definitions = Vec::new();
    let mut implementations = Vec::new();
    let mut tests = Vec::new();
    let mut usages_local = Vec::new();
    let mut usages_cross = Vec::new();

    for m in matches {
        match facet_of(&m, primary_pkg.as_ref()) {
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
    let mut totals = FacetTotals::default();

    for m in matches {
        let slot = match facet_of(m, primary_pkg.as_ref()) {
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
    #[test]
    fn facet_totals_agrees_with_facet_matches_bucket_for_bucket() {
        let scope = Path::new("/repo");
        let matches = vec![
            m("/repo/src/lib.rs", 1, "pub fn thing() {}", true, None),
            m(
                "/repo/src/lib.rs",
                9,
                "impl Trait for Thing {}",
                true,
                Some("Trait"),
            ),
            m("/repo/src/lib_test.rs", 3, "thing();", false, None),
            m("/repo/src/other.rs", 4, "    thing();", false, None),
            m("/elsewhere/far.rs", 7, "thing();", false, None),
            m("/repo/src/lib.rs", 12, "    thing();", false, None),
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

        // And the fixture must actually exercise more than one bucket, or the assertions
        // above pass trivially.
        let non_empty = [
            totals.definitions,
            totals.implementations,
            totals.tests,
            totals.usages_local + totals.usages_cross,
        ]
        .iter()
        .filter(|n| **n > 0)
        .count();
        assert!(
            non_empty >= 3,
            "fixture must span several facets, got {totals:?}"
        );
    }
}
