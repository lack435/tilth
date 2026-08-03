//! Translating Perforce ignore syntax into gitignore syntax.
//!
//! Perforce is the norm for Unreal work and Epic ships a well-maintained `.p4ignore`
//! covering exactly the directories that make a UE tree unwalkable — `*/Intermediate/*`,
//! `*/Saved/*`, `*/Binaries/*`, `*/obj/*`.
//!
//! # Why translate rather than alias the filename
//!
//! `WalkBuilder::add_custom_ignore_filename(".p4ignore")` is one line but applies
//! **gitignore** semantics to **P4IGNORE** syntax, and Epic's own file opens by warning the
//! two differ. The difference that matters is anchoring:
//!
//! - gitignore anchors any pattern containing a non-trailing `/` to the ignore file's
//!   directory, so `*/Intermediate/*` matches `Engine/Intermediate/x` and nothing deeper.
//! - P4IGNORE patterns are unanchored unless they begin with `/`; they match any path suffix
//!   on segment boundaries, so the same pattern matches at every depth.
//!
//! Measured on a real UE5 checkout: of 133,279 files under
//! `Intermediate|Saved|Binaries|DerivedDataCache`, **50,893 sit three or more segments
//! deep**. Aliasing would silently fail to ignore 38% of the build output, and silent
//! under-ignoring is the worst failure mode because it looks like it worked.
//!
//! # The rules
//!
//! | P4 construct | emitted gitignore |
//! |---|---|
//! | `#…`, blank | dropped |
//! | leading `!` | preserved, remainder translated |
//! | starts with `/` (anchored) | unchanged |
//! | anything else (unanchored) | prefixed `**/` |
//! | a non-trailing `**` segment | `*/**` |
//!
//! The last rule reimposes a floor: P4's `**` means *one or more* segments, git's means
//! *zero or more*. A trailing `**` is left alone — no following segment for the floor to
//! apply to, and git's trailing `/**` already agrees with P4.
//!
//! # Provenance, and two traps
//!
//! Validated against `p4 ignores -i` as oracle: 20 real Epic pattern shapes × 36 paths =
//! 720 checks, 0 mismatches. The table is encoded below as exact-string assertions.
//!
//! - `p4 ignores -i` prints **only** when a path IS ignored, so "no output" reads as "not
//!   ignored" — and with a **relative** path argument it exits 0 printing nothing *always*.
//!   A broken invocation is indistinguishable from "nothing is ignored"; it once produced a
//!   wall of 28 confident false mismatches. Pass absolute paths and canary the harness on a
//!   known-ignored file before trusting a single negative.
//! - Applying the `**` floor as two sequential string replaces makes the second rewrite
//!   re-process what the first inserted (`*/**/` → `*/*/**/`), quietly adding a mandatory
//!   segment. It is done segment-wise in one pass for that reason.

/// Translate one `.p4ignore` line into gitignore syntax.
///
/// Returns `None` for comments and blank lines.
pub(crate) fn translate_line(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return None;
    }

    let (negated, pattern) = match trimmed.strip_prefix('!') {
        Some(rest) => (true, rest),
        None => (false, trimmed),
    };
    if pattern.is_empty() {
        return None;
    }

    let anchored = pattern.starts_with('/');
    let body = pattern.strip_prefix('/').unwrap_or(pattern);

    // One pass over the segments. See the module header for why this is not two replaces.
    let segments: Vec<&str> = body.split('/').collect();
    let last = segments.len().saturating_sub(1);
    let rebuilt = segments
        .iter()
        .enumerate()
        .map(|(i, seg)| {
            if *seg == "**" && i < last {
                "*/**"
            } else {
                seg
            }
        })
        .collect::<Vec<_>>()
        .join("/");

    let out = if anchored {
        format!("/{rebuilt}")
    } else {
        format!("**/{rebuilt}")
    };
    Some(if negated { format!("!{out}") } else { out })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The oracle table, verbatim: every distinct pattern shape in Epic's shipped
    /// `.p4ignore` / `.p4ignore.txt`, with the translation `p4 ignores -i` agreed with
    /// across 36 probe paths. Deviating from any right-hand side means deviating from
    /// Perforce, so these are exact-string assertions rather than behavioural ones.
    #[test]
    fn translations_match_the_p4_oracle() {
        let cases = [
            // Anchored patterns pass through — same meaning in both syntaxes.
            ("/*.sln", "/*.sln"),
            ("/Engine/DerivedDataCache/*", "/Engine/DerivedDataCache/*"),
            (
                "/Engine/Source/Programs/Shared/EpicGames.*/bin/*",
                "/Engine/Source/Programs/Shared/EpicGames.*/bin/*",
            ),
            // Unanchored: prefixed so git stops anchoring them to the ignore file's dir.
            ("*.suo", "**/*.suo"),
            ("*.pyc", "**/*.pyc"),
            (".DS_Store", "**/.DS_Store"),
            (".vs", "**/.vs"),
            (".idea/", "**/.idea/"),
            ("*/Intermediate/*", "**/*/Intermediate/*"),
            ("*/Saved/*", "**/*/Saved/*"),
            ("*/Binaries/*", "**/*/Binaries/*"),
            ("*/obj/*", "**/*/obj/*"),
            ("*/__pycache__/*", "**/*/__pycache__/*"),
            ("*/.git/*", "**/*/.git/*"),
            (
                "*/BlueprintAssist/NodeSizeCache/*",
                "**/*/BlueprintAssist/NodeSizeCache/*",
            ),
            ("EpicGames.*/bin/*", "**/EpicGames.*/bin/*"),
            (
                "Engine/Programs/UnrealBuildTool/*",
                "**/Engine/Programs/UnrealBuildTool/*",
            ),
            // `**` floor: one or more segments in P4, zero or more in git.
            (
                "**/DerivedDataCache/Boot.ddc",
                "**/*/**/DerivedDataCache/Boot.ddc",
            ),
            (
                "**/DerivedDataCache/**/*.udd",
                "**/*/**/DerivedDataCache/*/**/*.udd",
            ),
            // Trailing `**` is left alone — no following segment to floor.
            (
                "UE/Bobcat/Content/__ExternalActors__/**",
                "**/UE/Bobcat/Content/__ExternalActors__/**",
            ),
        ];
        for (p4, want) in cases {
            assert_eq!(
                translate_line(p4).as_deref(),
                Some(want),
                "translation drifted from the p4-validated table for {p4:?}"
            );
        }
    }

    #[test]
    fn comments_and_blanks_are_dropped() {
        assert_eq!(translate_line(""), None);
        assert_eq!(translate_line("   "), None);
        assert_eq!(translate_line("# a comment"), None);
        assert_eq!(translate_line("  # indented comment"), None);
        // A bare "!" has nothing to negate.
        assert_eq!(translate_line("!"), None);
    }

    #[test]
    fn negation_is_preserved_around_the_translation() {
        assert_eq!(
            translate_line("!**/ThirdParty/LibJuice/**").as_deref(),
            Some("!**/*/**/ThirdParty/LibJuice/**")
        );
        assert_eq!(
            translate_line("!**/OnlineSubsystemPlayFab/Platforms/*/Lib").as_deref(),
            Some("!**/*/**/OnlineSubsystemPlayFab/Platforms/*/Lib")
        );
        // Anchored negation keeps its anchor rather than gaining a `**/`.
        assert_eq!(
            translate_line("!/build.log").as_deref(),
            Some("!/build.log")
        );
    }

    /// The single-pass rule, pinned directly. Two sequential string replaces produce
    /// `*/*/**/` here — an extra mandatory path segment, which silently stops the
    /// pattern matching at the depth P4 matches it at.
    #[test]
    fn double_star_floor_is_applied_once_not_twice() {
        let got = translate_line("**/DerivedDataCache/Boot.ddc").unwrap();
        assert_eq!(got, "**/*/**/DerivedDataCache/Boot.ddc");
        // `/*/*/` is the signature of the doubled rewrite (`**/*/*/**/…`). Note the
        // correct output DOES contain `*/*/**` as a substring — `**/*/**` — so that is
        // not a usable discriminator, which this assertion originally got wrong.
        assert!(
            !got.contains("/*/*/"),
            "the `**` floor was applied twice, adding a mandatory segment: {got}"
        );
    }
}
