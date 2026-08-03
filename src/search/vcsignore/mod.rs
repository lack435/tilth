//! Which files a walk skips because the checkout's own VCS says they are generated.
//!
//! # Why this owns the matching instead of delegating
//!
//! `ignore` can apply `.gitignore` itself, and an earlier attempt let it, adding a second
//! matcher in `filter_entry` for `.p4ignore`. That was reverted after review, for two
//! reasons that are properties of the arrangement rather than bugs in it:
//!
//! 1. **The crate's matcher runs before `filter_entry`** (`walk.rs:1108` sequential,
//!    `:1838` parallel). Anything a `.gitignore` at or below the walk root excluded never
//!    reached our code, so we could not count it, and could not tell the caller a file had
//!    been hidden. That is the default scope of every tool. Measured: `tilth_search
//!    kind=callers` reported "1 call site" for a symbol with 2, with no note at all.
//! 2. **A second matcher can only subtract.** Layered on top of a correct verdict it cannot
//!    re-include anything, so nested `.gitignore` negation was unreachable:
//!    `root/.gitignore: *.json` plus `root/config/.gitignore: !settings.json` returned zero
//!    files where git includes `settings.json`.
//!
//! Both dissolve if the matching is ours: every decision passes through one place, so it is
//! observable, and precedence is ours to get right.
//!
//! # Precedence
//!
//! The rule, which the reverted attempt got wrong by using `.any()` over its matchers:
//! applicable ignore files are consulted **shallowest first, and the last file to express an
//! opinion wins**. `.any()` treats a `!` re-include as "no opinion" and ORs the ignores, so
//! a deeper whitelist can never win — which is exactly the nested-negation failure above.
//! [`Gitignore::matched`] already reports `Whitelist` for `!`; the fix is to stop discarding
//! it.
//!
//! Git's other rule — a file cannot be re-included once an ancestor directory is excluded —
//! is carried by [`DirRules::excluded`]. It was first written off as needing no code, on the
//! reasoning that a walk prunes excluded directories and never offers their contents. True
//! of a walk, and irrelevant to a function: the oracle asks about paths directly and caught
//! it at once, reporting `build/` ignored and `build/x.rs` kept. Correctness that depends on
//! the caller's traversal order is not correctness.
//!
//! # git and Perforce are combined by OR, not by precedence
//!
//! Within one syntax, precedence decides. Across the two, a path is skipped if *either*
//! says so, and a `!` in one cannot re-include what the other excluded. A tree carrying both
//! files has two independent statements about what is generated; letting a `.p4ignore`
//! negation quietly override a `.gitignore` rule would be surprising, and on the UE checkout
//! measured here the two files genuinely differ.
//!
//! # Bounded upward
//!
//! Ancestor ignore files apply, but only up to the workspace root. Unbounded, this silenced
//! in-repo source from a `.gitignore` outside the project entirely — something
//! `git check-ignore` reports as not ignored. `ignore`'s own parent scan has the same hole
//! once `require_git(false)` is set: `Dir::add_parents` walks to the filesystem root and the
//! `saw_git` guard that would stop it is dead. Hence `parents(false)` on the builder and our
//! own bounded ascent.
//!
//! # Trust
//!
//! Owning the matching means owning the risk of diverging from git. The [`oracle`] module
//! builds real repositories and compares **every** verdict against `git check-ignore
//! --no-index` itself — the same differential method that made the P4 translation
//! trustworthy at 720/720. It paid for itself on the first run, finding a whole class of
//! divergence (7 of 72 verdicts) that the hand-written unit tests above all passed.

#[cfg(test)]
mod oracle;
pub(crate) mod translate;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use dashmap::DashMap;
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use ignore::Match;

/// Names that mark a directory as the top of a checkout. The ascent stops at the first
/// ancestor carrying one, that ancestor included.
const WORKSPACE_MARKERS: &[&str] = &[
    ".git",
    ".p4config",
    "Cargo.toml",
    "go.mod",
    "package.json",
    "pyproject.toml",
    "Default.uprojectdirs",
];

/// Ignore filenames per syntax. `.p4ignore.txt` is consulted alongside `.p4ignore` because
/// `p4 set P4IGNORE=` writes the registry on Windows rather than the process environment, so
/// `std::env::var` cannot see it — and Epic's UE setup instructs exactly
/// `p4 set P4IGNORE=.p4ignore.txt`. Reading only the env var missed the shipped file on
/// precisely the checkouts this targets.
const GIT_NAMES: &[&str] = &[".gitignore"];
const P4_NAMES: &[&str] = &[".p4ignore", ".p4ignore.txt"];

/// One syntax's matchers for one directory, ordered shallowest first.
///
/// `Arc<Gitignore>`, not `Gitignore`. Every directory's stack is built by extending its
/// parent's, so a plain `Vec<Gitignore>` deep-copies each ancestor's **compiled `GlobSet`**
/// once per directory. On a UE checkout that is 26,694 copies of a 191-glob set — measured
/// at 6.6s of the 6.9s total overhead, while a warm `is_ignored` costs 2.14us. Refcounts
/// make extending a stack proportional to its depth rather than its content.
type Layer = Vec<Arc<Gitignore>>;

/// What one directory contributes, and whether it is itself excluded.
struct DirRules {
    git: Layer,
    p4: Layer,
    /// Is this directory excluded — by its own ancestors' rules, or by theirs?
    ///
    /// Carried rather than recomputed because it is the whole of git's "a file cannot be
    /// re-included once an ancestor directory is excluded" rule, and computing it per path
    /// would mean re-walking the ancestor chain on every entry. Defined recursively as
    /// `parent.excluded || <parent's rules say this directory is ignored>`, so one cached
    /// lookup on a path's parent answers for the entire chain above it.
    excluded: bool,
}

/// The ignore rules in force for a walk, resolved lazily per directory.
pub(crate) struct VcsIgnore {
    /// Highest directory the ascent may reach, inclusive.
    ceiling: PathBuf,
    cache: DashMap<PathBuf, Arc<DirRules>>,
    skipped: AtomicUsize,
}

impl VcsIgnore {
    /// Resolve the rules that apply to `scope`, or `None` if the tree carries none.
    pub(crate) fn for_scope(scope: &Path) -> Option<Self> {
        let start = if scope.is_dir() {
            scope.to_path_buf()
        } else {
            scope.parent()?.to_path_buf()
        };
        let ceiling = workspace_root(&start);
        let me = Self {
            ceiling,
            cache: DashMap::new(),
            skipped: AtomicUsize::new(0),
        };
        // A tree with no ignore file anywhere above the scope may still have them below, so
        // this cannot conclude "none" from the ancestors alone — but it is worth answering
        // cheaply for the common case of neither.
        let r = me.rules_for(&start);
        if r.git.is_empty() && r.p4.is_empty() && !any_ignore_file_in(&start) {
            return None;
        }
        Some(me)
    }

    /// Should the walk skip this path?
    ///
    /// Git's rule has two parts and both are needed. A path under an excluded directory is
    /// excluded no matter what any deeper pattern says — that is what stops a `!` re-include
    /// from reaching inside a pruned tree. Only when no ancestor excludes it does the path's
    /// own verdict decide.
    ///
    /// The ancestor half is not optional even though a walk prunes excluded directories and
    /// would never offer their contents: the differential oracle asks about paths directly,
    /// and it found this — `build/` ignored, `build/x.rs` reported as kept. Relying on the
    /// caller's traversal order for a correct answer makes the function wrong for anyone who
    /// asks a different way.
    pub(crate) fn is_ignored(&self, path: &Path, is_dir: bool) -> bool {
        let Some(dir) = path.parent() else {
            return false;
        };
        let rules = self.rules_for(dir);
        if rules.excluded {
            return true;
        }
        // OR across syntaxes, precedence within each. See the module header.
        verdict(&rules.git, path, is_dir) || verdict(&rules.p4, path, is_dir)
    }

    /// Record a skip. Separate from [`VcsIgnore::is_ignored`] so the query stays
    /// side-effect free and the count means "paths the walk actually declined", not "paths
    /// we asked about".
    pub(crate) fn note_skipped(&self) {
        self.skipped.fetch_add(1, Ordering::Relaxed);
        note_skipped_globally();
    }

    #[cfg(test)]
    pub(crate) fn skipped(&self) -> usize {
        self.skipped.load(Ordering::Relaxed)
    }

    /// The rules for `dir`: its own ignore files stacked on its parent's, plus whether the
    /// directory is itself excluded.
    ///
    /// Handed out as an `Arc` rather than behind a `DashMap` guard: holding a reference
    /// across the recursive parent lookup would deadlock on a shard this thread already
    /// holds. And an `Arc` rather than a value, because `is_ignored` runs once per walked
    /// entry and every `Gitignore` in the stack carries a compiled `GlobSet` — returning
    /// `DirRules` by value cost **313s against 30s** on a 444k-file checkout, a 10x
    /// regression that no amount of matching cleverness would have paid back.
    fn rules_for(&self, dir: &Path) -> Arc<DirRules> {
        if let Some(hit) = self.cache.get(dir) {
            return Arc::clone(&hit);
        }
        let at_ceiling = dir == self.ceiling || !dir.starts_with(&self.ceiling);
        let parent = if at_ceiling { None } else { dir.parent() };

        let (mut git, mut p4, excluded) = match parent {
            Some(p) => {
                let up = self.rules_for(p);
                // A directory is excluded if its parent chain is, or if the rules in force
                // *above* it say so. Its own ignore files cannot exempt it — git does not
                // read them once the directory is out.
                let self_excluded =
                    up.excluded || verdict(&up.git, dir, true) || verdict(&up.p4, dir, true);
                (up.git.clone(), up.p4.clone(), self_excluded)
            }
            None => (Vec::new(), Vec::new(), false),
        };
        if let Some(m) = build(dir, GIT_NAMES, false) {
            git.push(Arc::new(m));
        }
        if let Some(m) = build(dir, P4_NAMES, true) {
            p4.push(Arc::new(m));
        }
        let rules = Arc::new(DirRules { git, p4, excluded });
        self.cache.insert(dir.to_path_buf(), Arc::clone(&rules));
        rules
    }
}

/// Shallowest first, last opinion wins.
fn verdict(layer: &Layer, path: &Path, is_dir: bool) -> bool {
    let mut ignored = false;
    for m in layer {
        match m.matched(path, is_dir) {
            Match::Ignore(_) => ignored = true,
            Match::Whitelist(_) => ignored = false,
            Match::None => {}
        }
    }
    ignored
}

/// Build one directory's matcher for a syntax, or `None` if it has no such file.
///
/// All files of a syntax in one directory fold into a single `Gitignore`, so ordering
/// between `.p4ignore` and `.p4ignore.txt` is the order they are listed.
fn build(dir: &Path, names: &[&str], translate: bool) -> Option<Gitignore> {
    let mut b = GitignoreBuilder::new(dir);
    let mut added = 0usize;
    for name in names {
        let Ok(contents) = std::fs::read_to_string(dir.join(name)) else {
            continue;
        };
        for line in contents.lines() {
            let pattern = if translate {
                match translate::translate_line(line) {
                    Some(p) => p,
                    None => continue,
                }
            } else {
                let t = line.trim();
                if t.is_empty() || t.starts_with('#') {
                    continue;
                }
                t.to_string()
            };
            // A pattern the source VCS accepts but globset rejects is skipped rather than
            // failing the whole file: one bad line must not silently drop every other rule.
            if b.add_line(None, &pattern).is_ok() {
                added += 1;
            }
        }
    }
    (added > 0).then(|| b.build().ok())?
}

// ── Per-request state ────────────────────────────────────────────────────────────────
//
// A walk's matcher is per-walk, but the caller-facing report is per-request and is emitted
// somewhere that has no handle on any matcher. These carry it across, reset at the top of a
// tool call exactly as the walk budget is.

/// Paths declined this request, and the generation they were declined under.
///
/// The generation is `walkbudget`'s, deliberately reused rather than duplicated: it answers
/// "which request does this walk belong to", is bumped in the same place, and a second
/// counter with its own notion of the same thing would be one more thing to keep in step.
/// Without it, a worker abandoned by an earlier timeout keeps pruning and its skips land on
/// the next request's report — telling a caller that files were hidden from a search that
/// hid nothing, and pointing them at an escape hatch they do not need.
static SKIPPED: AtomicUsize = AtomicUsize::new(0);
static SKIPPED_GEN: AtomicUsize = AtomicUsize::new(0);
static INCLUDE_IGNORED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Begin a request: clear the count and adopt the current generation.
pub(crate) fn begin_request(include_ignored: bool) {
    SKIPPED.store(0, Ordering::Relaxed);
    SKIPPED_GEN.store(crate::walkbudget::generation(), Ordering::Relaxed);
    INCLUDE_IGNORED.store(include_ignored, Ordering::Relaxed);
}

pub(crate) fn include_ignored() -> bool {
    INCLUDE_IGNORED.load(Ordering::Relaxed)
}

fn note_skipped_globally() {
    if crate::walkbudget::generation() == SKIPPED_GEN.load(Ordering::Relaxed) {
        SKIPPED.fetch_add(1, Ordering::Relaxed);
    }
}

/// What this request hid, as caller-facing prose. `None` when it hid nothing.
///
/// Honouring ignore files is ripgrep's contract and a good default, but on a game checkout
/// it hides material an agent may genuinely need, and silence about it is the failure mode
/// this codebase keeps having to fix. Measured on a real UE5 tree: **451** first-party
/// `Engine/Plugins/*/Content/Python/**` editor scripts (UE ships executable Python inside
/// `Content/`, which Epic's `.gitignore` excludes wholesale), **7,665** bundled `CPython`
/// stdlib files, **330** YAML including 43 first-party infra-as-code — those lost to an
/// upstream authoring bug, `!.yaml` where `!*.yaml` was meant, which faithful honouring
/// propagates — and ~290 vendored C sources, with a demonstrated symbol-search regression.
/// On tilth's own repo the same survey lost zero tracked files.
///
/// So the count is reported and the recovery is named in the same breath. Without it an
/// agent only discovers the omission if it already suspected the file existed, which is the
/// position a search tool is supposed to get you out of.
///
/// **A pruned directory counts once, not its contents** — the walk never enumerates inside
/// it, so the contents are not knowable without doing the work the prune exists to avoid.
/// The note says so rather than implying the number is a file count.
pub(crate) fn skipped_note() -> Option<String> {
    let n = SKIPPED.load(Ordering::Relaxed);
    if n == 0 {
        return None;
    }
    Some(format!(
        "NOTE: {n} path(s) skipped — .gitignore/.p4ignore excludes them (a skipped directory \
         counts once, not its contents). If something you expected is missing, retry with \
         include_ignored: true.\n\n"
    ))
}

fn any_ignore_file_in(dir: &Path) -> bool {
    GIT_NAMES
        .iter()
        .chain(P4_NAMES.iter())
        .any(|n| dir.join(n).exists())
}

/// Nearest ancestor that looks like the top of a checkout, or the filesystem root.
fn workspace_root(start: &Path) -> PathBuf {
    let mut cursor = Some(start);
    let mut last = start;
    while let Some(d) = cursor {
        last = d;
        if WORKSPACE_MARKERS.iter().any(|m| d.join(m).exists()) {
            return d.to_path_buf();
        }
        cursor = d.parent();
    }
    last.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(root: &Path, rel: &str, body: &str) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, body).unwrap();
    }

    /// The nested-negation case the reverted design could not express.
    ///
    /// `.any()` over independent matchers treats `!settings.json` as "no opinion" and ORs
    /// the ignores, so the deeper whitelist never wins. git includes `settings.json`.
    #[test]
    fn a_deeper_negation_re_includes_what_a_shallower_rule_ignored() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write(root, ".git/HEAD", "ref: refs/heads/main\n");
        write(root, ".gitignore", "*.json\n");
        write(root, "config/.gitignore", "!settings.json\n");
        write(root, "config/settings.json", "{}\n");
        write(root, "config/other.json", "{}\n");

        let vi = VcsIgnore::for_scope(root).expect("rules exist");
        assert!(
            !vi.is_ignored(&root.join("config/settings.json"), false),
            "a deeper `!` must re-include; last opinion wins"
        );
        assert!(
            vi.is_ignored(&root.join("config/other.json"), false),
            "the shallower rule still applies to everything the deeper file is silent on"
        );
    }

    /// Both syntaxes apply, and a `!` in one does not reach across into the other.
    #[test]
    fn git_and_p4_are_combined_by_or() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write(root, ".git/HEAD", "ref: refs/heads/main\n");
        write(root, ".gitignore", "secret.txt\n");
        write(root, ".p4ignore", "!secret.txt\n*/Intermediate/*\n");
        write(root, "secret.txt", "x\n");
        write(root, "Engine/Intermediate/gen.cpp", "void G(){}\n");
        // THREE segments before `Intermediate`. A one-level fixture cannot detect the
        // anchoring bug this translation exists to fix: under gitignore semantics
        // `*/Intermediate/*` still matches `Engine/Intermediate/x` — one segment, then
        // Intermediate, then the file — so the test passed with the fix reverted. It is
        // the deep case that diverges, and the deep case is 50,893 of the 133,279
        // build-output files on the measured UE checkout.
        write(
            root,
            "Engine/Plugins/Acme/Intermediate/Build/deep.cpp",
            "void D(){}\n",
        );

        let vi = VcsIgnore::for_scope(root).expect("rules exist");
        assert!(
            vi.is_ignored(&root.join("secret.txt"), false),
            "a p4 negation must not re-include what .gitignore excluded"
        );
        assert!(
            vi.is_ignored(&root.join("Engine/Intermediate/gen.cpp"), false),
            "the translated p4 rule must match one level down"
        );
        assert!(
            vi.is_ignored(
                &root.join("Engine/Plugins/Acme/Intermediate/Build/deep.cpp"),
                false
            ),
            "P4 patterns are unanchored and must match at ANY depth — this is the case \
             gitignore semantics get wrong, and the only one that distinguishes them"
        );
    }

    /// An ignore file outside the workspace must not reach in.
    #[test]
    fn the_ascent_stops_at_the_workspace_root() {
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), ".gitignore", "src/\n");
        let project = tmp.path().join("myrepo");
        write(&project, ".git/HEAD", "ref: refs/heads/main\n");
        write(&project, "src/foo.rs", "pub fn f() {}\n");

        let vi = VcsIgnore::for_scope(&project);
        let ignored = vi.is_some_and(|v| v.is_ignored(&project.join("src/foo.rs"), false));
        assert!(
            !ignored,
            "a .gitignore above the repo silenced in-repo source; git says it does not apply"
        );
    }

    /// Epic's UE idiom: deny everything, re-include directories and source.
    ///
    /// This is the shape that defeated the reverted design's probe heuristic and turned the
    /// whole feature off below a UE root.
    #[test]
    fn the_ue_whitelist_idiom_resolves_the_way_git_resolves_it() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write(root, ".git/HEAD", "ref: refs/heads/main\n");
        write(root, ".gitignore", "*\n!*/\n!*.cpp\n!*.h\n");
        write(root, "Engine/Source/real.cpp", "void R(){}\n");
        write(root, "Engine/Source/notes.txt", "x\n");

        let vi = VcsIgnore::for_scope(root).expect("rules exist");
        assert!(
            !vi.is_ignored(&root.join("Engine"), true),
            "`!*/` re-includes directories, so descent must continue"
        );
        assert!(
            !vi.is_ignored(&root.join("Engine/Source/real.cpp"), false),
            "`!*.cpp` re-includes source"
        );
        assert!(
            vi.is_ignored(&root.join("Engine/Source/notes.txt"), false),
            "an extension with no whitelist line stays ignored, as git has it"
        );
    }

    /// Hiding a path must be reported, and the report must name the way back.
    ///
    /// The whole value of honouring ignore files is that it hides things; the whole risk is
    /// hiding the wrong thing. On a real UE tree the default hides 451 first-party
    /// `Content/Python` editor scripts, so "0 files" with no note is a lie the caller has no
    /// way to catch — and, unlike the reverted design, this one can see every skip because
    /// nothing is excluded behind its back.
    #[test]
    fn skipping_a_path_is_reported() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write(root, ".git/HEAD", "ref: refs/heads/main\n");
        write(root, ".gitignore", "Saved/\n");
        write(root, "Saved/hidden.rs", "fn h() {}\n");
        write(root, "kept.rs", "fn k() {}\n");

        begin_request(false); // also zeroes the counter
        let seen: Vec<String> = crate::search::base_walk_builder(root)
            .build()
            .filter_map(Result::ok)
            .filter_map(|e| e.file_name().to_str().map(str::to_string))
            .collect();
        assert!(seen.contains(&"kept.rs".to_string()));
        assert!(!seen.contains(&"hidden.rs".to_string()));

        let note = skipped_note().expect("hiding a path must be reported");
        assert!(
            note.contains("include_ignored"),
            "the report must name the way back: {note}"
        );
    }

    #[test]
    fn a_tree_with_no_ignore_files_costs_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), "src/a.rs", "fn a() {}\n");
        assert!(
            VcsIgnore::for_scope(tmp.path()).is_none(),
            "no ignore files anywhere: the walk should not carry a matcher at all"
        );
    }
}
