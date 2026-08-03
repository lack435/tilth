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

/// A real repository boundary. The **nearest** of these wins.
///
/// Nearest, because that is what git does: a nested repository is its own repository and the
/// outer one's `.gitignore` does not reach into it. Taking the outermost instead — an
/// over-correction for the decapitation problem below — made tilth hide `inner/src/thing.rs`
/// under an outer repo's `*.rs` rule while `git check-ignore` reports it as not ignored.
///
/// Note what is deliberately NOT here, and why the list is only `.git`:
///
/// - `.p4ignore` says "there are rules here", never "the checkout starts here". Treating it
///   as a boundary decapitated the UE root, which carries a dozen nested `.p4ignore.txt`.
/// - `.p4config` is the same mistake in a subtler costume. P4CONFIG's whole design is
///   nearest-file lookup *from a subdirectory*, so a `.p4config` below a checkout root is
///   the normal Perforce idiom rather than an edge case. With it in this list, a `.git` repo
///   containing `sub/.p4config` stopped resolving its own root `.gitignore`: measured,
///   `repo/.gitignore: *.gen.cpp` no longer hid `repo/sub/src/a.gen.cpp`, which
///   `git check-ignore` reports as ignored.
///
/// `.git` alone, because it alone actually delimits a repository.
const REPO_MARKERS: &[&str] = &[".git"];

/// Weaker signals, used only when no [`REPO_MARKERS`] exists anywhere above.
///
/// A package manifest means "a project lives here", not "the checkout starts here", so the
/// **outermost** wins among these — the opposite of `REPO_MARKERS`, and for the same reason
/// in reverse. Stopping at the nearest one decapitated the UE root: that tree has no `.git`
/// at all, ships 14 nested `package.json` files, and scoping at
/// `Engine/Source/Programs/Horde/HordeDashboard` returned 279 files with no note against 229
/// for the same subtree reached from its parent.
///
/// These exist to bound an otherwise unbounded ascent in trees with no VCS root, nothing
/// more.
const PROJECT_MARKERS: &[&str] = &[
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
    /// Whether patterns match case-insensitively, resolved once from the checkout.
    case_insensitive: bool,
    /// The request this matcher was built for, captured once at construction.
    ///
    /// Compared at each skip so a walk abandoned by an earlier timeout cannot add to the
    /// NEXT request's report. The first version read the shared generation on both sides at
    /// call time, which compares a value to itself and can never reject anything.
    generation: usize,
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
        let ceiling_for_config = ceiling.clone();
        let me = Self {
            ceiling,
            cache: DashMap::new(),
            case_insensitive: resolve_ignorecase(&ceiling_for_config),
            generation: crate::walkbudget::generation(),
        };

        // The walk root is never excluded by rules from ABOVE it. A caller naming a
        // directory has said something more specific than a blanket rule that happens to
        // cover it, and answering `tilth_files --scope <repo>/Saved` with "0 files" is the
        // worst response available — `Saved`, `Intermediate` and `Binaries` are exactly the
        // directories someone scopes into deliberately.
        //
        // Implemented by seeding the root's own entry with `excluded: false` before any
        // lookup can populate it. Rules found at or below the root still apply, so this
        // clears the blanket exclusion without turning the feature off inside the scope.
        //
        // Residual, stated rather than papered over: this fixes rules that name the
        // directory (`Saved/`). A rule that names only its *contents* (`*/Intermediate/*`)
        // does not match the directory itself, so scoping directly at an `Intermediate`
        // still returns nothing. Closing that needs a probe for "would a child be
        // excluded?", and a probe of exactly that shape is what silently disabled the whole
        // feature under a UE root two attempts ago. Not reintroduced without a better idea.
        let seeded = me.rules_for(&start);
        me.cache.insert(
            start.clone(),
            Arc::new(DirRules {
                git: seeded.git.clone(),
                p4: seeded.p4.clone(),
                excluded: false,
            }),
        );

        // NOT short-circuited on "no ignore files above this directory". A `.gitignore`
        // deeper in the tree is discovered during descent and must still apply; concluding
        // "this tree has no rules" from the ancestors alone gave the same files opposite
        // verdicts depending on which directory the caller scoped at — 5 files from the
        // root, 2 from the subdirectory, with no note in the wrong-answer case.
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
    pub(crate) fn note_skipped(&self, path: &Path) {
        note_skipped(self.generation, path);
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
        if let Some(m) = build(dir, GIT_NAMES, false, self.case_insensitive) {
            git.push(Arc::new(m));
        }
        if let Some(m) = build(dir, P4_NAMES, true, self.case_insensitive) {
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
fn build(dir: &Path, names: &[&str], translate: bool, case_insensitive: bool) -> Option<Gitignore> {
    let mut b = GitignoreBuilder::new(dir);
    b.case_insensitive(case_insensitive).ok();
    let mut added = 0usize;
    for name in names {
        // Read as BYTES, not `read_to_string`. One invalid UTF-8 byte made `read_to_string`
        // return `Err`, the `let ... else` treated the file as absent, and the ENTIRE ignore
        // file was silently discarded — the exact failure the per-line comment below
        // promises not to commit, one level up. Lossy decoding keeps every line that is
        // valid and mangles only the one that is not.
        let Ok(bytes) = std::fs::read(dir.join(name)) else {
            continue;
        };
        let text = String::from_utf8_lossy(&bytes);
        // Strip a UTF-8 BOM. `str::trim` does not remove U+FEFF (it is not `White_Space`),
        // so the first pattern silently became `\u{FEFF}*.log` and matched nothing. Windows
        // PowerShell writes UTF-8 *with* BOM by default, so this is routine here, and git
        // strips it.
        let contents = text.strip_prefix('\u{FEFF}').unwrap_or(&text);
        for line in contents.lines() {
            let pattern = if translate {
                match translate::translate_line(line) {
                    Some(p) => p,
                    None => continue,
                }
            } else {
                // `trim_end`, not `trim`. Git strips trailing whitespace but **leading
                // whitespace is significant**, so trimming both hid a file git tracks:
                // `   leading.txt` became a live rule instead of matching a name that
                // starts with spaces.
                let t = line.trim_end();
                if t.trim().is_empty() || t.starts_with('#') {
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
/// A **set**, not a counter. One `tilth_search` performs several walks and each builds its
/// own matcher, so a counter reported the same path once per walk — exactly 2x on a two-walk
/// search, 12,480 for 6,240 real paths on a UE subtree. The note calls the number "path(s)",
/// so it has to be paths.
///
/// Capped at [`MAX_TRACKED_SKIPS`], past which the note says "at least". The job is to tell
/// a caller something was hidden, not to hold a million paths in memory to make the number
/// exact.
///
/// The generation is `walkbudget`'s, deliberately reused rather than duplicated: it answers
/// "which request does this walk belong to" and is bumped in the same place. Each matcher
/// captures it at construction and passes it back in — comparing the *shared* value on both
/// sides, as the first version did, compares a value to itself and can never reject
/// anything. Without a working guard, a worker abandoned by an earlier timeout keeps pruning
/// and its skips land on the next request's report.
const MAX_TRACKED_SKIPS: usize = 50_000;
static SKIPPED: std::sync::LazyLock<DashMap<PathBuf, ()>> = std::sync::LazyLock::new(DashMap::new);
static SKIPPED_OVERFLOW: AtomicUsize = AtomicUsize::new(0);
static SKIPPED_GEN: AtomicUsize = AtomicUsize::new(0);
static INCLUDE_IGNORED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Begin a request: clear the record and adopt the current generation.
pub(crate) fn begin_request(include_ignored: bool) {
    SKIPPED.clear();
    SKIPPED_OVERFLOW.store(0, Ordering::Relaxed);
    SKIPPED_GEN.store(crate::walkbudget::generation(), Ordering::Relaxed);
    INCLUDE_IGNORED.store(include_ignored, Ordering::Relaxed);
}

pub(crate) fn include_ignored() -> bool {
    INCLUDE_IGNORED.load(Ordering::Relaxed)
}

fn note_skipped(walk_generation: usize, path: &Path) {
    if walk_generation != SKIPPED_GEN.load(Ordering::Relaxed) {
        return;
    }
    if SKIPPED.len() >= MAX_TRACKED_SKIPS {
        SKIPPED_OVERFLOW.fetch_add(1, Ordering::Relaxed);
        return;
    }
    SKIPPED.insert(path.to_path_buf(), ());
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
    let n = SKIPPED.len();
    if n == 0 {
        return None;
    }
    let at_least = if SKIPPED_OVERFLOW.load(Ordering::Relaxed) > 0 {
        "at least "
    } else {
        ""
    };
    Some(format!(
        "NOTE: {at_least}{n} path(s) skipped — .gitignore/.p4ignore excludes them (a skipped \
         directory counts once, not its contents). If something you expected is missing, \
         retry with include_ignored: true.\n\n"
    ))
}

/// Does this checkout match ignore patterns case-insensitively?
///
/// Read from the repository's own `core.ignorecase`, not from `cfg!(windows)`. The platform
/// is only git's *default*: `git init` sets `ignorecase=true` on NTFS and APFS and `false`
/// on ext4, and a user may set either anywhere. Hard-coding the platform over-hid on the
/// dangerous side — with `core.ignorecase=false` on Windows, `git check-ignore` reports
/// `src/a.log` as NOT ignored under a `*.LOG` rule while tilth hid it, and no repo could opt
/// out because the value was a compile-time constant.
///
/// Falls back to the platform default when there is no git config to read, which is the same
/// guess `git init` would have made.
fn resolve_ignorecase(repo_root: &Path) -> bool {
    let Ok(config) = std::fs::read_to_string(repo_root.join(".git/config")) else {
        return cfg!(any(windows, target_os = "macos"));
    };
    for line in config.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("ignorecase") else {
            continue;
        };
        let Some(value) = rest.trim_start().strip_prefix('=') else {
            continue;
        };
        return !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "false" | "0" | "no" | "off"
        );
    }
    cfg!(any(windows, target_os = "macos"))
}

/// The top of the checkout `start` belongs to.
///
/// **Nearest** repository marker, else **outermost** project marker, else the filesystem
/// root. The two directions are deliberate and were each wrong once in the other direction:
///
/// - nearest `.git` matches git, which treats a nested repository as its own and does not
///   apply an outer repo's rules inside it;
/// - outermost project manifest, because a manifest marks a package inside a checkout, and
///   stopping at the nearest one drops the root rules that matter most.
///
/// Terminates: `Path::parent()` yields `None` at the drive root (and at a `\\?\C:\` verbatim
/// prefix), so the ascent is bounded even in a tree carrying no marker at all.
fn workspace_root(start: &Path) -> PathBuf {
    let mut outermost_project: Option<PathBuf> = None;
    let mut last = start.to_path_buf();

    let mut cursor = Some(start);
    while let Some(d) = cursor {
        last = d.to_path_buf();
        if REPO_MARKERS.iter().any(|m| d.join(m).exists()) {
            return d.to_path_buf();
        }
        if PROJECT_MARKERS.iter().any(|m| d.join(m).exists()) {
            outermost_project = Some(d.to_path_buf());
        }
        cursor = d.parent();
    }
    outermost_project.unwrap_or(last)
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

    /// A tree with NO repository marker anywhere still finds its root.
    ///
    /// This is the UE shape: a Perforce checkout with no `.git` and no `.p4config`, whose
    /// root is identifiable only by a project marker (`Default.uprojectdirs`). It is also
    /// the only path that reaches the `outermost_project` fallback — every other test here
    /// short-circuits on `.git`, which left that line untested and its tripwire vacuous.
    #[test]
    fn a_tree_with_only_project_markers_uses_the_outermost() {
        let tmp = tempfile::tempdir().unwrap();
        // ABOVE the checkout: must not reach in. Without this the test cannot detect an
        // unbounded ascent at all — a ceiling that is too high still finds the checkout's
        // own rules on the way past, so widening it looks identical to getting it right.
        write(tmp.path(), ".gitignore", "*.cpp\n");

        let root = tmp.path().join("checkout");
        write(&root, "Default.uprojectdirs", "./\n");
        write(&root, ".gitignore", "*.uasset\n");
        write(&root, "Engine/Plugins/App/package.json", "{}\n");
        write(&root, "Engine/Plugins/App/a.uasset", "x\n");
        write(&root, "Engine/Plugins/App/keep.cpp", "void k(){}\n");

        let scope = root.join("Engine/Plugins/App");
        let vi = VcsIgnore::for_scope(&scope).expect("rules");
        assert!(
            vi.is_ignored(&scope.join("a.uasset"), false),
            "the checkout-root .gitignore was not reached — the ascent stopped at the \
             nested package.json instead of continuing to the outermost project marker"
        );
        assert!(
            !vi.is_ignored(&scope.join("keep.cpp"), false),
            "a .gitignore ABOVE the checkout reached in — the ascent is unbounded"
        );
    }

    /// A `.p4config` below a checkout root is not a boundary.
    ///
    /// P4CONFIG is defined by nearest-file lookup *from a subdirectory*, so a `.p4config`
    /// under a root is the normal Perforce idiom. Treated as a repo marker it stopped a git
    /// repo resolving its own root `.gitignore`: `repo/.gitignore: *.gen.cpp` no longer hid
    /// `repo/sub/src/a.gen.cpp`, which `git check-ignore` reports as ignored. Same mistake
    /// `.p4ignore` made one round earlier, in a subtler costume.
    #[test]
    fn a_p4config_below_the_root_is_not_a_boundary() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write(root, ".git/HEAD", "ref: refs/heads/main\n");
        write(root, ".gitignore", "*.gen.cpp\n");
        write(root, "sub/.p4config", "P4PORT=x\n");
        write(root, "sub/src/a.gen.cpp", "void a(){}\n");
        write(root, "sub/src/b.cpp", "void b(){}\n");

        let scope = root.join("sub");
        let vi = VcsIgnore::for_scope(&scope).expect("rules");
        assert!(
            vi.is_ignored(&scope.join("src/a.gen.cpp"), false),
            "a .p4config below the root decapitated the repo's own .gitignore"
        );
        assert!(!vi.is_ignored(&scope.join("src/b.cpp"), false));
    }

    /// `core.ignorecase` comes from the checkout, not from the platform.
    ///
    /// Hard-coding `cfg!(windows)` over-hid on the dangerous side and no repo could opt out.
    /// The behavioural half is an oracle scenario; this pins the parser, including that an
    /// absent config falls back to git's own platform default rather than to `false`.
    #[test]
    fn ignorecase_is_read_from_the_repository() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        write(
            root,
            ".git/config",
            "[core]\n\trepositoryformatversion = 0\n",
        );
        assert_eq!(
            resolve_ignorecase(root),
            cfg!(any(windows, target_os = "macos")),
            "no ignorecase key: fall back to the platform default git itself would pick"
        );

        write(root, ".git/config", "[core]\n\tignorecase = false\n");
        assert!(!resolve_ignorecase(root));

        write(root, ".git/config", "[core]\n\tignorecase = true\n");
        assert!(resolve_ignorecase(root));

        let bare = tmp.path().join("nogit");
        std::fs::create_dir_all(&bare).unwrap();
        assert_eq!(
            resolve_ignorecase(&bare),
            cfg!(any(windows, target_os = "macos")),
            "no git config at all: same platform default"
        );
    }

    /// A nested repository is its own boundary — the outer repo's rules stop at it.
    ///
    /// This is git's own rule, and the fix for the *previous* boundary bug over-corrected
    /// straight past it: taking the outermost marker meant an outer `*.rs` hid
    /// `inner/src/thing.rs`, which `git check-ignore` reports as not ignored. Nearest repo
    /// marker, outermost project marker — the two go in opposite directions on purpose.
    #[test]
    fn a_nested_repository_is_its_own_boundary() {
        let tmp = tempfile::tempdir().unwrap();
        let outer = tmp.path();
        write(outer, ".git/HEAD", "ref: refs/heads/main\n");
        write(outer, ".gitignore", "*.rs\n");
        write(outer, "outer.rs", "fn o() {}\n");
        let inner = outer.join("inner");
        write(&inner, ".git/HEAD", "ref: refs/heads/main\n");
        write(&inner, "src/thing.rs", "fn t() {}\n");

        let vi = VcsIgnore::for_scope(&inner).expect("the inner repo resolves rules");
        assert!(
            !vi.is_ignored(&inner.join("src/thing.rs"), false),
            "an outer repository's .gitignore reached inside a nested repository; \
             git check-ignore says it does not apply"
        );

        // ...and the outer repo's rule still governs its own files.
        let outer_vi = VcsIgnore::for_scope(outer).expect("rules");
        assert!(outer_vi.is_ignored(&outer.join("outer.rs"), false));
    }

    /// One invalid byte must not discard the whole file.
    ///
    /// `read_to_string` returns `Err` on invalid UTF-8, and the `let ... else { continue }`
    /// treated that as "no ignore file here" — so a single stray byte silently dropped every
    /// rule in it, which is the opposite of the per-line promise three lines below the read.
    /// Not expressible as an oracle scenario: the corpus holds `&str`, and this needs bytes.
    #[test]
    fn a_non_utf8_byte_does_not_discard_the_whole_ignore_file() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write(root, ".git/HEAD", "ref: refs/heads/main\n");
        // latin-1 é in the first pattern, valid rules after it.
        let mut bytes = b"caf\xe9.txt\n".to_vec();
        bytes.extend_from_slice(b"*.log\nbuild/\n");
        std::fs::write(root.join(".gitignore"), bytes).unwrap();
        write(root, "a.log", "x\n");
        write(root, "build/b.txt", "x\n");
        write(root, "keep.rs", "fn k() {}\n");

        let vi = VcsIgnore::for_scope(root).expect("rules exist despite the bad byte");
        assert!(
            vi.is_ignored(&root.join("a.log"), false),
            "a rule after the invalid byte was dropped with the whole file"
        );
        assert!(vi.is_ignored(&root.join("build"), true));
        assert!(!vi.is_ignored(&root.join("keep.rs"), false));
    }

    /// A tree with no ignore files hides nothing.
    ///
    /// This used to assert `for_scope` returned `None`, short-circuited on "no ignore file
    /// in the scope directory or above it". That conclusion does not follow: a `.gitignore`
    /// deeper in the tree is discovered during descent, and skipping the matcher meant the
    /// same files got opposite verdicts depending on where the caller scoped — 5 from the
    /// root, 2 from the subdirectory, and no note in the wrong-answer case. The matcher is
    /// now always built; it is lazy, so an empty tree costs a cached lookup per directory.
    #[test]
    fn a_tree_with_no_ignore_files_hides_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write(root, "src/a.rs", "fn a() {}\n");
        write(root, "src/b.log", "noise\n");

        let seen: Vec<String> = crate::search::base_walk_builder(root)
            .build()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_some_and(|t| t.is_file()))
            .filter_map(|e| e.file_name().to_str().map(str::to_string))
            .collect();
        assert!(seen.contains(&"a.rs".to_string()), "{seen:?}");
        assert!(seen.contains(&"b.log".to_string()), "{seen:?}");
    }

    /// A `.gitignore` that exists only DEEPER than the scope must still apply.
    ///
    /// The early-out this replaces looked only at the scope directory and its ancestors, so
    /// scoping at the root left `sub/.gitignore` entirely unread while scoping at `sub`
    /// honoured it.
    #[test]
    fn a_deeper_ignore_file_applies_when_scoping_above_it() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write(root, ".git/HEAD", "ref: refs/heads/main\n");
        write(root, "keep.rs", "fn k() {}\n");
        write(root, "sub/.gitignore", "*.log\n");
        write(root, "sub/hidden.log", "x\n");
        write(root, "sub/keep.txt", "x\n");

        let names = |scope: &Path| -> Vec<String> {
            crate::search::base_walk_builder(scope)
                .build()
                .filter_map(Result::ok)
                .filter(|e| e.file_type().is_some_and(|t| t.is_file()))
                .filter_map(|e| e.file_name().to_str().map(str::to_string))
                .collect()
        };

        for scope in [root, &root.join("sub")] {
            let seen = names(scope);
            assert!(
                !seen.contains(&"hidden.log".to_string()),
                "scope {}: a deeper .gitignore was not applied: {seen:?}",
                scope.display()
            );
        }
        assert!(names(root).contains(&"keep.rs".to_string()));
    }

    /// Scoping directly at a directory an ancestor rule excludes must still search it.
    ///
    /// `Saved`, `Intermediate` and `Binaries` are exactly the directories someone scopes
    /// into deliberately, and "0 files" for a directory the caller named is the worst
    /// answer available. Before the fix the walk-root exemption was applied to the root
    /// *entry*, which is never a result — every child was still evaluated against a parent
    /// whose `excluded` flag was already set, so the whole subtree vanished.
    #[test]
    fn scoping_into_an_ignored_directory_still_searches_it() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write(root, ".git/HEAD", "ref: refs/heads/main\n");
        write(root, ".gitignore", "Saved/\n");
        write(root, "Saved/a.rs", "fn a() {}\n");
        write(root, "Saved/deep/b.rs", "fn b() {}\n");
        write(root, "keep.rs", "fn k() {}\n");

        let seen: Vec<String> = crate::search::base_walk_builder(&root.join("Saved"))
            .build()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_some_and(|t| t.is_file()))
            .filter_map(|e| e.file_name().to_str().map(str::to_string))
            .collect();
        assert!(seen.contains(&"a.rs".to_string()), "{seen:?}");
        assert!(
            seen.contains(&"b.rs".to_string()),
            "nested content under an explicitly scoped ignored directory: {seen:?}"
        );

        // And the rule still applies when the caller did NOT name it.
        let from_root: Vec<String> = crate::search::base_walk_builder(root)
            .build()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_some_and(|t| t.is_file()))
            .filter_map(|e| e.file_name().to_str().map(str::to_string))
            .collect();
        assert!(!from_root.contains(&"a.rs".to_string()), "{from_root:?}");
        assert!(from_root.contains(&"keep.rs".to_string()), "{from_root:?}");
    }

    /// The outermost VCS root wins, not the nearest marker.
    ///
    /// A nested package manifest is a project inside a checkout, not a new checkout.
    /// Stopping at it dropped the root rules: measured on a real UE tree, scoping at a
    /// subdirectory containing `package.json` returned 279 files with no note against 229
    /// for the same subtree reached from its parent.
    #[test]
    fn a_nested_project_manifest_does_not_decapitate_the_root_rules() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write(root, ".git/HEAD", "ref: refs/heads/main\n");
        write(root, "package.json", "{}\n");
        write(root, ".gitignore", "*.gen.ts\n");
        write(root, "packages/app/package.json", "{}\n");
        write(root, "packages/app/.gitignore", "*.local.ts\n");
        write(root, "packages/app/src/thing.gen.ts", "x\n");
        write(root, "packages/app/src/z.local.ts", "x\n");
        write(root, "packages/app/src/ok.ts", "x\n");

        let seen: Vec<String> = crate::search::base_walk_builder(&root.join("packages/app"))
            .build()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_some_and(|t| t.is_file()))
            .filter_map(|e| e.file_name().to_str().map(str::to_string))
            .collect();
        assert!(
            !seen.contains(&"thing.gen.ts".to_string()),
            "the root .gitignore was never read — the ascent stopped at the nested \
             package.json: {seen:?}"
        );
        assert!(!seen.contains(&"z.local.ts".to_string()), "{seen:?}");
        assert!(seen.contains(&"ok.ts".to_string()), "{seen:?}");
    }
}
