//! Differential oracle: every verdict [`VcsIgnore`] reaches is compared against git's own.
//!
//! Owning the matching (see the module header) means owning the risk of diverging from git.
//! A unit test can only pin the cases its author thought of, and the cases that hurt are the
//! ones nobody thought of — the reverted attempt passed its own tests while disagreeing with
//! git about nested negation.
//!
//! So the authority here is `git check-ignore --no-index --stdin`, run against a real
//! repository built per scenario. `--no-index` because it decides on patterns alone, which
//! is the same question this matcher answers; without it git consults the index and would be
//! answering a different one.
//!
//! Skipped, not failed, when `git` is absent — CI has it, and a developer without it should
//! not see a red suite for a tool they do not have.

use std::path::Path;
use std::process::{Command, Stdio};

use super::VcsIgnore;

/// One tree: the ignore files to write, and the paths to ask about.
struct Scenario {
    name: &'static str,
    /// `git config` keys applied to the fixture repo before either side is asked.
    ///
    /// Exists for `core.ignorecase`, which is a property of the checkout rather than the
    /// patterns: hard-coding it to the platform over-hid on the dangerous side, and the
    /// oracle could not see that while every fixture used git's default.
    git_config: &'static [(&'static str, &'static str)],
    /// (relative path, contents) for each ignore file.
    ignore_files: &'static [(&'static str, &'static str)],
    /// Relative paths, forward-slash. A trailing `/` means "ask about a directory".
    paths: &'static [&'static str],
}

const SCENARIOS: &[Scenario] = &[
    Scenario {
        git_config: &[],
        name: "plain names and extensions",
        ignore_files: &[(".gitignore", "*.log\nbuild/\nexact.txt\n")],
        paths: &[
            "a.log",
            "src/b.log",
            "exact.txt",
            "src/exact.txt",
            "keep.rs",
            "build/",
            "build/x.rs",
            "src/build/",
            "src/build/y.rs",
        ],
    },
    Scenario {
        git_config: &[],
        name: "anchored vs unanchored",
        ignore_files: &[(".gitignore", "/root-only.txt\nanywhere.txt\n/dir/\n")],
        paths: &[
            "root-only.txt",
            "sub/root-only.txt",
            "anywhere.txt",
            "sub/anywhere.txt",
            "dir/",
            "dir/f.rs",
            "sub/dir/",
            "sub/dir/f.rs",
        ],
    },
    Scenario {
        git_config: &[],
        name: "double star",
        ignore_files: &[(".gitignore", "**/gen/**\na/**/b.txt\n**/deep.log\n")],
        paths: &[
            "gen/x.rs",
            "a/gen/x.rs",
            "a/b/gen/x.rs",
            "a/b.txt",
            "a/x/b.txt",
            "a/x/y/b.txt",
            "deep.log",
            "p/deep.log",
            "p/q/deep.log",
        ],
    },
    Scenario {
        git_config: &[],
        name: "negation in one file",
        ignore_files: &[(".gitignore", "*.json\n!keep.json\n*.tmp\n!/root.tmp\n")],
        paths: &[
            "a.json",
            "keep.json",
            "sub/keep.json",
            "sub/a.json",
            "root.tmp",
            "sub/root.tmp",
            "x.tmp",
        ],
    },
    Scenario {
        git_config: &[],
        name: "nested file overrides shallower",
        ignore_files: &[
            (".gitignore", "*.json\n"),
            ("config/.gitignore", "!settings.json\n"),
        ],
        paths: &[
            "config/settings.json",
            "config/other.json",
            "src/settings.json",
            "top.json",
        ],
    },
    Scenario {
        git_config: &[],
        name: "nested file tightens rather than loosens",
        ignore_files: &[
            (".gitignore", "!*.keep\n"),
            ("vendor/.gitignore", "*.keep\n"),
        ],
        paths: &[
            "a.keep",
            "vendor/a.keep",
            "vendor/deep/a.keep",
            "other/a.keep",
        ],
    },
    Scenario {
        git_config: &[],
        name: "three levels of alternating opinion",
        ignore_files: &[
            (".gitignore", "*.x\n"),
            ("a/.gitignore", "!*.x\n"),
            ("a/b/.gitignore", "*.x\n"),
        ],
        paths: &["t.x", "a/t.x", "a/b/t.x", "a/b/c/t.x", "a/c/t.x"],
    },
    Scenario {
        git_config: &[],
        name: "UE whitelist idiom",
        ignore_files: &[(
            ".gitignore",
            "*\n!*/\n!*.c\n!*.cpp\n!*.h\n!*.py\n!.gitignore\n",
        )],
        paths: &[
            "Engine/",
            "Engine/Source/",
            "Engine/Source/a.cpp",
            "Engine/Source/a.h",
            "Engine/Source/a.py",
            "Engine/Source/notes.txt",
            "Engine/Source/a.yaml",
            "top.cpp",
            "top.txt",
        ],
    },
    Scenario {
        git_config: &[],
        name: "directory-only rules at depth",
        ignore_files: &[(".gitignore", "Intermediate/\nSaved/\nBinaries/\n")],
        paths: &[
            "Intermediate/",
            "Intermediate/a.cpp",
            "Engine/Intermediate/",
            "Engine/Intermediate/a.cpp",
            "Engine/Plugins/X/Intermediate/a.cpp",
            "Engine/Saved/log.txt",
            "Engine/Source/a.cpp",
        ],
    },
    Scenario {
        git_config: &[],
        name: "character classes and single-char wildcard",
        ignore_files: &[(".gitignore", "f?le.txt\n[abc]start.rs\n*.o[12]\n")],
        paths: &[
            "file.txt",
            "fle.txt",
            "astart.rs",
            "dstart.rs",
            "x.o1",
            "x.o2",
            "x.o3",
        ],
    },
    // The four cases below were all found by review, all diverged from git, and none was
    // caught by the eleven scenarios above — they are here so the oracle owns them rather
    // than a hand-written assertion about what git "should" do.
    Scenario {
        git_config: &[],
        name: "leading whitespace is significant to git",
        // `trim()` on both edges made `   leading.txt` a live rule and hid a file git
        // tracks — the dangerous direction. Git strips only trailing whitespace.
        ignore_files: &[(".gitignore", "   leading.txt\nkeep.rs\n")],
        paths: &["leading.txt", "keep.rs", "other.txt"],
    },
    Scenario {
        git_config: &[],
        name: "UTF-8 BOM does not eat the first pattern",
        // `str::trim` does not strip U+FEFF, so the first pattern became `\u{FEFF}*.log`
        // and matched nothing. PowerShell writes UTF-8 with BOM by default on this platform.
        ignore_files: &[(".gitignore", "\u{FEFF}*.log\nbuild/\n")],
        paths: &["a.log", "sub/b.log", "build/x.txt", "keep.rs"],
    },
    Scenario {
        git_config: &[],
        name: "case-mismatched patterns",
        // NTFS is case-insensitive and `git init` sets core.ignorecase=true, so git ignores
        // `a.log` under a `*.LOG` rule. Every other scenario uses case-matching patterns, so
        // this divergence was invisible while 72/72 held.
        ignore_files: &[(".gitignore", "*.LOG\nBuild/\n")],
        paths: &["a.log", "A.LOG", "build/x.txt", "Build/y.txt", "keep.rs"],
    },
    Scenario {
        // `core.ignorecase=false` on a platform whose default is true. tilth used to hard-code
        // the platform, so it hid `a.log` under `*.LOG` while git reports it as NOT ignored —
        // over-hiding, with no way for a repo to opt out.
        git_config: &[("core.ignorecase", "false")],
        name: "case-sensitive repo on a case-insensitive platform",
        ignore_files: &[(
            ".gitignore",
            "*.LOG
Build/
",
        )],
        paths: &["a.log", "A.LOG", "build/x.txt", "Build/y.txt", "keep.rs"],
    },
    Scenario {
        git_config: &[],
        name: "trailing spaces and escapes",
        ignore_files: &[(
            ".gitignore",
            "with\\ space.txt\ntrailing   \n#notacomment\n",
        )],
        paths: &["with space.txt", "trailing", "#notacomment"],
    },
];

fn git_available() -> bool {
    Command::new("git")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn git(root: &Path, args: &[&str]) {
    let ok = Command::new("git")
        .args(args)
        .current_dir(root)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    assert!(ok, "git {args:?} failed in {}", root.display());
}

/// git's verdict for every path, in one invocation.
///
/// `check-ignore` prints only the paths it ignores, so absence is the "not ignored" answer —
/// which also means a broken invocation looks exactly like "nothing is ignored". The caller
/// canaries that below rather than trusting silence.
fn git_ignored(root: &Path, paths: &[&str]) -> Vec<bool> {
    use std::io::Write;
    let mut child = Command::new("git")
        .args(["check-ignore", "--no-index", "--stdin"])
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn git check-ignore");
    {
        let stdin = child.stdin.as_mut().expect("stdin");
        for p in paths {
            writeln!(stdin, "{}", p.trim_end_matches('/')).expect("write path");
        }
    }
    let out = child.wait_with_output().expect("wait");
    let listed: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|l| l.trim().replace('\\', "/"))
        .collect();
    paths
        .iter()
        .map(|p| listed.iter().any(|l| l == p.trim_end_matches('/')))
        .collect()
}

#[test]
fn every_verdict_agrees_with_git_check_ignore() {
    if !git_available() {
        eprintln!("skipping vcsignore oracle: no git on PATH");
        return;
    }

    let mut checks = 0usize;
    let mut mismatches: Vec<String> = Vec::new();

    for sc in SCENARIOS {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        git(root, &["init", "-q", "."]);
        git(root, &["config", "user.email", "t@t"]);
        git(root, &["config", "user.name", "t"]);
        for (k, v) in sc.git_config {
            git(root, &["config", k, v]);
        }

        for (rel, body) in sc.ignore_files {
            let p = root.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, body).unwrap();
        }
        for rel in sc.paths {
            let p = root.join(rel.trim_end_matches('/'));
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            if rel.ends_with('/') {
                std::fs::create_dir_all(&p).unwrap();
            } else {
                std::fs::write(&p, "x\n").unwrap();
            }
        }

        // Canary: this scenario must produce at least one ignored path, or a silently broken
        // invocation would read as "everything agrees" and the scenario would prove nothing.
        let git_says = git_ignored(root, sc.paths);
        assert!(
            git_says.iter().any(|b| *b),
            "scenario {:?} produced no ignored paths from git — the oracle proves nothing here",
            sc.name
        );

        let vi = VcsIgnore::for_scope(root).unwrap_or_else(|| {
            panic!(
                "scenario {:?} wrote ignore files but none were found",
                sc.name
            )
        });

        for (rel, &want) in sc.paths.iter().zip(git_says.iter()) {
            let is_dir = rel.ends_with('/');
            let got = vi.is_ignored(&root.join(rel.trim_end_matches('/')), is_dir);
            checks += 1;
            if got != want {
                mismatches.push(format!(
                    "  [{}] {rel}: git={} tilth={}",
                    sc.name,
                    if want { "IGNORED" } else { "kept" },
                    if got { "IGNORED" } else { "kept" },
                ));
            }
        }
    }

    // Guards against the corpus silently shrinking (a scenario deleted, a `paths` list
    // emptied) — a green oracle over nothing is the failure mode this whole file exists to
    // avoid. The number is what the corpus currently produces, not a target.
    assert!(checks >= 89, "only {checks} checks ran; the corpus shrank");
    assert!(
        mismatches.is_empty(),
        "{} of {checks} verdicts disagree with git check-ignore:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}
