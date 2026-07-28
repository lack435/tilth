"""C++ navigation tasks against leveldb (pinned at the 1.23 tag).

C++ had no benchmark coverage, so regressions in the shared tree-sitter
node-kind tables (`src/lang/treesitter.rs`, `src/lang/outline.rs`) could only be
caught by unit tests. These three tasks cover the parts of C++ handling with no
analogue in the other four benchmark repos:

  * `leveldb_status_type` — resolving a class declared in a header whose head
    carries an export/visibility macro (`class LEVELDB_EXPORT Status`), plus the
    member functions defined out-of-line in a matching `.cc`.
  * `leveldb_corruption_callers` — call sites of a static member function
    invoked qualified (`Status::Corruption(...)`), with the enclosing function
    named for each site.
  * `leveldb_env_header_deps` — the include graph of a public header, in both
    directions.

Ground truth is taken from what the pinned commit actually contains — verified
against the source, not against tilth's output. That distinction matters: an
earlier draft calibrated the strings to what tilth prints, which both graded
baseline runs against tilth's blind spots and froze those blind spots in as
expected behaviour.

What these tasks can and cannot catch: grading is a substring match over the
agent's text, and every mode has `Read` available, so an agent can always
recover from a wrong tilth answer by reading the file. That makes these
cost-and-turn regression guards for the C++ tool paths, plus a tripwire for
gross breakage — not correctness guards for the node-kind tables. Table
correctness still belongs to the unit tests in `src/lang/`. Each task below
therefore requires at least one string that only exists at a site the agent has
to navigate to, so a plausible-sounding answer assembled from general knowledge
of leveldb fails.
"""

from tasks.base import Task, GroundTruth


class LevelDBStatusTypeTask(Task):
    """Easy tier: single type, one header plus its out-of-line implementation.

    `class LEVELDB_EXPORT Status` is the pattern that breaks naive name
    extraction — the visibility macro sits between `class` and the type name, so
    a parser that takes the first identifier after `class` reports the type as
    `LEVELDB_EXPORT`.

    Be honest about the grading floor here: `LEVELDB_EXPORT`, `state_` and
    `CopyState` are all literal text in include/leveldb/status.h, so a single
    `Read` of that header satisfies three of the five strings, and `status.h` ->
    `status.cc` is a convention guess. `Not implemented` is the string that
    forces the agent out of the header — `Status::ToString` maps `kNotSupported`
    to the prefix `"Not implemented: "` (util/status.cc:55), which does not match
    the enum name and appears nowhere else in the tree that answers this
    question. An answer built from general knowledge of leveldb gets it wrong.
    """

    @property
    def name(self) -> str:
        return "leveldb_status_type"

    @property
    def repo(self) -> str:
        return "leveldb"

    @property
    def requires_tool_use(self) -> list[str]:
        # Type resolution is the path under test, reachable through either a
        # structural symbol search or grok. Reading the header with tilth_read alone
        # does not exercise it — that is just `cat`.
        return ["mcp__tilth__tilth_grok|mcp__tilth__tilth_search"]

    @property
    def prompt(self) -> str:
        return (
            "Describe leveldb's `Status` class: which header declares it, what "
            "macro appears in its declaration between `class` and the type "
            "name, what its single data member is and how that member encodes "
            "the error code and message, and which of its member functions are "
            "implemented outside the header (name the file). From that "
            "implementation file, also give the exact human-readable prefix "
            "that `ToString()` produces for the `kNotSupported` code."
        )

    @property
    def ground_truth(self) -> GroundTruth:
        # LEVELDB_EXPORT: the macro in the class head, the thing a broken C++
        # class-head parse would report as the type name.
        # state_ / CopyState: the private section of the declaration.
        # status.cc: util/status.cc holds the out-of-line Status::CopyState,
        # Status::ToString and the private Status::Status(Code, ...) ctor.
        # "Not implemented": util/status.cc:55. The one string that cannot be
        # produced from the header or from general leveldb knowledge — see the
        # class docstring.
        return GroundTruth(
            required_strings=[
                "LEVELDB_EXPORT",
                "state_",
                "CopyState",
                "status.cc",
                "Not implemented",
            ],
        )


class LevelDBCorruptionCallersTask(Task):
    """Hard tier: call sites of a qualified static member function.

    `Status::Corruption` is a static factory declared inline in
    include/leveldb/status.h and called qualified from 30 sites across db/ and
    table/. The three graded files hold exactly four of those sites, and all
    four enclosing scopes are class members, not free functions:
    `DBImpl::Recover` (db_impl.cc:360), `DBImpl::RecoverLogFile` (:435),
    `log::Reader::ReportCorruption` (log_reader.cc:179) and
    `Repairer::ConvertLogToTable` (repair.cc:186).

    The three files are named in the prompt on purpose. An earlier phrasing
    asked for "at least four call sites" and graded against three specific
    ones; a run that listed eight correct sites failed because none of them
    happened to be in db/repair.cc.

    Pinning the files keeps the graded detail stable, but on its own it removed
    the search entirely: with the files named, the task is answerable by reading
    three files, and a `tilth_forced` run confirmed the agent did exactly that —
    eight plain symbol searches and nine reads, never a `kind="callers"` search.
    A task that never enters the code path it guards is not a guard.

    So the prompt asks for a repo-wide survey of the call sites before narrowing
    to the three graded files. That phrasing is load-bearing and was measured:
    with it, every run that got past the first rep used `kind="callers"`, where
    the pinned-files-only phrasing never did. The survey is not itself graded —
    see `ground_truth` — it exists to steer the route. `requires_tool_use` is
    what actually enforces that the route was taken.

    The enclosing-function names alone are a weak signal — `ReportCorruption` is
    also declared at db/log_reader.h:85 and every name appears in a plain outline
    of the file the prompt names, so an agent could outline and guess. The two
    message literals close that hole: they exist only at the call sites
    themselves, and `missing files` covers the db/db_impl.cc:360 site in
    `DBImpl::Recover` that the enclosing-name strings otherwise miss. `Recover`
    cannot be required directly — under substring grading it is satisfied by
    `RecoverLogFile`.

    The prompt also warns that a same-named `Corruption` method exists on the
    log reporter. tilth's callers search cannot filter by qualification, so
    `--callers Corruption` returns 33 sites: the 30 `Status::Corruption` calls
    plus 3 `Reporter::Corruption` calls. Without the warning, an agent that
    pastes that list reports `log::Reader::ReportDrop` as a `Status::Corruption`
    caller — wrong — and still passes.
    """

    @property
    def name(self) -> str:
        return "leveldb_corruption_callers"

    @property
    def requires_tool_use(self) -> list[str]:
        # The whole point of this task. Caller detection for C++ qualified statics is
        # the path being guarded, and it runs for a `kind="callers"` search or for
        # grok (which assembles callers itself). Without this requirement the task
        # passed on Glob + Bash + Read with zero tilth calls — a green result that
        # said nothing about the code it was written to protect.
        return [
            "mcp__tilth__tilth_search:kind=callers|mcp__tilth__tilth_grok",
        ]

    @property
    def repo(self) -> str:
        return "leveldb"

    @property
    def prompt(self) -> str:
        return (
            "leveldb signals a corrupt database by building a `Status` through "
            "the static factory `Status::Corruption`. Find where that factory "
            "is declared and which `Code` enum value it stores. Then survey its "
            "call sites across the whole repository to see which files contain "
            "them. From that survey, report on the call sites in db/db_impl.cc, "
            "db/log_reader.cc and db/repair.cc specifically: for each, give the "
            "enclosing function or method that contains the call and the message "
            "string passed to the factory. Note that an unrelated `Corruption` "
            "method exists on the log reporter; count only calls to "
            "`Status::Corruption` itself."
        )

    @property
    def ground_truth(self) -> GroundTruth:
        # kCorruption: the enum value the factory encodes.
        # Enclosing scopes of real call sites at the pinned commit:
        # DBImpl::RecoverLogFile (db/db_impl.cc:434-435, one call over two
        # lines), log::Reader::ReportCorruption (db/log_reader.cc:179) and
        # Repairer::ConvertLogToTable (db/repair.cc:186).
        # Message literals, which exist only at the call sites:
        # "log record too small" (db_impl.cc:435 and repair.cc:186) and
        # "missing files" (db_impl.cc:358, snprintf'd into the buf passed at
        # :360 inside DBImpl::Recover — the site the scope names miss).
        # Deliberately no repo-wide count here, though the prompt asks for the survey.
        # A count is not gradeable through tilth: the callers view caps at 10 matches,
        # so the 30 `Status::Corruption` calls across 12 files cannot be tallied from
        # its output. Measured — both `tilth_forced` runs used `kind="callers"` as
        # intended and still answered "6 files", while hybrid runs answered 12 because
        # they had grep. Grading a count therefore penalises using tilth, which is
        # backwards. The survey stays in the prompt because it demonstrably steers the
        # agent onto the callers path (see the class docstring); the *graded* facts are
        # the per-file details, which are stable at the pinned commit.
        return GroundTruth(
            required_strings=[
                "kCorruption",
                "RecoverLogFile",
                "ReportCorruption",
                "ConvertLogToTable",
                "log record too small",
                "missing files",
            ],
        )

    @property
    def task_type(self) -> str:
        return "navigate"


class LevelDBEnvHeaderDepsTask(Task):
    """Medium tier: both directions of a public header's include graph.

    include/leveldb/env.h is the widest public header in the tree: it pulls in
    leveldb/export.h and leveldb/status.h, and 39 tracked files include it
    directly (vs 23 for db.h, 20 for slice.h).

    Note that tilth's `--deps` reports only 8 dependents for it, because its
    "Used by" section lists a dependent only when it resolved a *symbol* usage —
    so every file that uses just the `Env` / `WritableFile` types is dropped,
    including util/env.cc, which defines `ReadFileToString`. The ground truth is
    deliberately not that set of 8: grading against tilth's own answer would
    fail baseline runs for summarising a different (correct) subset of 39, and
    would freeze the type-only-dependent gap in as expected behaviour. Instead
    the prompt pins db/version_set.cc, so both modes are asked the same
    determinate question — it is a non-obvious dependent whose `Recover` calls
    both `Log` and `ReadFileToString` at db/version_set.cc:872.
    """

    @property
    def name(self) -> str:
        return "leveldb_env_header_deps"

    @property
    def requires_tool_use(self) -> list[str]:
        # Both directions of the include graph come from tilth_deps; nothing else
        # computes dependents. A grep can list `#include` lines but not what includes
        # this header, so an answer without tilth_deps did not test the path.
        return ["mcp__tilth__tilth_deps"]

    @property
    def repo(self) -> str:
        return "leveldb"

    @property
    def prompt(self) -> str:
        return (
            "Analyse the dependencies of the header include/leveldb/env.h in "
            "both directions: which headers does it include, and which source "
            "files in this repository depend on it? For db/version_set.cc in "
            "particular, name the free functions declared in env.h that it "
            "actually calls."
        )

    @property
    def ground_truth(self) -> GroundTruth:
        # export.h / status.h: the two project headers env.h includes.
        # version_set.cc: a non-obvious dependent.
        # ReadFileToString: a free helper declared in env.h and called from
        # db/version_set.cc's Recover — requires the dependent's symbol usage,
        # not just the file list.
        return GroundTruth(
            required_strings=[
                "export.h",
                "status.h",
                "version_set.cc",
                "ReadFileToString",
            ],
        )

    @property
    def task_type(self) -> str:
        return "navigate"
