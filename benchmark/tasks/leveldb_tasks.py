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

Ground truth is taken from what the pinned commit actually contains, checked
against tilth's own output so a passing answer is reachable in a couple of
calls rather than only by exhaustive grep.
"""

from tasks.base import Task, GroundTruth


class LevelDBStatusTypeTask(Task):
    """Easy tier: single type, one header plus its out-of-line implementation.

    `class LEVELDB_EXPORT Status` is the pattern that breaks naive name
    extraction — the visibility macro sits between `class` and the type name, so
    a parser that takes the first identifier after `class` reports the type as
    `LEVELDB_EXPORT`. The private section (`state_`, `CopyState`) is only
    reachable if the declaration's full extent was resolved, and `CopyState` is
    defined out-of-line in `util/status.cc`, which requires following the
    qualified definition out of the header.
    """

    @property
    def name(self) -> str:
        return "leveldb_status_type"

    @property
    def repo(self) -> str:
        return "leveldb"

    @property
    def prompt(self) -> str:
        return (
            "Describe leveldb's `Status` class: which header declares it, what "
            "macro appears in its declaration between `class` and the type "
            "name, what its single data member is and how that member encodes "
            "the error code and message, and which of its member functions are "
            "implemented outside the header (name the file)."
        )

    @property
    def ground_truth(self) -> GroundTruth:
        # LEVELDB_EXPORT: the macro in the class head, the thing a broken C++
        # class-head parse would report as the type name.
        # state_ / CopyState: the private section, only visible if the class
        # declaration's extent resolved correctly.
        # status.cc: util/status.cc holds the out-of-line Status::CopyState,
        # Status::ToString and the private Status::Status(Code, ...) ctor.
        return GroundTruth(
            required_strings=[
                "LEVELDB_EXPORT",
                "state_",
                "CopyState",
                "status.cc",
            ],
        )


class LevelDBCorruptionCallersTask(Task):
    """Hard tier: call sites of a qualified static member function.

    `Status::Corruption` is a static factory declared inline in
    include/leveldb/status.h and called qualified from ~30 sites across db/ and
    table/. The required strings are the *enclosing* functions of the sites in
    three named files — including two that are methods rather than free
    functions (`log::Reader::ReportCorruption`, `Repairer::ConvertLogToTable`).
    Naming them requires resolving the scope around a call site, not just
    matching the call text.

    The three files are named in the prompt on purpose. An earlier phrasing
    asked for "at least four call sites" and graded against three specific
    ones; a run that listed eight correct sites failed because none of them
    happened to be in db/repair.cc. The scope resolution is what's under test,
    so the sites are pinned and the answer is deterministic.
    """

    @property
    def name(self) -> str:
        return "leveldb_corruption_callers"

    @property
    def repo(self) -> str:
        return "leveldb"

    @property
    def prompt(self) -> str:
        return (
            "leveldb signals a corrupt database by building a `Status` through "
            "the static factory `Status::Corruption`. Find where that factory "
            "is declared and which `Code` enum value it stores. Then find its "
            "call sites in these three files — db/db_impl.cc, db/log_reader.cc "
            "and db/repair.cc — and for every one of those sites name the "
            "enclosing function or method that contains the call."
        )

    @property
    def ground_truth(self) -> GroundTruth:
        # kCorruption: the enum value the factory encodes — requires reading the
        # private Code enum in status.h, not just the factory signature.
        # The other three are enclosing scopes of real call sites at the pinned
        # commit: DBImpl::RecoverLogFile (db/db_impl.cc:434-435),
        # log::Reader::ReportCorruption (db/log_reader.cc:179) and
        # Repairer::ConvertLogToTable (db/repair.cc:185-186). All three are
        # unguessable without actually locating the call sites.
        return GroundTruth(
            required_strings=[
                "kCorruption",
                "RecoverLogFile",
                "ReportCorruption",
                "ConvertLogToTable",
            ],
        )

    @property
    def task_type(self) -> str:
        return "navigate"


class LevelDBEnvHeaderDepsTask(Task):
    """Medium tier: both directions of a public header's include graph.

    include/leveldb/env.h is the widest public header in the tree: it pulls in
    leveldb/export.h and leveldb/status.h, and eight files depend on it. The
    interesting dependent is db/version_set.cc, which is not obvious from the
    header itself and which uses both `Log` and `ReadFileToString`.
    """

    @property
    def name(self) -> str:
        return "leveldb_env_header_deps"

    @property
    def repo(self) -> str:
        return "leveldb"

    @property
    def prompt(self) -> str:
        return (
            "Analyse the dependencies of the header include/leveldb/env.h in "
            "both directions: which headers does it include, and which source "
            "files in this repository depend on it? For the non-test "
            "dependents, name the free functions declared in env.h that they "
            "actually call."
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
