"""Tests for the benchmark harness's own logic.

Run with:  python -m unittest discover -s benchmark -p 'test_*.py'

Stdlib `unittest` on purpose — the harness has no test dependency and should not
gain one. These cover `_unmet_tool_requirements`, whose failure mode is silent: a
malformed or over-broad requirement does not error, it just mis-grades a task
forever. An earlier revision of this matcher had eight such cases.
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from parse import ToolCall, Turn, RunResult, parse_codex_json  # noqa: E402
from run import TASKS, _unmet_tool_requirements  # noqa: E402


def _run(*calls_per_turn: list[ToolCall]) -> RunResult:
    """Build a RunResult with one Turn per argument."""
    turns = [
        Turn(
            index=i,
            input_tokens=0,
            output_tokens=0,
            cache_creation_tokens=0,
            cache_read_tokens=0,
            tool_calls=list(calls),
        )
        for i, calls in enumerate(calls_per_turn)
    ]
    return RunResult(
        session_id="test",
        turns=turns,
        num_turns=len(turns),
        total_cost_usd=0.0,
        duration_ms=0,
        duration_api_ms=0,
        total_input_tokens=0,
        total_output_tokens=0,
        total_cache_creation_tokens=0,
        total_cache_read_tokens=0,
        result_text="",
    )


def _call(name: str, **args) -> ToolCall:
    return ToolCall(name=name, input=dict(args), tool_use_id="x", turn_index=0)


SEARCH = "mcp__tilth__tilth_search"
GROK = "mcp__tilth__tilth_grok"
DEPS = "mcp__tilth__tilth_deps"
CALLERS_REQ = [f"{SEARCH}:kind=callers|{GROK}"]


class TestToolRequirementMatching(unittest.TestCase):
    def test_callers_search_satisfies(self):
        run = _run([_call(SEARCH, query="Corruption", kind="callers")])
        self.assertEqual(_unmet_tool_requirements(run, CALLERS_REQ), [])

    def test_plain_search_does_not_satisfy_callers(self):
        # The bug this whole mechanism exists for: a symbol search is the same tool as
        # a callers search but does not run caller detection.
        run = _run([_call(SEARCH, query="Corruption")])
        self.assertEqual(_unmet_tool_requirements(run, CALLERS_REQ), CALLERS_REQ)

    def test_grok_satisfies_via_alternation(self):
        run = _run([_call(GROK, target="Corruption")])
        self.assertEqual(_unmet_tool_requirements(run, CALLERS_REQ), [])

    def test_builtin_tools_only_does_not_satisfy(self):
        run = _run([_call("Bash", command="grep -rn x"), _call("Read", file_path="a")])
        self.assertEqual(_unmet_tool_requirements(run, CALLERS_REQ), CALLERS_REQ)

    def test_matches_across_later_turns(self):
        run = _run([_call("Bash")], [_call("Read")], [_call(SEARCH, kind="callers")])
        self.assertEqual(_unmet_tool_requirements(run, CALLERS_REQ), [])

    def test_empty_run_is_unmet(self):
        self.assertEqual(_unmet_tool_requirements(_run(), CALLERS_REQ), CALLERS_REQ)
        self.assertEqual(_unmet_tool_requirements(_run([]), CALLERS_REQ), CALLERS_REQ)

    def test_boolean_argument_uses_json_spelling(self):
        # Requirements are written as an author would type them (`full=true`), while the
        # parsed input holds Python `True`.
        run = _run([_call(GROK, target="X", full=True)])
        self.assertEqual(_unmet_tool_requirements(run, [f"{GROK}:full=true"]), [])

    def test_argument_match_is_case_insensitive(self):
        run = _run([_call(SEARCH, kind="Callers")])
        self.assertEqual(_unmet_tool_requirements(run, [f"{SEARCH}:kind=callers"]), [])

    def test_numeric_argument(self):
        run = _run([_call(SEARCH, query="X", expand=5)])
        self.assertEqual(_unmet_tool_requirements(run, [f"{SEARCH}:expand=5"]), [])

    def test_whitespace_is_tolerated(self):
        run = _run([_call(DEPS, path="a.h")])
        self.assertEqual(_unmet_tool_requirements(run, [f"  {DEPS}  "]), [])
        run2 = _run([_call(GROK, target="X")])
        self.assertEqual(
            _unmet_tool_requirements(run2, [f"{SEARCH}:kind=callers | {GROK}"]), []
        )

    def test_negated_argument_excludes_content_search(self):
        # A structural search is any tilth_search that is not a content search.
        req = [f"{SEARCH}:kind!=content"]
        content = _run([_call(SEARCH, query="X", kind="content")])
        self.assertEqual(_unmet_tool_requirements(content, req), req)
        symbol = _run([_call(SEARCH, query="X", kind="symbol")])
        self.assertEqual(_unmet_tool_requirements(symbol, req), [])

    def test_negated_argument_accepts_an_omitted_argument(self):
        # `kind` defaults to symbol and is usually omitted, so an absent argument must
        # satisfy `kind!=content` — otherwise the requirement rejects the common case.
        run = _run([_call(SEARCH, query="X")])
        self.assertEqual(_unmet_tool_requirements(run, [f"{SEARCH}:kind!=content"]), [])

    def test_absent_argument_does_not_satisfy_a_positive_match(self):
        run = _run([_call(SEARCH, query="X")])
        req = [f"{SEARCH}:kind=callers"]
        self.assertEqual(_unmet_tool_requirements(run, req), req)

    def test_malformed_argument_spec_raises(self):
        # `"tool:kind"` with no `=` previously matched exactly the calls it was meant to
        # exclude, mis-grading the task silently and permanently.
        run = _run([_call(SEARCH, kind="callers")])
        with self.assertRaises(ValueError):
            _unmet_tool_requirements(run, [f"{SEARCH}:kind"])
        with self.assertRaises(ValueError):
            _unmet_tool_requirements(run, [":kind=callers"])


class TestCodexToolNaming(unittest.TestCase):
    """codex and Claude Code must identify the same tool by the same name.

    codex reports `server` and `tool` separately with a bare tool name; Claude Code
    reports one flattened `mcp__server__tool`. Requirements are written in the
    flattened form, so without normalisation every task declaring one would fail on
    `--models gpt5`/`o3` regardless of behaviour — and per-runner `tool_calls`
    breakdowns would key the same tool differently.
    """

    def _codex_output(self, arguments) -> str:
        events = [
            {"type": "thread.started", "thread_id": "t1"},
            {"type": "turn.started"},
            {
                "type": "item.completed",
                "item": {
                    "type": "mcp_tool_call",
                    "id": "i1",
                    "server": "tilth",
                    "tool": "tilth_search",
                    "arguments": arguments,
                },
            },
            {"type": "turn.completed", "usage": {}},
        ]
        return "\n".join(json.dumps(e) for e in events)

    def test_bare_codex_tool_name_is_flattened(self):
        run = parse_codex_json(self._codex_output({"kind": "callers"}), "gpt-5-codex")
        names = [tc.name for turn in run.turns for tc in turn.tool_calls]
        self.assertEqual(names, [SEARCH])
        # ...and therefore satisfies a requirement written the Claude Code way.
        self.assertEqual(_unmet_tool_requirements(run, CALLERS_REQ), [])

    def test_string_arguments_payload_does_not_crash(self):
        # Guard: a non-object `arguments` degrades to "no arguments" instead of raising
        # AttributeError deep inside requirement matching.
        run = parse_codex_json(self._codex_output("not-an-object"), "gpt-5-codex")
        self.assertEqual(_unmet_tool_requirements(run, CALLERS_REQ), CALLERS_REQ)


class TestDeclaredRequirementsAreWellFormed(unittest.TestCase):
    """Guards every requirement any task declares, so a typo cannot ship.

    Catches both a misspelled tool name — which would fail the task on every run —
    and an argument spec the matcher would reject at runtime.
    """

    KNOWN_TOOLS = {
        "mcp__tilth__tilth_search",
        "mcp__tilth__tilth_read",
        "mcp__tilth__tilth_files",
        "mcp__tilth__tilth_deps",
        "mcp__tilth__tilth_grok",
        "mcp__tilth__tilth_diff",
        "mcp__tilth__tilth_write",
        "mcp__tilth__tilth_savings",
    }

    def test_every_declared_requirement_is_valid(self):
        empty = _run()
        for name, task in TASKS.items():
            for requirement in task.requires_tool_use:
                for alternative in requirement.split("|"):
                    tool = alternative.strip().partition(":")[0].strip()
                    self.assertIn(
                        tool,
                        self.KNOWN_TOOLS,
                        f"task {name} requires unknown tool {tool!r}",
                    )
                # Must parse; raises ValueError on a malformed argument spec.
                _unmet_tool_requirements(empty, [requirement])


if __name__ == "__main__":
    unittest.main()
