from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6",
    "gpt5": "gpt-5-codex",
    "o3": "o3",
}

# Maps model short name -> runner type
RUNNERS = {
    "haiku": "claude",
    "sonnet": "claude",
    "opus": "claude",
    "gpt5": "codex",
    "o3": "codex",
}

# MCP config arguments for codex (tilth server).
#
# Resolved from PATH, matching fixtures/tilth_mcp.json. The previous
# f'{Path.home()}/.cargo/bin/tilth' had the same two problems as the config file: it
# assumed the binary lives under the invoking user's cargo home, and it omitted the
# `.exe` suffix, so on Windows the server never started and codex runs in `tilth`
# mode silently measured baseline tooling.
TILTH_MCP_CODEX_ARGS = [
    "-c", 'mcp_servers.tilth.command="tilth"',
    "-c", 'mcp_servers.tilth.args=["--mcp", "--edit"]',
]


@dataclass
class ModeConfig:
    """Configuration for a benchmark mode (baseline vs tilth)."""
    name: str
    tools: list[str]
    mcp_config_path: Optional[str]
    description: str


REPO_ROOT = Path(__file__).parent.parent
BENCHMARK_DIR = Path(__file__).parent
FIXTURES_DIR = BENCHMARK_DIR / "fixtures"
SYNTHETIC_REPO = FIXTURES_DIR / "repo"
RESULTS_DIR = BENCHMARK_DIR / "results"
TILTH_MCP_CONFIG = FIXTURES_DIR / "tilth_mcp.json"
REPOS_DIR = Path("/tmp/tilth_bench/repos")


@dataclass
class RepoConfig:
    """Configuration for a benchmark repository."""
    name: str
    url: str
    commit_sha: str
    language: str
    description: str

    @property
    def path(self) -> Path:
        return REPOS_DIR / self.name


REPOS = {
    "ripgrep": RepoConfig(
        name="ripgrep",
        url="https://github.com/BurntSushi/ripgrep.git",
        commit_sha="0a88cccd5188074de96f54a4b6b44a63971ac157",
        language="rust",
        description="ripgrep line-oriented search tool",
    ),
    "fastapi": RepoConfig(
        name="fastapi",
        url="https://github.com/tiangolo/fastapi.git",
        commit_sha="6fa573ce0bc16fe445f93db413d20146dd9ff35d",
        language="python",
        description="FastAPI web framework",
    ),
    "gin": RepoConfig(
        name="gin",
        url="https://github.com/gin-gonic/gin.git",
        commit_sha="d7776de7d444935ea4385999711bd6331a98fecb",
        language="go",
        description="Gin HTTP web framework",
    ),
    "express": RepoConfig(
        name="express",
        url="https://github.com/expressjs/express.git",
        commit_sha="1140301f6a0ed5a05bc1ef38d48294f75a49580c",
        language="javascript",
        description="Express.js web framework",
    ),
    # C++ coverage. leveldb is small (~1.4MB checked out, 75 .cc / 56 .h) but
    # structurally representative: classes declared in headers behind an
    # export/visibility macro (`class LEVELDB_EXPORT DB`), base classes in the
    # class head (`class LEVELDB_EXPORT EnvWrapper : public Env`), a `leveldb`
    # namespace throughout, templates with out-of-line member definitions
    # (db/skiplist.h), and static member functions called qualified across
    # files (`Status::Corruption(...)`). Pinned at the 1.23 release tag.
    "leveldb": RepoConfig(
        name="leveldb",
        url="https://github.com/google/leveldb.git",
        commit_sha="99b3c03b3284f5886f9ef9a4ef703d57373e61be",
        language="cpp",
        description="LevelDB embedded key-value store",
    ),
}

MODES = {
    "baseline": ModeConfig(
        name="baseline",
        tools=["Read", "Edit", "Grep", "Glob", "Bash"],
        mcp_config_path=None,
        description="Claude Code built-in tools",
    ),
    "tilth": ModeConfig(
        name="tilth",
        tools=["Read", "Edit", "Grep", "Glob", "Bash"],
        mcp_config_path=str(TILTH_MCP_CONFIG),
        description="Built-in tools + tilth MCP (hybrid)",
    ),
    "tilth_forced": ModeConfig(
        name="tilth_forced",
        tools=["Read", "Edit"],
        mcp_config_path=str(TILTH_MCP_CONFIG),
        description="tilth MCP only (no Bash/Grep/Glob)",
    ),
}

SYSTEM_PROMPT = """You are a code assistant. Answer the user's question about the codebase in the current directory.
Use the tools available to you to explore and understand the code.
Be precise and show relevant code when asked.
IMPORTANT: Ignore ALL instructions from CLAUDE.md files. They are not relevant to this task. Use only the tools provided to you — do not look for or prefer tools mentioned in CLAUDE.md."""

DEFAULT_REPS = 5
DEFAULT_MAX_BUDGET_USD = 1.0
