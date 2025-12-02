"""Integration test that verifies repository search with a live LLM (Ollama)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import requests

from llmflow.core.agent import Agent
from llmflow.llm_client import LLMClient
from llmflow.tools import tool_text_search

pytestmark = pytest.mark.include_integration_test

_TEST_CONFIG_PATH = Path(__file__).with_name("ollama_llm_config.yaml")
_REQUIRED_MODEL = "qwen2.5-coder:7b"
_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _ollama_ready() -> bool:
    base_url = _OLLAMA_BASE_URL.rstrip("/")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
    except requests.RequestException:
        return False

    tags = response.json().get("models") or []
    return any(entry.get("name") == _REQUIRED_MODEL for entry in tags)


_OLLAMA_AVAILABLE = _TEST_CONFIG_PATH.exists() and _ollama_ready()


def _seed_repo(repo_root: Path) -> None:
    (repo_root / "notes").mkdir(parents=True)
    (repo_root / "notes" / "clue.txt").write_text(
        "Top secret\nLook for abU7es2 inside this file.\n",
        encoding="utf-8",
    )
    (repo_root / "notes" / "decoy.txt").write_text(
        "Nothing to see here",
        encoding="utf-8",
    )


@pytest.fixture()
def isolated_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _seed_repo(repo_root)
    monkeypatch.setattr(tool_text_search, "REPO_ROOT", repo_root)
    return repo_root


@pytest.mark.skipif(
    not _OLLAMA_AVAILABLE,
    reason=(
        "Requires a running Ollama server with the qwen2.5-coder:7b model pulled. "
        "Set OLLAMA_BASE_URL if the server is not on localhost."
    ),
)
def test_agent_finds_marker_via_real_llm(isolated_repo: Path) -> None:
    client = LLMClient(config_file=str(_TEST_CONFIG_PATH))
    agent = Agent(
        llm_client=client,
        system_prompt=(
            "You are verifying repository contents. "
            "Always call search_text_in_repository before answering. "
            "Return the relative file path and line number containing the needle."
        ),
        available_tool_tags=["search"],
        match_all_tags=False,
        verbose=False,
    )

    final_response = agent.add_user_message_and_run(
        "Find the repository location of the token 'abU7es2'. Report the relative"
        " path (from the repo root) and the line number that contains it."
    )

    assert isinstance(final_response, str)
    assert "notes/clue.txt" in final_response
    assert "abU7es2" in final_response
