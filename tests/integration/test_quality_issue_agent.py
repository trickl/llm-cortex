"""Full-stack integration test for the quality issue agent."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Set

import pytest
import requests
import yaml

from llmflow.core.agent import Agent
from llmflow.llm_client import LLMClient
from llmflow.tools import get_module_for_tool_name, load_tool_module

pytestmark = pytest.mark.include_integration_test

_TEST_CONFIG_PATH = Path(__file__).with_name("ollama_llm_config.yaml")
_REQUIRED_MODEL = "granite4:3b"
_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_QUALITY_AGENT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "agents"
    / "quality_issue_agent_ollama.yaml"
)
_REQUIRED_ENV_VARS: Iterable[str] = (
    "QLTY_API_TOKEN",
    "QLTY_OWNER_KEY",
    "QLTY_PROJECT_KEY",
    "GITHUB_TOKEN",
    "GITHUB_REPO_SLUG",
    "QUALITY_AGENT_TEST_REPO_URL",
)


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


@pytest.mark.skipif(
    not _QUALITY_AGENT_CONFIG.exists(),
    reason="Requires configs/agents/quality_issue_agent_ollama.yaml to be present.",
)
@pytest.mark.skipif(
    not _OLLAMA_AVAILABLE,
    reason=(
        "Requires a running Ollama server with the granite4:3b model pulled. "
        "Set OLLAMA_BASE_URL if the server is not on localhost."
    ),
)
def test_quality_issue_agent_closes_first_lint_issue_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing_env = [name for name in _REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing_env:
        pytest.skip(f"Missing environment variables: {', '.join(sorted(missing_env))}")

    repo_url = os.environ["QUALITY_AGENT_TEST_REPO_URL"]
    owner_key = os.environ["QLTY_OWNER_KEY"]
    project_key = os.environ["QLTY_PROJECT_KEY"]
    repo_slug = os.environ["GITHUB_REPO_SLUG"]

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setenv("PROJECT_REPO_ROOT", str(repo_root))

    config_data = yaml.safe_load(_QUALITY_AGENT_CONFIG.read_text(encoding="utf-8"))
    agent_config = config_data["agent"]
    workflow = agent_config.get("workflow", {})
    agent_goals = agent_config.get("goals")
    tag_settings = agent_config.get("tools", {}).get("tags", {})
    include_tags: List[str] = list(tag_settings.get("include", []))
    if "shell" not in include_tags:
        include_tags.append("shell")
    explicit_tools: Set[str] = set(agent_config.get("tools", {}).get("explicit", []))

    llm_client = LLMClient(config_file=str(_TEST_CONFIG_PATH))

    for tool_name in explicit_tools:
        module_name = get_module_for_tool_name(tool_name)
        if module_name:
            load_tool_module(module_name)

    agent = Agent(
        llm_client=llm_client,
        system_prompt=agent_config["base_context"]["system_prompt"],
        initial_goals=agent_goals,
        available_tool_tags=include_tags,
        match_all_tags=tag_settings.get("match_all", False),
        max_iterations=workflow.get("max_iterations"),
        verbose=True,
    )

    activated_tool_names = {
        schema["function"]["name"] for schema in agent.active_tools_schemas
    }

    assert llm_client.model == _REQUIRED_MODEL
    assert activated_tool_names, "Agent should load at least one tool schema."
    assert activated_tool_names.issuperset(explicit_tools)
    assert agent.available_tool_tags == include_tags
    assert agent.max_iterations == workflow.get("max_iterations")

    user_prompt = (
        f"Target owner/project: {owner_key}/{project_key}. "
        f"Repository URL: {repo_url} (slug {repo_slug}). "
        "Use your workflow to address the first open lint issue returned by the Qlty tools, "
        "or state clearly that no lint issues are currently open before stopping."
    )

    final_message = agent.add_user_message_and_run(user_prompt)

    assert isinstance(final_message, str) and final_message.strip(), "Agent must return a final message."
    assert "pull request" in final_message.lower() or "no lint issues" in final_message.lower()
