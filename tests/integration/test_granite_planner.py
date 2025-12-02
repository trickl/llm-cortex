"""Granite-focused integration test for JavaPlanner structured output."""
from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Sequence

import pytest
import requests

from llmflow.llm_client import LLMClient
from llmflow.planning import JavaPlanner, JavaPlanRequest

pytestmark = pytest.mark.include_integration_test

_TEST_CONFIG_PATH = Path(__file__).with_name("ollama_llm_config.yaml")
_REQUIRED_MODEL = "qwen2.5-coder:7b"
_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_FORCE_GRANITE_TESTS = os.getenv("LLMFLOW_FORCE_GRANITE_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
}
_AGENT_TASK_HEADER = (
    "Create a Java class named Planner that carries out the user's request. "
    "Keep the structure minimal, lean on helper methods for decomposition, and "
    "invoke available planning tools via the stub class whenever a step can be executed directly."
)
_AGENT_STYLE_TOOLS = [
    "log",
    "listFilesInTree",
    "readTextFile",
    "overwriteTextFile",
    "applyTextRewrite",
    "cloneRepo",
    "stagePaths",
    "commitChanges",
]


def _granite_ready() -> bool:
    if not _TEST_CONFIG_PATH.exists():
        return False

    base_url = _OLLAMA_BASE_URL.rstrip("/")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
    except requests.RequestException:
        return False

    tags = response.json().get("models") or []
    if not any(entry.get("name") == _REQUIRED_MODEL for entry in tags):
        return False

    try:
        probe = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": _REQUIRED_MODEL,
                "messages": [{"role": "user", "content": "ping"}],
                "stream": False,
            },
            timeout=5,
        )
        probe.raise_for_status()
    except requests.RequestException:
        return False

    return True


_GRANITE_AVAILABLE = _granite_ready()


def _agent_style_task(user_request: str) -> str:
    normalized = textwrap.dedent(user_request).strip()
    return f"{_AGENT_TASK_HEADER}\n\nUser request:\n{normalized}"


def _agent_style_context(summary: str, conversation: Sequence[tuple[str, str]] | None = None) -> str:
    lines = []
    if summary:
        lines.append(f"Previous plan summary:\n{summary.strip()}")
    if conversation:
        lines.append("Recent conversation:")
        for role, message in conversation:
            lines.append(f"- {role}: {message.strip()}")
    return "\n\n".join(lines).strip()


_COMPLEXITY_SCENARIOS = [
    {
        "name": "minimal_smoke",
        "task": (
            "Draft a Java class named PlanSteps that logs three numbered remediation steps "
            "using PlanningToolStubs.log and leaves TODO comments for any shell commands."
        ),
        "context": (
            "You must call PlanningToolStubs.log at least once."
            "Respond using the define_java_plan schema only."
        ),
        "goals": ["Summarize a remediation workflow"],
        "allowed_tools": ["log"],
        "metadata": {"plan_id": "granite-structured-smoke"},
    },
    {
        "name": "agent_lite",
        "task": _agent_style_task(
            "Audit repo branches, log the three most recent, and note any that look stale."
        ),
        "context": _agent_style_context(
            "Most recent summary captured flaky integration tests and suggested fresh logs.",
            (
                ("user", "Please gather current repo branches before rerunning CI."),
                ("assistant", "Acknowledged—collecting branch details next."),
            ),
        ),
        "goals": [
            "Surface unmerged branches",
            "Log remediation notes",
        ],
        "allowed_tools": _AGENT_STYLE_TOOLS[:4],
        "metadata": {"plan_id": "granite-agent-lite"},
        "additional_constraints": [
            "Ensure each helper calls either another helper or an allowed planning tool.",
            "Limit helper bodies to at most seven statements.",
        ],
    },
    {
        "name": "agent_full",
        "task": _agent_style_task(
            "Diagnose failing quality checks, patch the offending configuration, and rerun tests "
            "only after updating documentation entries."
        ),
        "context": _agent_style_context(
            "Previous plan summarized multiple tool errors; new attempt must avoid redundant tool calls.",
            (
                ("user", "Focus on the logging plus git tools; no shell access."),
                ("assistant", "Will rely on structured plan execution."),
                ("user", "Remember to leave TODOs for unclear sections."),
            ),
        ),
        "goals": [
            "Resolve quality gate failure",
            "Update relevant documentation",
            "Stage and summarize changes",
        ],
        "allowed_tools": _AGENT_STYLE_TOOLS,
        "metadata": {"plan_id": "granite-agent-full"},
        "additional_constraints": [
            "Follow the Java planning specification strictly, emitting only one top-level class.",
            "Prefer helper decomposition depth of at least three levels.",
            "If a tool cannot run, add TODO comments documenting the gap.",
        ],
    },
]


@pytest.mark.skipif(
    not _GRANITE_AVAILABLE and not _FORCE_GRANITE_TESTS,
    reason=(
        "Requires a running Ollama server with the qwen2.5-coder:7b model pulled. "
        "Set OLLAMA_BASE_URL if the server is not on localhost."
    ),
)
@pytest.mark.parametrize("scenario", _COMPLEXITY_SCENARIOS, ids=lambda data: data["name"])
def test_granite_structured_java_plan_levels(scenario: dict[str, object]) -> None:
    """Exercise Granite JSON mode with progressively richer prompts."""

    try:
        llm_client = LLMClient(config_file=str(_TEST_CONFIG_PATH))
    except RuntimeError as exc:
        pytest.skip(f"Ollama server unavailable or lacks tool support: {exc}")
    planner = JavaPlanner(llm_client)

    request = JavaPlanRequest(
        task=scenario["task"],
        context=scenario.get("context"),
        goals=scenario.get("goals", []),
        tool_names=scenario.get("allowed_tools", ["log"]),
        additional_constraints=scenario.get("additional_constraints", []),
        metadata=scenario.get("metadata", {}),
    )

    print(
        f"[Granite] Running scenario '{scenario['name']}' with tools={request.tool_names}",
        flush=True,
    )
    result = planner.generate_plan(request)

    assert result.plan_source.strip(), "Planner should return non-empty Java source."
    assert "class" in result.plan_source, "Java source must declare a class."
    assert result.raw_response.get("java"), "Structured response must include the 'java' field."
