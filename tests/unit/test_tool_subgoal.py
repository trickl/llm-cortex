"""Unit tests for the subgoal tooling wrappers."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from llmflow.tools import tool_subgoal


class FakeLLMClient:
    def __init__(self, config_file: str):
        self.config_file = config_file


class FakeAgent:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        FakeAgent.last_kwargs = kwargs
        FakeAgent.last_prompt = ""

    def add_user_message_and_run(self, prompt: str) -> str:
        FakeAgent.last_prompt = prompt
        return "subgoal complete"

    def get_context_trace(self) -> List[Dict[str, Any]]:
        return [{"stage": "test", "message_count": 2}]


def test_run_subgoal_invokes_nested_agent(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setattr(tool_subgoal, "LLMClient", lambda config_file: FakeLLMClient(config_file))
    monkeypatch.setattr(tool_subgoal, "Agent", FakeAgent)

    llm_path = tmp_path / "llm.yaml"
    llm_path.write_text("provider_config:\n  provider: generic\n  model: fake\n", encoding="utf-8")

    result = tool_subgoal.run_subgoal(
        goal_name="test_goal",
        objective="Do something useful",
        context={"foo": "bar"},
        allowed_tool_tags=["file_system"],
        llm_config_path=str(llm_path),
        verbose=True,
    )

    assert result["success"] is True
    assert result["final_message"] == "subgoal complete"
    assert FakeAgent.last_kwargs["available_tool_tags"] == ["file_system"]
    assert "Context packet" in FakeAgent.last_prompt


def test_run_subgoal_validates_inputs():
    error = tool_subgoal.run_subgoal(goal_name=" ", objective="X")
    assert error["success"] is False
    assert "goal_name" in error["error"]

    error = tool_subgoal.run_subgoal(goal_name="goal", objective=" ")
    assert error["success"] is False
    assert "objective" in error["error"]


def test_run_understand_issue_wrapper_builds_context(monkeypatch: pytest.MonkeyPatch):
    captured: Dict[str, Any] = {}

    def fake_run_subgoal(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {"success": True, "final_message": "diagnosis"}

    monkeypatch.setattr(tool_subgoal, "run_subgoal", fake_run_subgoal)

    result = tool_subgoal.run_understand_issue_subgoal(
        issue_id="abc123",
        issue_summary="Line too long",
        repo_path="/work/repo",
        candidate_files=["foo.py"],
        lint_rule="E501",
    )

    assert result["success"] is True
    assert captured["goal_name"] == "understand_issue"
    assert captured["allowed_tool_tags"] == ["file_system", "text_search"]
    assert captured["context"]["candidate_files"] == ["foo.py"]


def test_run_patch_issue_wrapper_builds_context(monkeypatch: pytest.MonkeyPatch):
    captured: Dict[str, Any] = {}

    def fake_run_subgoal(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {"success": True, "final_message": "patched"}

    monkeypatch.setattr(tool_subgoal, "run_subgoal", fake_run_subgoal)

    result = tool_subgoal.run_patch_issue_subgoal(
        repo_path="/work/repo",
        issue_id="abc123",
        diagnosis_summary="Need to shorten line",
        target_files=["foo.py"],
        tests_to_run=["pytest foo.py"],
        desired_outcome="Line under 100 chars",
    )

    assert result["success"] is True
    assert captured["goal_name"] == "patch_issue"
    assert captured["allowed_tool_tags"] == ["file_system", "git", "shell"]
    assert captured["context"]["tests_to_run"] == ["pytest foo.py"]


def test_run_understand_issue_wrapper_defaults_summary(monkeypatch: pytest.MonkeyPatch):
    captured: Dict[str, Any] = {}

    def fake_run_subgoal(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {"success": True}

    monkeypatch.setattr(tool_subgoal, "run_subgoal", fake_run_subgoal)

    tool_subgoal.run_understand_issue_subgoal(
        issue_id="abc123",
        repo_path="/work/repo",
    )

    assert captured["context"]["issue_summary"] == "Summary not provided"


def test_run_patch_issue_wrapper_defaults_diagnosis(monkeypatch: pytest.MonkeyPatch):
    captured: Dict[str, Any] = {}

    def fake_run_subgoal(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {"success": True}

    monkeypatch.setattr(tool_subgoal, "run_subgoal", fake_run_subgoal)

    tool_subgoal.run_patch_issue_subgoal(
        repo_path="/work/repo",
        issue_id="abc123",
    )

    assert captured["context"]["diagnosis_summary"] == "Diagnosis not provided"


def test_run_fetch_issue_subgoal_includes_filters(monkeypatch: pytest.MonkeyPatch):
    captured: Dict[str, Any] = {}

    def fake_run_subgoal(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {"success": True}

    monkeypatch.setattr(tool_subgoal, "run_subgoal", fake_run_subgoal)

    tool_subgoal.run_fetch_issue_subgoal(
        owner_key_or_id="trickl",
        project_key_or_id="agent-cortex",
        categories=["lint"],
        statuses=["open"],
        levels=["medium"],
    )

    assert captured["goal_name"] == "fetch_issue"
    assert captured["allowed_tool_tags"] == ["qlty_single_issue"]
    assert captured["context"]["levels"] == ["medium"]


def test_run_prepare_workspace_subgoal_uses_env(monkeypatch: pytest.MonkeyPatch):
    captured: Dict[str, Any] = {}

    def fake_run_subgoal(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {"success": True}

    monkeypatch.setattr(tool_subgoal, "run_subgoal", fake_run_subgoal)
    monkeypatch.setenv("PROJECT_REPO_ROOT", "/tmp/worktree")

    tool_subgoal.run_prepare_workspace_subgoal(
        repo_url="git@example.com:repo.git",
        issue_reference="lint-1",
        branch_prefix="fix/issue-",
    )

    assert captured["goal_name"] == "prepare_workspace"
    assert "file_system" in captured["allowed_tool_tags"]
    assert captured["context"]["project_repo_root"] == "/tmp/worktree"


def test_run_finalize_issue_subgoal_captures_context(monkeypatch: pytest.MonkeyPatch):
    captured: Dict[str, Any] = {}

    def fake_run_subgoal(**kwargs: Any) -> Dict[str, Any]:
        captured.update(kwargs)
        return {"success": True}

    monkeypatch.setattr(tool_subgoal, "run_subgoal", fake_run_subgoal)

    tool_subgoal.run_finalize_issue_subgoal(
        repo_path="/work/repo",
        branch_name="feature",
        issue_id="abc123",
        tests_summary="pytest -q",
    )

    assert captured["goal_name"] == "finalize_issue"
    assert captured["context"]["tests_summary"] == "pytest -q"
