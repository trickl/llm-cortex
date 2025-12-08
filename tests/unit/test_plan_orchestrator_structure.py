"""Tests for PlanOrchestrator structural replanning flow."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import pytest

from llmflow.planning.java_plan_compiler import JavaCompilationResult
from llmflow.planning.java_planner import JavaPlanRequest, JavaPlanResult
from llmflow.planning.plan_orchestrator import PlanOrchestrator


_SIMPLE_PLAN = """
public class Planner {
    public static void main(String[] args) {
    }
}
""".strip()


class _DummyPlanner:
    def __init__(self, plan_sources: Iterable[str]) -> None:
        self._plan_sources = list(plan_sources)
        self.requests: List[JavaPlanRequest] = []

    def generate_plan(self, request: JavaPlanRequest) -> JavaPlanResult:
        if not self._plan_sources:
            raise AssertionError("No more plan sources available")
        self.requests.append(request)
        plan_source = self._plan_sources.pop(0)
        return JavaPlanResult(
            plan_id=f"plan-{len(self.requests)}",
            plan_source=plan_source,
            raw_response={},
            prompt_messages=[],
            metadata={},
        )

    @staticmethod
    def compute_prompt_hash(_: JavaPlanRequest) -> str:
        return "dummy-hash"


class _DummyCompiler:
    def compile(  # noqa: D401 - interface compatibility
        self,
        plan_source: str,
        *,
        tool_stub_source: str | None = None,
        tool_stub_class_name: str | None = None,
        working_dir: Path | None = None,
    ) -> JavaCompilationResult:
        del plan_source, tool_stub_source, tool_stub_class_name
        if working_dir:
            working_dir.mkdir(parents=True, exist_ok=True)
            (working_dir / "Planner.class").write_bytes(b"")
        return JavaCompilationResult(success=True, command=("javac",))


class _RunnerFactory:
    def __init__(self, responses: Iterable[Dict[str, object]]) -> None:
        self._responses = list(responses)

    def __call__(self):
        if not self._responses:
            raise AssertionError("No more runner responses available")
        response = self._responses.pop(0)

        class _SingleUseRunner:
            def __init__(self, payload: Dict[str, object]) -> None:
                self._payload = payload

            def execute(self, *_, **__):
                return self._payload

        return _SingleUseRunner(response)


@pytest.fixture(name="dummy_request")
def _dummy_request_fixture() -> JavaPlanRequest:
    return JavaPlanRequest(
        task="Implement helper",
        tool_names=[],
        tool_schemas=[],
        metadata={},
    )


def test_orchestrator_replans_stub_without_retry_budget(tmp_path: Path, dummy_request: JavaPlanRequest) -> None:
    planner = _DummyPlanner([_SIMPLE_PLAN, _SIMPLE_PLAN])
    responses = [
        {
            "success": False,
            "errors": [
                {
                    "type": "stub_method",
                    "function": "hasOpenIssues",
                    "message": "Helper 'hasOpenIssues' is a placeholder.",
                    "comment": "Stub comment",
                }
            ],
            "metadata": {},
            "trace": [],
        },
        {
            "success": True,
            "errors": [],
            "metadata": {},
            "trace": [],
        },
    ]
    orchestrator = PlanOrchestrator(
        planner,
        _RunnerFactory(responses),
        max_retries=0,
        max_structure_attempts=3,
        plan_compiler=_DummyCompiler(),
        plan_artifact_root=tmp_path / "plans",
        enable_plan_cache=False,
    )

    result = orchestrator.execute_with_retries(dummy_request)

    assert result["success"] is True
    assert len(result["attempts"]) == 2
    assert planner.requests and len(planner.requests) == 2
    assert "hasOpenIssues" in planner.requests[1].task
    assert "Stub comment" in planner.requests[1].context
    helper_focus = planner.requests[1].metadata.get("helper_focus")
    assert helper_focus and helper_focus["function"] == "hasOpenIssues"
    assert helper_focus["comment"] == "Stub comment"
    assert result["attempts"][0]["execution"]["errors"][0]["type"] == "stub_method"


def test_orchestrator_honors_structure_attempt_limit(tmp_path: Path, dummy_request: JavaPlanRequest) -> None:
    planner = _DummyPlanner([_SIMPLE_PLAN, _SIMPLE_PLAN, _SIMPLE_PLAN])
    stub_response = {
        "success": False,
        "errors": [
            {
                "type": "stub_method",
                "function": "hasOpenIssues",
                "message": "Helper 'hasOpenIssues' is a placeholder.",
                "comment": "Stub comment",
            }
        ],
        "metadata": {},
        "trace": [],
    }
    responses = [stub_response, stub_response, stub_response]
    orchestrator = PlanOrchestrator(
        planner,
        _RunnerFactory(responses),
        max_retries=0,
        max_structure_attempts=2,
        plan_compiler=_DummyCompiler(),
        plan_artifact_root=tmp_path / "plans",
        enable_plan_cache=False,
    )

    result = orchestrator.execute_with_retries(dummy_request)

    assert result["success"] is False
    assert len(result["attempts"]) == 3
    assert planner.requests and len(planner.requests) == 3
    assert result["attempts"][-1]["execution"]["errors"][0]["type"] == "stub_method"
