from dataclasses import dataclass
from typing import Dict, List

import pytest

from llmflow.planning import JavaPlanRequest, PlanOrchestrator


@dataclass
class StubPlanResult:
    plan_id: str
    plan_source: str
    metadata: Dict[str, str]
    raw_response: Dict[str, str]
    prompt_messages: List[Dict[str, str]]


class DummyPlanner:
    def __init__(self, plans: List[str]):
        self._plans = list(plans)
        self.requests: List[JavaPlanRequest] = []

    def generate_plan(self, request: JavaPlanRequest) -> StubPlanResult:
        if not self._plans:
            raise RuntimeError("No more plans queued")
        self.requests.append(request)
        source = self._plans.pop(0)
        return StubPlanResult(
            plan_id=f"stub-{len(self.requests)}",
            plan_source=source,
            metadata={},
            raw_response={},
            prompt_messages=[],
        )


class DummyRunner:
    def __init__(self, outcomes: List[Dict[str, object]]):
        self._outcomes = list(outcomes)
        self.calls: List[Dict[str, object]] = []

    def execute(self, plan_source: str, **kwargs) -> Dict[str, object]:
        if not self._outcomes:
            raise RuntimeError("No more outcomes queued")
        self.calls.append({"plan_source": plan_source, **kwargs})
        return self._outcomes.pop(0)


@pytest.fixture
def runner_factory():
    def _factory(outcomes):
        runner = DummyRunner(outcomes)

        def make_runner():
            return runner

        return runner, make_runner

    return _factory


def test_orchestrator_succeeds_without_retries(runner_factory):
    planner = DummyPlanner([
        """
        public class Plan {
            public void main() {
                return;
            }
        }
        """,
    ])
    runner, make_runner = runner_factory([
        {"success": True, "errors": [], "metadata": {}},
    ])
    orchestrator = PlanOrchestrator(planner, make_runner)
    request = JavaPlanRequest(task="Do thing", goals=["goal"])

    result = orchestrator.execute_with_retries(request)

    assert result["success"] is True
    assert len(result["attempts"]) == 1
    assert runner.calls[0]["plan_source"].lstrip().startswith("public class Plan")
    assert planner.requests[0].task == "Do thing"
    assert "telemetry" in result
    assert "summary" in result
    assert "Attempt 1" in result["summary"]


def test_orchestrator_retries_and_returns_failure(runner_factory):
    planner = DummyPlanner([
        """
        public class Plan {
            public void main() {
                return;
            }
        }
        """,
        """
        public class Plan {
            public void main() {
                return;
            }
        }
        """,
    ])
    runner, make_runner = runner_factory([
        {"success": False, "errors": [{"type": "validation_error", "message": "missing main", "line": 3}]},
        {"success": False, "errors": []},
    ])
    orchestrator = PlanOrchestrator(planner, make_runner, max_retries=1)
    request = JavaPlanRequest(task="Do thing", goals=["goal"], additional_constraints=["Stay safe"])

    result = orchestrator.execute_with_retries(request)

    assert result["success"] is False
    assert len(result["attempts"]) == 2
    assert planner.requests[1].metadata["attempt_index"] == 2
    assert len(planner.requests[1].additional_constraints) >= 2
    assert result["telemetry"]["attempt_count"] == 2
    assert result["telemetry"]["attempt_summaries"][-1]["status"] == "failure"


def test_telemetry_includes_tool_usage(runner_factory):
    planner = DummyPlanner([
        """
        public class Plan {
            public void main() {
                return;
            }
        }
        """,
    ])
    runner, make_runner = runner_factory([
        {
            "success": True,
            "errors": [],
            "metadata": {"functions": 2},
            "trace": [
                {"type": "syscall_start", "name": "log"},
                {"type": "syscall_start", "name": "cloneRepo"},
                {"type": "syscall_start", "name": "log"},
            ],
        }
    ])
    orchestrator = PlanOrchestrator(planner, make_runner)
    request = JavaPlanRequest(task="Do thing", goals=["goal"])

    result = orchestrator.execute_with_retries(request)

    telemetry = result["telemetry"]
    assert telemetry["tool_usage"]["log"] == 2
    assert telemetry["tool_usage"]["cloneRepo"] == 1
    assert "cloneRepo" in result["summary"]
