from __future__ import annotations

from typing import List

from dsl.syscall_registry import SyscallRegistry

from llmflow.planning import CPLPlanOrchestrator, CPLPlanRequest, CPLPlanner
from llmflow.planning.plan_runner import CPLPlanRunner


class SequenceLLMClient:
    """Deterministic LLM stub that yields predefined CPL plans."""

    def __init__(self, responses: List[str]):
        self._responses = list(responses)
        self.calls = 0

    def generate(self, messages, tools=None, **kwargs):
        if not self._responses:
            raise RuntimeError("No more responses queued for SequenceLLMClient")
        self.calls += 1
        return {"role": "assistant", "content": self._responses.pop(0)}


def _registry_factory(capture: List[str]):
    def factory() -> SyscallRegistry:
        registry = SyscallRegistry()

        def log(message: str) -> None:
            capture.append(message)
            return None

        registry.register("log", log)
        return registry

    return factory


def test_cpl_retry_loop_end_to_end():
    first_plan_missing_main = """plan {
        function helper() : Void {
            syscall.log(\"noop\");
            return;
        }
    }
    """

    second_plan_succeeds = """plan {
        function main() : Void {
            syscall.log(\"success\");
            return;
        }
    }
    """

    llm = SequenceLLMClient([first_plan_missing_main, second_plan_succeeds])
    planner = CPLPlanner(llm, dsl_specification="SPEC")
    messages: List[str] = []
    orchestrator = CPLPlanOrchestrator(
        planner,
        runner_factory=lambda: CPLPlanRunner(
            registry_factory=_registry_factory(messages),
            dsl_specification="SPEC",
        ),
        max_retries=1,
    )

    request = CPLPlanRequest(task="Demonstrate retry", goals=["ship"])

    result = orchestrator.execute_with_retries(request, capture_trace=True)

    assert result["success"] is True
    assert len(result["attempts"]) == 2
    assert result["attempts"][0]["execution"]["success"] is False
    assert result["attempts"][0]["execution"]["errors"][0]["type"] == "validation_error"
    assert messages == ["success"]
    assert result["telemetry"]["attempt_count"] == 2
    assert "Attempt 1" in result["summary"]
    assert llm.calls == 2