from __future__ import annotations

from typing import List

from llmflow.planning import JavaPlanRequest, JavaPlanner, PlanOrchestrator
from llmflow.planning.plan_runner import PlanRunner
from llmflow.runtime.syscall_registry import SyscallRegistry


class SequenceLLMClient:
    """Deterministic LLM stub that yields predefined Java plans."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def structured_generate(self, *, messages, response_model, **kwargs):
        if not self._responses:
            raise RuntimeError("No more responses queued for SequenceLLMClient")
        self.calls += 1
        payload = self._responses.pop(0)
        return response_model(**payload)


def _registry_factory(capture: List[str]):
    def factory() -> SyscallRegistry:
        registry = SyscallRegistry()

        def log(message: str) -> None:
            capture.append(message)
            return None

        registry.register("log", log)
        return registry

    return factory


def test_java_retry_loop_end_to_end():
    first_plan_missing_main = {
        "java": """
        public class Plan {
            public void helper() {
                syscall.log("noop");
                return;
            }
        }
        """,
    }
    second_plan_succeeds = {
        "java": """
        public class Plan {
            public void main() {
                syscall.log("success");
                return;
            }
        }
        """,
    }

    llm = SequenceLLMClient([first_plan_missing_main, second_plan_succeeds])
    planner = JavaPlanner(llm, specification="SPEC")
    messages: List[str] = []
    orchestrator = PlanOrchestrator(
        planner,
        runner_factory=lambda: PlanRunner(
            registry_factory=_registry_factory(messages),
            specification="SPEC",
        ),
        max_retries=1,
    )

    request = JavaPlanRequest(task="Demonstrate retry", goals=["ship"])

    result = orchestrator.execute_with_retries(request, capture_trace=True)

    assert result["success"] is True
    assert len(result["attempts"]) == 2
    assert result["attempts"][0]["execution"]["success"] is False
    assert result["attempts"][0]["execution"]["errors"][0]["type"] == "validation_error"
    assert messages == ["success"]
    assert result["telemetry"]["attempt_count"] == 2
    assert "Attempt 1" in result["summary"]
    assert llm.calls == 2