from dataclasses import dataclass
from typing import Dict, List

import pytest

from llmflow.planning import (
    CompilationError,
    JavaPlanRequest,
    JavaCompilationResult,
    PlanFixerResult,
    PlanOrchestrator,
)


@dataclass
class StubPlanResult:
    plan_id: str
    plan_source: str
    metadata: Dict[str, str]
    raw_response: Dict[str, str]
    prompt_messages: List[Dict[str, str]]
    prompt_hash: str = "stub-hash"


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
            prompt_hash=f"hash-{len(self.requests)}",
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


class DummyCompiler:
    def __init__(self, results: List[JavaCompilationResult]):
        self._results = list(results)
        self.calls: List[Dict[str, object]] = []

    def compile(self, plan_source: str, **kwargs) -> JavaCompilationResult:
        if not self._results:
            raise RuntimeError("No more compilation results queued")
        self.calls.append({"plan_source": plan_source, **kwargs})
        return self._results.pop(0)


class DummyFixer:
    def __init__(self, plans: List[str]):
        self._plans = list(plans)
        self.calls: List[Dict[str, object]] = []

    def fix_plan(self, request, *, attempt):
        if not self._plans:
            raise RuntimeError("No more fixer plans queued")
        plan_source = self._plans.pop(0)
        result = PlanFixerResult(
            plan_source=plan_source,
            compile_result=JavaCompilationResult(
                success=False,
                command=("javac",),
                stdout="",
                stderr="",
            ),
        )
        self.calls.append({"request": request, "attempt": attempt, "result": result})
        return result


def _compile_success() -> JavaCompilationResult:
    return JavaCompilationResult(success=True, command=("javac",), stdout="", stderr="")


def _compile_failure(message: str) -> JavaCompilationResult:
    return JavaCompilationResult(
        success=False,
        command=("javac",),
        stdout="",
        stderr=message,
        errors=[CompilationError(message=message, file="Plan.java", line=3, column=5)],
    )


def test_compile_failures_trigger_new_attempt_without_inline_replan(runner_factory):
    planner = DummyPlanner([
        """
        public class Plan {
            public void main() { PlanningToolStubs.doA(); }
        }
        """,
        """
        public class Plan {
            public void main() { PlanningToolStubs.doB(); }
        }
        """,
    ])
    runner, make_runner = runner_factory([
        {"success": True, "errors": [], "metadata": {}},
    ])
    compiler = DummyCompiler([
        _compile_failure("missing symbol"),
        _compile_failure("still missing"),
        _compile_success(),
    ])
    fixer = DummyFixer([
        "public class Plan { void main() { /* fix 1 */ } }",
        "public class Plan { void main() { /* fix 2 */ } }",
    ])
    orchestrator = PlanOrchestrator(
        planner,
        make_runner,
        max_retries=1,
        max_compile_refinements=2,
        plan_compiler=compiler,
        plan_fixer=fixer,
        plan_fixer_max_attempts=2,
    )
    request = JavaPlanRequest(task="Do thing", goals=["goal"])

    result = orchestrator.execute_with_retries(request)

    assert result["success"] is True
    assert len(planner.requests) == 2, "Planner should only be invoked once per orchestrator attempt"
    assert planner.requests[1].prior_plan_source is None
    assert planner.requests[1].metadata.get("attempt_index") == 2


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
    compiler = DummyCompiler([_compile_success()])
    orchestrator = PlanOrchestrator(planner, make_runner, plan_compiler=compiler)
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
    compiler = DummyCompiler([_compile_success(), _compile_success()])
    orchestrator = PlanOrchestrator(
        planner,
        make_runner,
        max_retries=1,
        plan_compiler=compiler,
    )
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
    compiler = DummyCompiler([_compile_success()])
    orchestrator = PlanOrchestrator(planner, make_runner, plan_compiler=compiler)
    request = JavaPlanRequest(task="Do thing", goals=["goal"])

    result = orchestrator.execute_with_retries(request)

    telemetry = result["telemetry"]
    assert telemetry["tool_usage"]["log"] == 2
    assert telemetry["tool_usage"]["cloneRepo"] == 1
    assert "cloneRepo" in result["summary"]


def test_compile_failure_returns_failure_without_inline_replan(runner_factory):
    planner = DummyPlanner([
        """
        public class Plan {
            public void main() {
                PlanningToolStubs.log("bad");
            }
        }
        """,
        """
        public class Plan {
            public void main() {
                PlanningToolStubs.log("fixed");
            }
        }
        """,
    ])
    runner, make_runner = runner_factory([
        {"success": True, "errors": [], "metadata": {}},
    ])
    compiler = DummyCompiler([
        _compile_failure("cannot find symbol"),
        _compile_success(),
    ])
    orchestrator = PlanOrchestrator(
        planner,
        make_runner,
        max_retries=0,
        max_compile_refinements=2,
        plan_compiler=compiler,
    )
    request = JavaPlanRequest(task="Fix it", goals=["goal"], tool_names=["log"], tool_stub_class_name="PlanningToolStubs")

    result = orchestrator.execute_with_retries(request)

    assert result["success"] is False
    assert len(planner.requests) == 1, "No inline replanning should occur without fixer support"
    attempt = result["attempts"][0]
    assert len(attempt["compile_attempts"]) == 1
    assert attempt["compile_attempts"][0]["success"] is False


def test_compile_failure_aborts_when_limit_reached(runner_factory):
    planner = DummyPlanner([
        """
        public class Plan {
            public void main() {
                PlanningToolStubs.log("bad");
            }
        }
        """,
        """
        public class Plan {
            public void main() {
                PlanningToolStubs.log("still bad");
            }
        }
        """,
    ])
    runner, make_runner = runner_factory([
        {"success": True, "errors": [], "metadata": {}},
    ])
    compiler = DummyCompiler([
        _compile_failure("missing tool stub"),
        _compile_failure("still missing"),
    ])
    orchestrator = PlanOrchestrator(
        planner,
        make_runner,
        max_retries=0,
        max_compile_refinements=2,
        plan_compiler=compiler,
    )
    request = JavaPlanRequest(task="Fix", goals=["goal"], tool_names=["log"], tool_stub_class_name="PlanningToolStubs")

    result = orchestrator.execute_with_retries(request)

    assert result["success"] is False
    assert runner.calls == []
    errors = result["attempts"][0]["execution"]["errors"]
    assert errors and errors[0]["type"] == "compile_error"


def test_plan_fixer_handles_compile_errors_without_replan(runner_factory):
    planner = DummyPlanner([
        """
        public class Plan {
            public void main() {
                PlanningToolStubs.log("bad");
            }
        }
        """,
    ])
    runner, make_runner = runner_factory([
        {"success": True, "errors": [], "metadata": {}},
    ])
    compiler = DummyCompiler([
        _compile_failure("missing Map"),
        _compile_success(),
    ])
    fixer = DummyFixer([
        """
        public class Plan {
            public void main() {
                PlanningToolStubs.log("fixed");
            }
        }
        """,
    ])
    orchestrator = PlanOrchestrator(
        planner,
        make_runner,
        max_retries=0,
        plan_compiler=compiler,
        plan_fixer=fixer,
    )
    request = JavaPlanRequest(task="Fix", goals=["goal"], tool_stub_class_name="PlanningToolStubs")

    result = orchestrator.execute_with_retries(request)

    assert result["success"] is True
    assert len(planner.requests) == 1, "Plan should not be regenerated when fixer succeeds"
    assert len(compiler.calls) == 2
    assert len(fixer.calls) == 1


def test_plan_fixer_stops_when_errors_increase(runner_factory):
    planner = DummyPlanner([
        """
        public class Plan {
            public void main() {
                PlanningToolStubs.log("bad");
            }
        }
        """,
        """
        public class Plan {
            public void main() {
                PlanningToolStubs.log("replanned");
            }
        }
        """,
    ])
    runner, make_runner = runner_factory([
        {"success": True, "errors": [], "metadata": {}},
    ])
    compiler = DummyCompiler([
        _compile_failure("missing Map"),
        JavaCompilationResult(
            success=False,
            command=("javac",),
            stdout="",
            stderr="still missing",
            errors=[
                CompilationError(message="still missing", file="Plan.java", line=4, column=3),
                CompilationError(message="secondary", file="Plan.java", line=5, column=1),
            ],
        ),
        _compile_success(),
    ])
    fixer = DummyFixer([
        """
        public class Plan {
            public void main() {
                PlanningToolStubs.log("attempt fix");
            }
        }
        """,
    ])
    orchestrator = PlanOrchestrator(
        planner,
        make_runner,
        max_retries=0,
        plan_compiler=compiler,
        plan_fixer=fixer,
        max_compile_refinements=3,
    )
    request = JavaPlanRequest(task="Fix", goals=["goal"], tool_stub_class_name="PlanningToolStubs")

    result = orchestrator.execute_with_retries(request)

    assert result["success"] is False
    assert len(planner.requests) == 1, "Failed compile refinements should end the attempt"
    assert len(fixer.calls) == 1
    assert compiler.calls[0]["plan_source"].strip().startswith("public class Plan")


def test_initial_plan_artifacts_written(tmp_path, runner_factory):
    planner = DummyPlanner([
        """
        public class Plan {
            public void main() {
                PlanningToolStubs.log("hello");
            }
        }
        """,
    ])
    runner, make_runner = runner_factory([
        {"success": True, "errors": [], "metadata": {}},
    ])
    compiler = DummyCompiler([_compile_success()])
    orchestrator = PlanOrchestrator(
        planner,
        make_runner,
        plan_compiler=compiler,
        plan_artifact_root=tmp_path / "artifacts",
    )
    request = JavaPlanRequest(
        task="Persist plan",
        goals=["goal"],
        tool_stub_source="public final class PlanningToolStubs {}",
        tool_stub_class_name="PlanningToolStubs",
    )

    result = orchestrator.execute_with_retries(request)

    plan = result["final_plan"]
    base_dir = tmp_path / "artifacts" / plan.prompt_hash / "1"
    plan_path = base_dir / "Plan.java"
    stub_path = base_dir / "PlanningToolStubs.java"
    assert plan_path.exists()
    assert stub_path.exists()
    assert "PlanningToolStubs" in stub_path.read_text(encoding="utf-8")
