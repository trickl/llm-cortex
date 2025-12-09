import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pytest

from llmflow.planning import (
    CompilationError,
    JavaPlanRequest,
    JavaCompilationResult,
    PlanFixerResult,
    PlanOrchestrator,
)
from llmflow.planning.plan_cache import PlanCache, compute_stub_hash


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
        prompt_hash = self.compute_prompt_hash(request)
        return StubPlanResult(
            plan_id=f"stub-{len(self.requests)}",
            plan_source=source,
            metadata={},
            raw_response={},
            prompt_messages=[],
            prompt_hash=prompt_hash,
        )

    def compute_prompt_hash(self, request: JavaPlanRequest) -> str:
        payload = {
            "task": request.task,
            "context": request.context,
            "constraints": list(request.additional_constraints or []),
            "helper_focus": (request.metadata or {}).get("helper_focus"),
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
        return f"hash-{digest}"


class DummyRunner:
    def __init__(self, outcomes: List[Dict[str, object]]):
        self._outcomes = list(outcomes)
        self.calls: List[Dict[str, object]] = []
        self.bound_artifacts: List[object] = []

    def execute(self, plan_source: str, **kwargs) -> Dict[str, object]:
        if not self._outcomes:
            raise RuntimeError("No more outcomes queued")
        self.calls.append({"plan_source": plan_source, **kwargs})
        return self._outcomes.pop(0)

    def bind_plan_artifacts(self, artifacts):
        self.bound_artifacts.append(artifacts)


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
        result = self._results.pop(0)
        working_dir = kwargs.get("working_dir")
        if working_dir is not None:
            work_path = Path(working_dir)
            work_path.mkdir(parents=True, exist_ok=True)
            if result.success:
                marker = work_path / "Dummy.class"
                marker.write_bytes(b"")
        return result


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


def _seed_cached_plan(
    plan_root: Path,
    *,
    prompt_hash: str,
    plan_source: str,
    stub_source: str,
    stub_class: str,
) -> None:
    attempt_dir = plan_root / prompt_hash / "1"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "Plan.java").write_text(plan_source.strip() + "\n", encoding="utf-8")
    classes_dir = attempt_dir / "classes"
    classes_dir.mkdir(parents=True, exist_ok=True)
    (classes_dir / "Dummy.class").write_bytes(b"0")
    (attempt_dir / "clean").touch()
    metadata = {
        "plan_id": prompt_hash,
        "prompt_hash": prompt_hash,
        "plan_metadata": {},
        "prompt_messages": [],
        "raw_response": {},
        "artifact_attempt": 1,
        "tool_stub_class_name": stub_class,
        "tool_stub_hash": compute_stub_hash(stub_source),
        "stored_at": "2025-01-01T00:00:00Z",
    }
    (attempt_dir / "plan_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


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
        enable_plan_cache=False,
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
    orchestrator = PlanOrchestrator(planner, make_runner, plan_compiler=compiler, enable_plan_cache=False)
    request = JavaPlanRequest(task="Do thing", goals=["goal"])

    result = orchestrator.execute_with_retries(request)

    assert result["success"] is True
    assert len(result["attempts"]) == 1
    assert runner.calls[0]["plan_source"].lstrip().startswith("public class Plan")
    assert planner.requests[0].task == "Do thing"
    assert "telemetry" in result
    assert "summary" in result
    assert "Attempt 1" in result["summary"]
    assert runner.bound_artifacts, "Runner should capture plan artifacts"
    artifact = runner.bound_artifacts[0]
    assert artifact.plan_class_name == "Plan"
    assert artifact.classes_dir.exists()


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
        enable_plan_cache=False,
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
    orchestrator = PlanOrchestrator(planner, make_runner, plan_compiler=compiler, enable_plan_cache=False)
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
        enable_plan_cache=False,
    )
    request = JavaPlanRequest(task="Fix it", goals=["goal"], tool_names=["log"], tool_stub_class_name="PlanningToolStubs")

    result = orchestrator.execute_with_retries(request)

    assert result["success"] is False
    assert len(planner.requests) == 1, "No inline replanning should occur without fixer support"
    attempt = result["attempts"][0]
    assert len(attempt["compile_attempts"]) == 1
    assert attempt["compile_attempts"][0]["success"] is False

def test_helper_focus_retries_apply_helper_specific_prompt(runner_factory):
    planner = DummyPlanner([
        """
        public class Plan {
            public void main() { }
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
        enable_plan_cache=False,
    )
    base_request = JavaPlanRequest(
        task="Original task",
        context="Some context",
        additional_constraints=["Stay safe"],
    )
    helper_focus = {
        "function": "hasOpenIssues",
        "comment": "Stub note",
        "message": "Helper is still a placeholder",
    }
    repair_hints = ["Implement helper method 'hasOpenIssues' so it performs the described behavior."]

    augmented = orchestrator._augment_request(
        base_request,
        attempt_idx=1,
        repair_hints=repair_hints,
        helper_focus=helper_focus,
    )

    assert augmented.task != base_request.task
    assert "hasOpenIssues" in augmented.task
    assert augmented.context == base_request.context
    assert "Focus exclusively" not in augmented.task
    assert "Broader workflow context" not in (augmented.context or "")
    assert "Stub note" in augmented.task
    assert "Helper note" not in augmented.task
    assert repair_hints[0] in augmented.additional_constraints
    assert augmented.metadata["helper_focus"]["function"] == "hasOpenIssues"
    assert augmented.metadata["repair_hints"] == repair_hints

    no_helper = orchestrator._augment_request(
        base_request,
        attempt_idx=1,
        repair_hints=repair_hints,
        helper_focus=None,
    )
    assert repair_hints[0] in no_helper.additional_constraints
    assert no_helper.task == base_request.task
    assert no_helper.context == base_request.context


def test_helper_focus_helper_cache_checked_and_persisted(runner_factory, tmp_path: Path, monkeypatch):
    cached_plan = """
    public class Planner {
        public static void main(String[] args) throws Exception {
            hasOpenIssues();
        }

        private static boolean hasOpenIssues() {
            // TODO: implement
            return false;
        }
    }
    """.strip()
    helper_plan = """
    public class Planner {
        public static void main(String[] args) throws Exception {
            hasOpenIssues();
        }

        private static boolean hasOpenIssues() throws Exception {
            PlanningToolStubs.log("checking");
            return true;
        }
    }
    """.strip()
    planner = DummyPlanner([helper_plan])
    runner, make_runner = runner_factory([
        {
            "success": False,
            "errors": [
                {
                    "type": "stub_method",
                    "function": "hasOpenIssues",
                    "message": "Helper is still a placeholder",
                    "comment": "Stub: Check if there are open issues.",
                }
            ],
        },
        {"success": True, "errors": [], "metadata": {}},
    ])
    compiler = DummyCompiler([_compile_success(), _compile_success()])
    stub_source = "public final class PlanningToolStubs { public static void log(String msg) { } }"
    request = JavaPlanRequest(
        task="Close the first lint issue",
        goals=["goal"],
        tool_stub_source=stub_source,
        tool_stub_class_name="PlanningToolStubs",
    )
    plan_root = tmp_path / "plans"
    cached_hash = planner.compute_prompt_hash(request)
    _seed_cached_plan(
        plan_root,
        prompt_hash=cached_hash,
        plan_source=cached_plan,
        stub_source=stub_source,
        stub_class="PlanningToolStubs",
    )
    load_calls = []

    original_load = PlanCache.load

    def tracking_load(self, prompt_hash, *, stub_hash, stub_class_name):
        result = original_load(self, prompt_hash, stub_hash=stub_hash, stub_class_name=stub_class_name)
        load_calls.append(
            {
                "prompt_hash": prompt_hash,
                "stub_hash": stub_hash,
                "stub_class_name": stub_class_name,
                "hit": result is not None,
            }
        )
        return result

    monkeypatch.setattr(PlanCache, "load", tracking_load)

    orchestrator = PlanOrchestrator(
        planner,
        make_runner,
        max_retries=1,
        max_structure_attempts=5,
        plan_compiler=compiler,
        plan_artifact_root=plan_root,
        enable_plan_cache=True,
    )

    result = orchestrator.execute_with_retries(request)

    assert result["success"] is True
    assert len(planner.requests) == 1, "Helper-focus retry should invoke the planner once"
    assert runner.calls[0]["plan_source"].strip().startswith("public class Planner")
    assert "PlanningToolStubs.log(\"checking\")" in runner.calls[1]["plan_source"]
    assert len(load_calls) >= 2
    assert load_calls[0]["hit"] is True, "Top-level attempt should hit cache"
    helper_hash = result["final_plan"].prompt_hash
    assert any(call["prompt_hash"] == helper_hash for call in load_calls)
    helper_miss = [call for call in load_calls if call["prompt_hash"] == helper_hash][-1]
    assert helper_miss["hit"] is False, "Helper attempt should check cache and miss before replanning"
    helper_cache = PlanCache(plan_root)
    helper_entry = helper_cache.load(
        helper_hash,
        stub_hash=compute_stub_hash(stub_source),
        stub_class_name="PlanningToolStubs",
    )
    assert helper_entry is not None, "Successful helper plans should be cached"


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
        enable_plan_cache=False,
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
        enable_plan_cache=False,
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
        enable_plan_cache=False,
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
        enable_plan_cache=False,
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
    metadata_path = base_dir / "plan_metadata.json"
    clean_path = base_dir / "clean"
    classes_dir = base_dir / "classes"
    assert plan_path.exists()
    assert stub_path.exists()
    assert "PlanningToolStubs" in stub_path.read_text(encoding="utf-8")
    assert metadata_path.exists()
    assert clean_path.exists()
    assert classes_dir.exists()
    assert any(classes_dir.rglob("*.class"))


def test_cached_plan_skips_planner(tmp_path, runner_factory):
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
    artifact_root = tmp_path / "artifacts"
    orchestrator = PlanOrchestrator(
        planner,
        make_runner,
        plan_compiler=compiler,
        plan_artifact_root=artifact_root,
    )
    request = JavaPlanRequest(
        task="Cache plan",
        goals=["goal"],
        tool_stub_class_name="PlanningToolStubs",
        tool_stub_source="public final class PlanningToolStubs { public static void log(String m) {} }",
    )
    result = orchestrator.execute_with_retries(request)
    assert result["success"] is True

    cached_hash = result["final_plan"].prompt_hash

    class NeverPlanner:
        def __init__(self, prompt_hash: str):
            self.calls = 0
            self._hash = prompt_hash

        def generate_plan(self, request):  # noqa: ANN001 - test stub
            self.calls += 1
            raise AssertionError("Planner should not be invoked when cache exists")

        def compute_prompt_hash(self, request):  # noqa: ANN001 - test stub
            del request
            return self._hash

    cache_runner, make_cache_runner = runner_factory([
        {"success": True, "errors": [], "metadata": {}},
    ])
    never_planner = NeverPlanner(cached_hash)
    cache_compiler = DummyCompiler([_compile_success()])
    orchestrator_cached = PlanOrchestrator(
        never_planner,
        make_cache_runner,
        plan_compiler=cache_compiler,
        plan_artifact_root=artifact_root,
    )
    cached_request = JavaPlanRequest(
        task="Cache plan",
        goals=["goal"],
        tool_stub_class_name="PlanningToolStubs",
        tool_stub_source="public final class PlanningToolStubs { public static void log(String m) {} }",
    )

    cached_result = orchestrator_cached.execute_with_retries(cached_request)

    assert cached_result["success"] is True
    assert never_planner.calls == 0
    assert cache_compiler.calls == []
