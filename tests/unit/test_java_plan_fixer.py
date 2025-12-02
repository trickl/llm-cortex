from typing import Any, Dict, List, Sequence

import pytest

from llmflow.planning import (
    CompilationError,
    JavaCompilationResult,
    JavaPlanFixer,
    PlanFixerRequest,
)


FIXED_PLAN = """
public class Planner {
    public static void main(String[] args) throws Exception {
        PlanningToolStubs.read_file("README.md", null);
    }
}
""".strip()


class DummyLLMClient:
    def __init__(self, payload):
        self.payload = payload
        self.messages: List[Dict[str, Any]] = []
        self.kwargs: Dict[str, Any] = {}

    def structured_generate(self, *, messages, response_model, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return response_model(**self.payload)


class DummyPlanCompiler:
    def __init__(self, results: Sequence[JavaCompilationResult]):
        self._results = list(results)
        self.calls: List[Dict[str, Any]] = []

    def compile(self, plan_source, **kwargs):
        if not self._results:
            raise AssertionError("No compilation results remaining")
        self.calls.append({"plan_source": plan_source, **kwargs})
        return self._results.pop(0)


@pytest.fixture
def compilation_errors() -> List[CompilationError]:
    return [
        CompilationError(
            message="cannot find symbol Map",
            file="Planner.java",
            line=3,
            column=5,
        )
    ]


def test_plan_fixer_runs_compile_iterations_and_persists_artifacts(tmp_path, compilation_errors):
    fail_result = JavaCompilationResult(
        success=False,
        command=("javac",),
        stderr="Planner.java:3: error: cannot find symbol",
        errors=list(compilation_errors),
    )
    success_result = JavaCompilationResult(
        success=True,
        command=("javac",),
        stdout="ok",
        stderr="",
        errors=[],
    )
    client = DummyLLMClient({"java": FIXED_PLAN, "notes": "added imports"})
    compiler = DummyPlanCompiler([fail_result, success_result])
    fixer = JavaPlanFixer(
        client,
        plan_compiler=compiler,
        artifact_root=tmp_path / "artifacts",
    )
    request = PlanFixerRequest(
        plan_id="plan-123",
        plan_source="public class Planner { void main() {} }",
        prompt_hash="abc123",
        compile_errors=compilation_errors,
        tool_stub_source="public final class PlanningToolStubs {}",
        tool_stub_class_name="PlanningToolStubs",
    )

    result = fixer.fix_plan(request, attempt=1)

    assert result.plan_source == FIXED_PLAN
    assert result.notes == "added imports"
    assert result.compile_result.success is True
    assert len(compiler.calls) == 2

    system_message = client.messages[0]["content"]
    assert "repair a single Java class" in system_message
    user_message = client.messages[1]["content"]
    assert "Compiler diagnostics" in user_message
    assert "line 3" in user_message
    assert "Tool stubs" in user_message
    assert "Return the entire corrected class definition" in user_message

    prompt_root = tmp_path / "artifacts" / "abc123"
    assert (prompt_root / "1.1" / "errors.log").exists()
    assert (prompt_root / "1.2" / "clean").exists()