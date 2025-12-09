from pathlib import Path
from typing import Any, Dict, List, Sequence

import pytest

from llmflow.planning import (
    CompilationError,
    JavaCompilationResult,
    JavaPlanFixer,
    PlanFixerRequest,
)
from llmflow.planning.java_plan_fixer import _ContextPatch


FIXED_PLAN = """
public class Planner {
    public static void main(String[] args) throws Exception {
        PlanningToolStubs.read_file("README.md", null);
    }
}
""".strip()


BROKEN_PLAN = """
public class Planner {
    public static void main(String[] args) throws Exception {
        System.out.println("TODO");
    }
}
""".strip()


def _make_error(line: int | None, *, column: int | None = 1, message: str | None = None) -> CompilationError:
    description = message or (f"error on line {line}" if line is not None else "error without line")
    return CompilationError(
        message=description,
        file="Planner.java",
        line=line,
        column=column,
    )



def _patch_text() -> str:
    return "\n".join(
        [
            "--- a/Planner.java",
            "+++ b/Planner.java",
            "@@ -2,5 +2,5 @@",
            " public class Planner {",
            "     public static void main(String[] args) throws Exception {",
            '-        System.out.println("TODO");',
            '+        PlanningToolStubs.read_file("README.md", null);',
            "     }",
            " }",
        ]
    )


def _multi_hunk_patch_text() -> str:
    return "\n".join(
        [
            "--- a/Planner.java",
            "+++ b/Planner.java",
            "@@ -2,5 +2,5 @@",
            " public class Planner {",
            "     public static void main(String[] args) throws Exception {",
            '-        System.out.println("TODO");',
            '+        PlanningToolStubs.read_file("README.md", null);',
            "     }",
            " }",
            "@@ -1,1 +1,2 @@",
            " public class Planner {",
            "+    // Added by test",
        ]
    )


def _leading_addition_patch_text() -> str:
    return "\n".join(
        [
            "--- a/Planner.java",
            "+++ b/Planner.java",
            "@@ -1,5 +1,7 @@",
            "-public class Planner {",
            "+import java.util.List;",
            "+",
            "+public class Planner {",
            "     public static void main(String[] args) throws Exception {",
            '         System.out.println("TODO");',
            "     }",
            " }",
        ]
    )


def _foreign_file_patch_text() -> str:
    return "\n".join(
        [
            "--- a/Other.java",
            "+++ b/Other.java",
            "@@ -1,1 +1,1 @@",
            "-public class Other {}",
            "+public class Other { void run() {} }",
        ]
    )


def _fenced_patch_text() -> str:
    return "\n".join([
        "```",
        _patch_text(),
        "```",
    ])


class PlainLLMClient:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.messages: List[Dict[str, Any]] = []
        self.calls = 0

    def generate(self, *, messages, **kwargs):
        self.messages = messages
        self.calls += 1
        return {"content": self.response_text}


class FlakyLLMClient:
    def __init__(self, response_text: str, side_effects):
        self.response_text = response_text
        self.side_effects = list(side_effects)
        self.messages: List[Dict[str, Any]] = []
        self.calls = 0

    def generate(self, *, messages, **kwargs):
        self.messages = messages
        self.calls += 1
        if self.side_effects:
            effect = self.side_effects.pop(0)
            if effect is not None:
                raise effect
        return {"content": self.response_text}


class DummyPlanCompiler:
    def __init__(self, results: Sequence[JavaCompilationResult]):
        self._results = list(results)
        self.calls: List[Dict[str, Any]] = []

    def compile(self, plan_source, **kwargs):
        if not self._results:
            raise AssertionError("No compilation results remaining")
        self.calls.append({"plan_source": plan_source, **kwargs})
        result = self._results.pop(0)
        working_dir = kwargs.get("working_dir")
        if working_dir is not None:
            work_path = Path(working_dir)
            work_path.mkdir(parents=True, exist_ok=True)
            if result.success:
                (work_path / "Planner.class").write_bytes(b"")
        return result


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
    client = PlainLLMClient(_patch_text())
    compiler = DummyPlanCompiler([fail_result, success_result])
    fixer = JavaPlanFixer(
        client,
        plan_compiler=compiler,
        artifact_root=tmp_path / "artifacts",
    )
    request = PlanFixerRequest(
        plan_id="plan-123",
        plan_source=BROKEN_PLAN,
        prompt_hash="abc123",
        task="Summarize README.md and email it to stakeholders.",
        compile_errors=compilation_errors,
        tool_stub_source="public final class PlanningToolStubs {}",
        tool_stub_class_name="PlanningToolStubs",
    )

    result = fixer.fix_plan(request, attempt=1)

    assert result.plan_source == FIXED_PLAN
    assert result.notes is None
    assert result.compile_result.success is True
    assert len(compiler.calls) == 2

    system_message = client.messages[0]["content"]
    assert "repair a single Java class" in system_message
    assert "localized contextual patches" in system_message
    assert "Emit plain-text unified diffs" in system_message
    assert "unified diff" in system_message
    assert "System Instruction: Strict Unified Diff Output Specification" in system_message
    assert "--- <old_filename>" in system_message
    assert "Strict Unified Diff Output Specification" in system_message
    user_message = client.messages[1]["content"]
    assert "User request" in user_message
    assert "Summarize README.md" in user_message
    assert "Compiler diagnostics" in user_message
    assert "line 3" in user_message
    assert "Tool stubs" in user_message
    assert "System Instruction: Strict Unified Diff Output Specification" in user_message
    assert "Follow the System Instruction" in user_message

    prompt_root = tmp_path / "artifacts" / "abc123"
    first_iter = prompt_root / "1.1"
    assert (first_iter / "errors.log").exists()
    patch_path = first_iter / "patch_response.txt"
    assert patch_path.exists()
    stored_patch = patch_path.read_text()
    assert "--- a/Planner.java" in stored_patch
    assert "+++ b/Planner.java" in stored_patch
    assert "@@ -2,5 +2,5 @@" in stored_patch
    pretty_patch = (first_iter / "patch.txt").read_text().strip()
    assert pretty_patch.startswith("===== PATCH 1 =====")
    assert "--- a/Planner.java" in pretty_patch
    assert "@@ -2,5 +2,5 @@" in pretty_patch
    assert "PlanningToolStubs" in pretty_patch
    assert (prompt_root / "1.2" / "clean").exists()


def test_plan_fixer_retries_after_validation_error(tmp_path, compilation_errors):
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
    client = FlakyLLMClient(
        _patch_text(),
        side_effects=[ValueError("invalid plan fragment"), None],
    )
    compiler = DummyPlanCompiler([fail_result, success_result])
    fixer = JavaPlanFixer(
        client,
        plan_compiler=compiler,
        artifact_root=tmp_path / "artifacts",
        max_retries=0,
    )
    request = PlanFixerRequest(
        plan_id="plan-123",
        plan_source=BROKEN_PLAN,
        prompt_hash="abc123",
        task="Summarize README.md and email it to stakeholders.",
        compile_errors=compilation_errors,
    )

    result = fixer.fix_plan(request, attempt=1)

    assert result.plan_source == FIXED_PLAN
    assert client.calls == 2


def test_plan_fixer_respects_custom_payload_retry_limit(tmp_path, compilation_errors):
    fail_result = JavaCompilationResult(
        success=False,
        command=("javac",),
        stderr="Planner.java:3: error",
        errors=list(compilation_errors),
    )
    client = FlakyLLMClient(
        _patch_text(),
        side_effects=[ValueError("invalid"), ValueError("still bad")],
    )
    compiler = DummyPlanCompiler([fail_result])
    fixer = JavaPlanFixer(
        client,
        plan_compiler=compiler,
        artifact_root=tmp_path / "artifacts",
        max_retries=0,
        fix_request_max_attempts=2,
    )
    request = PlanFixerRequest(
        plan_id="plan-123",
        plan_source=BROKEN_PLAN,
        prompt_hash="abc123",
        task="Summarize README.md and email it to stakeholders.",
        compile_errors=compilation_errors,
    )

    with pytest.raises(RuntimeError):
        fixer.fix_plan(request, attempt=1)

    assert client.calls == 2


def test_plan_fixer_resets_plan_source_on_duplicate_class_errors(tmp_path, compilation_errors):
    duplicate_error = CompilationError(
        message="duplicate class: Planner",
        file="Planner.java",
        line=1,
        column=1,
    )
    first_fail = JavaCompilationResult(
        success=False,
        command=("javac",),
        stderr="Planner.java:3: error",
        errors=list(compilation_errors),
    )
    duplicate_fail = JavaCompilationResult(
        success=False,
        command=("javac",),
        stderr="duplicate class",
        errors=[duplicate_error],
    )
    repeat_fail = JavaCompilationResult(
        success=False,
        command=("javac",),
        stderr="Planner.java:3: error",
        errors=list(compilation_errors),
    )
    success_result = JavaCompilationResult(
        success=True,
        command=("javac",),
        stdout="ok",
        stderr="",
        errors=[],
    )
    compiler = DummyPlanCompiler([first_fail, duplicate_fail, repeat_fail, success_result])
    client = PlainLLMClient(_patch_text())
    fixer = JavaPlanFixer(
        client,
        plan_compiler=compiler,
        artifact_root=tmp_path / "artifacts",
    )
    request = PlanFixerRequest(
        plan_id="plan-duplicate",
        plan_source=BROKEN_PLAN,
        prompt_hash="dup123",
        task="Summarize README.md.",
        compile_errors=compilation_errors,
    )

    result = fixer.fix_plan(request, attempt=1)

    assert result.plan_source == FIXED_PLAN
    assert result.compile_result.success is True
    assert result.notes is not None and "duplicate class" in result.notes.lower()
    assert client.calls == 2
    assert compiler.calls[1]["plan_source"] == FIXED_PLAN
    assert compiler.calls[2]["plan_source"] == BROKEN_PLAN


def test_plan_fixer_limits_compiler_errors_in_prompt():
    errors = [
        _make_error(40),
        _make_error(5),
        _make_error(12),
        _make_error(2),
    ]
    compile_result = JavaCompilationResult(
        success=False,
        command=("javac",),
        stdout="",
        stderr="",
        errors=errors,
    )
    fixer = JavaPlanFixer(PlainLLMClient(_patch_text()))
    request = PlanFixerRequest(
        plan_id="plan-123",
        plan_source=BROKEN_PLAN,
        prompt_hash="abc123",
        task="Summarize README.md.",
        compile_errors=errors,
    )

    prompt = fixer._build_user_prompt(BROKEN_PLAN, request, compile_result)

    assert "line 2" in prompt
    assert "line 5" in prompt
    assert "line 12" in prompt
    assert "line 40" not in prompt


def test_plan_fixer_respects_configurable_error_limit():
    errors = [
        _make_error(15),
        _make_error(3),
        _make_error(9),
    ]
    compile_result = JavaCompilationResult(
        success=False,
        command=("javac",),
        stdout="",
        stderr="",
        errors=errors,
    )
    fixer = JavaPlanFixer(PlainLLMClient(_patch_text()), max_prompt_errors=1)
    request = PlanFixerRequest(
        plan_id="plan-123",
        plan_source=BROKEN_PLAN,
        prompt_hash="abc123",
        task="Summarize README.md.",
        compile_errors=errors,
    )

    prompt = fixer._build_user_prompt(BROKEN_PLAN, request, compile_result)

    assert "line 3" in prompt
    assert "line 9" not in prompt
    assert "line 15" not in prompt


def test_diff_chunk_to_snippets_handles_multiple_hunks():
    fixer = JavaPlanFixer(PlainLLMClient(_patch_text()))
    snippets = fixer._diff_chunk_to_snippets(_multi_hunk_patch_text())
    assert len(snippets) == 2
    assert any("Added by test" in snippet.content for snippet in snippets)


def test_parse_patch_handles_leading_additions():
    fixer = JavaPlanFixer(PlainLLMClient(_leading_addition_patch_text()))
    patches, _ = fixer._parse_patch_text(
        _leading_addition_patch_text(),
        BROKEN_PLAN,
        allowed_filenames={"Planner.java"},
    )
    assert patches
    first_patch = patches[0]
    assert first_patch.before_context != ""
    assert "public class Planner" in first_patch.replacement_code


def test_parse_patch_skips_unexpected_files():
    fixer = JavaPlanFixer(PlainLLMClient(_foreign_file_patch_text()))
    skips: List[str] = []
    with pytest.raises(ValueError):
        fixer._parse_patch_text(
            _foreign_file_patch_text(),
            BROKEN_PLAN,
            allowed_filenames={"Planner.java"},
            skip_callback=skips.append,
        )
    assert skips
    assert any("not in allowed files" in reason for reason in skips)

def test_parse_patch_accepts_code_fence_wrappers():
    fixer = JavaPlanFixer(PlainLLMClient(_fenced_patch_text()))
    patches, _ = fixer._parse_patch_text(
        _fenced_patch_text(),
        BROKEN_PLAN,
        allowed_filenames={"Planner.java"},
    )
    assert patches


def test_parse_patch_extracts_notes_from_fenced_payload():
    fixer = JavaPlanFixer(PlainLLMClient(_fenced_patch_text()))
    response = "\n".join(
        [
            "```diff",
            _patch_text(),
            "<<<NOTES>>>",
            "Reverted duplicated block",
            "```",
        ]
    )
    patches, notes = fixer._parse_patch_text(
        response,
        BROKEN_PLAN,
        allowed_filenames={"Planner.java"},
    )
    assert patches
    assert notes == "Reverted duplicated block"


def test_apply_contextual_patches_reports_context_on_failure():
    patch = _ContextPatch(
        before_context="missing context\n",
        current_code="",
        after_context="unreachable\n",
        replacement_code="replacement\n",
    )
    with pytest.raises(ValueError) as excinfo:
        JavaPlanFixer._apply_contextual_patches(BROKEN_PLAN, [patch])
    assert "patch #1" in str(excinfo.value)


def test_apply_contextual_patches_uses_fuzzy_matching_for_file_header():
    patch = _ContextPatch(
        before_context="import java.util.List;\nimport java.util.Map;\npublic class Planner {\n",
        current_code="",
        after_context="    public static void main(String[] args) throws Exception {\n",
        replacement_code=(
            "import java.util.List;\n"
            "import java.util.Map;\n"
            "public class Planner {\n"
        ),
    )
    updated = JavaPlanFixer._apply_contextual_patches(BROKEN_PLAN, [patch])
    assert updated.startswith(
        "import java.util.List;\nimport java.util.Map;\npublic class Planner"
    )