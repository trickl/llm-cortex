"""LLM-powered fixer that refines Java plans on disk."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from pydantic import BaseModel, Field, field_validator

from llmflow.logging_utils import PLAN_LOGGER_NAME

from .artifact_layout import ensure_prompt_artifact_dir, format_artifact_attempt_label
from .java_plan_compiler import (
    CompilationError,
    JavaCompilationResult,
    JavaPlanCompiler,
)
from .java_planner import _normalize_java_source  # type: ignore[attr-defined]


class _PlanFixPayload(BaseModel):
    java: str = Field(..., description="Updated Java plan source.")
    notes: Optional[str] = Field(
        default=None,
        description="Optional commentary about the applied fix.",
    )

    @field_validator("java")
    @classmethod
    def _normalize_java(cls, value: str) -> str:
        return _normalize_java_source(value)


@dataclass
class PlanFixerRequest:
    """Inputs required to repair a compiled Java plan."""

    plan_id: str
    plan_source: str
    prompt_hash: str
    compile_errors: Sequence[CompilationError] = field(default_factory=list)
    tool_stub_source: Optional[str] = None
    tool_stub_class_name: Optional[str] = None


@dataclass
class PlanFixerResult:
    """Outputs produced by :class:`JavaPlanFixer`."""

    plan_source: str
    compile_result: JavaCompilationResult
    notes: Optional[str] = None


class JavaPlanFixer:
    """Use the configured LLM to repair compiler errors in-place."""

    def __init__(
        self,
        llm_client,
        *,
        max_iterations: int = 10,
        max_retries: int = 1,
        artifact_root: Optional[Path] = None,
        plan_compiler: Optional[JavaPlanCompiler] = None,
    ) -> None:
        self._llm_client = llm_client
        self._max_iterations = max(1, max_iterations)
        self._max_retries = max(0, max_retries)
        self._artifact_root = Path(artifact_root or Path("plans"))
        self._plan_compiler = plan_compiler or JavaPlanCompiler()
        self._logger = PLAN_LOGGER_NAME

    def fix_plan(self, request: PlanFixerRequest, *, attempt: int) -> PlanFixerResult:
        prompt_dir = ensure_prompt_artifact_dir(
            self._artifact_root,
            request.prompt_hash,
            request.plan_id,
        )
        plan_source = request.plan_source
        notes: List[str] = []
        plan_logger = self._get_plan_logger()

        for iteration in range(1, self._max_iterations + 1):
            iteration_label = format_artifact_attempt_label(attempt, iteration)
            iteration_dir = prompt_dir / iteration_label
            compile_result = self._plan_compiler.compile(
                plan_source,
                tool_stub_source=request.tool_stub_source,
                tool_stub_class_name=request.tool_stub_class_name,
                working_dir=iteration_dir,
            )
            self._persist_compile_artifacts(iteration_dir, compile_result)
            error_count = len(compile_result.errors)
            plan_logger.info(
                "plan_fixer_compile plan_id=%s attempt=%s iteration=%s success=%s errors=%s dir=%s",
                request.plan_id,
                attempt,
                iteration,
                compile_result.success,
                error_count,
                iteration_dir,
            )
            if compile_result.success:
                return PlanFixerResult(
                    plan_source=plan_source,
                    compile_result=compile_result,
                    notes=self._merge_notes(notes),
                )
            if iteration == self._max_iterations:
                break
            plan_logger.info(
                "plan_fixer_llm_request plan_id=%s attempt=%s iteration=%s label=%s errors=%s",
                request.plan_id,
                attempt,
                iteration,
                iteration_label,
                error_count,
            )
            payload = self._request_fix(plan_source, request, compile_result)
            plan_source = payload.java.strip()
            if payload.notes:
                notes.append(payload.notes.strip())

        return PlanFixerResult(
            plan_source=plan_source,
            compile_result=compile_result,
            notes=self._merge_notes(notes),
        )

    def _request_fix(
        self,
        plan_source: str,
        request: PlanFixerRequest,
        compile_result: JavaCompilationResult,
    ) -> _PlanFixPayload:
        messages = self._build_messages(plan_source, request, compile_result)
        return self._llm_client.structured_generate(
            messages=messages,
            response_model=_PlanFixPayload,
            max_retries=self._max_retries,
        )

    def _build_messages(
        self,
        plan_source: str,
        request: PlanFixerRequest,
        compile_result: JavaCompilationResult,
    ) -> List[dict]:
        system_prompt = self._build_system_prompt(request)
        user_prompt = self._build_user_prompt(plan_source, request, compile_result)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _build_system_prompt(self, request: PlanFixerRequest) -> str:
        if request.tool_stub_source:
            stub_clause = "Use only the provided planning tool stubs; never invent additional helper classes."
        else:
            stub_clause = "The plan already references PlanningToolStubs; keep using the same static helpers."
        lines = [
            "You repair a single Java class that orchestrates planning tools.",
            "Given the previous plan and the Java compiler diagnostics, produce a corrected plan that compiles without introducing new context.",
            "Preserve the existing structure and helper functions whenever possible.",
            stub_clause,
            "Respond ONLY with Java source (no markdown).",
        ]
        return " ".join(line.strip() for line in lines if line.strip()).strip()

    def _build_user_prompt(
        self,
        plan_source: str,
        request: PlanFixerRequest,
        compile_result: JavaCompilationResult,
    ) -> str:
        plan_block = textwrap.dedent(
            f"""
            Previous plan:
            ```java
            {plan_source.strip()}
            ```
            """
        ).strip()
        errors_block = self._format_errors(compile_result.errors)
        sections = [plan_block, f"Compiler diagnostics:\n{errors_block}"]
        if request.tool_stub_source:
            sections.append(
                textwrap.dedent(
                    f"""
                    Tool stubs:
                    ```java
                    {request.tool_stub_source.strip()}
                    ```
                    """
                ).strip()
            )
        else:
            sections.append(
                "Existing plan already imports PlanningToolStubs. Continue to use the same static helpers."
            )
        sections.append(
            "Requirements:\n- Keep the class self-contained.\n- Only adjust code necessary to resolve the diagnostics.\n- Maintain all PlanningToolStubs.<name>(...) calls."
        )
        sections.append("Return the entire corrected class definition.")
        return "\n\n".join(sections).strip()

    @staticmethod
    def _format_errors(errors: Sequence[CompilationError]) -> str:
        if not errors:
            return "(Compiler produced no structured diagnostics; ensure the plan compiles.)"
        lines: List[str] = []
        for idx, error in enumerate(errors, start=1):
            location_parts: List[str] = []
            if error.file:
                location_parts.append(error.file)
            if error.line is not None:
                loc = f"line {error.line}"
                if error.column is not None:
                    loc += f", column {error.column}"
                location_parts.append(loc)
            location = f" ({'; '.join(location_parts)})" if location_parts else ""
            lines.append(f"{idx}. {error.message}{location}")
            if error.raw:
                lines.append(textwrap.indent(error.raw, "   "))
        return "\n".join(lines).strip()

    @staticmethod
    def _merge_notes(notes: Sequence[str]) -> Optional[str]:
        filtered = [note for note in (note.strip() for note in notes) if note]
        if not filtered:
            return None
        return "\n\n".join(filtered)

    def _persist_compile_artifacts(
        self,
        iteration_dir: Path,
        compile_result: JavaCompilationResult,
    ) -> None:
        iteration_dir.mkdir(parents=True, exist_ok=True)
        clean_path = iteration_dir / "clean"
        errors_path = iteration_dir / "errors.log"
        if compile_result.success:
            clean_path.touch()
            if errors_path.exists():
                errors_path.unlink()
            return
        if clean_path.exists():
            clean_path.unlink()
        stderr = compile_result.stderr.strip() if compile_result.stderr else ""
        if not stderr:
            stderr = "\n".join(error.message for error in compile_result.errors)
        errors_path.write_text(stderr or "Unknown compilation failure", encoding="utf-8")

    def _get_plan_logger(self):
        import logging

        return logging.getLogger(self._logger)

    @property
    def artifact_root(self) -> Path:
        """Return the root directory used for plan artifacts."""

        return self._artifact_root


__all__ = ["JavaPlanFixer", "PlanFixerRequest", "PlanFixerResult"]
