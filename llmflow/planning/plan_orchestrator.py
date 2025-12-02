"""High-level orchestration with retry/repair loops for Java plans."""
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from llmflow.logging_utils import (
    PLAN_LOGGER_NAME,
    format_execution_failure_reason,
    log_plan_failure_summary,
)

from .artifact_layout import (
    ensure_prompt_artifact_dir,
    format_artifact_attempt_label,
    reset_prompt_artifact_dir,
)
from .java_plan_compiler import JavaCompilationResult, JavaPlanCompiler
from .java_plan_fixer import JavaPlanFixer, PlanFixerRequest, PlanFixerResult
from .java_planner import JavaPlanRequest, JavaPlanResult, JavaPlanner, _compute_prompt_hash
from .plan_runner import PlanRunner


class PlanOrchestrator:
    """Coordinate plan generation, execution, and targeted retries."""

    def __init__(
        self,
        planner: JavaPlanner,
        runner_factory: Callable[[], PlanRunner],
        *,
        max_retries: int = 1,
        max_error_hints: int = 3,
        max_compile_refinements: int = 3,
        plan_compiler: Optional[JavaPlanCompiler] = None,
        plan_fixer: Optional[JavaPlanFixer] = None,
        plan_fixer_max_attempts: int = 3,
        plan_artifact_root: Optional[Path] = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if max_error_hints < 1:
            raise ValueError("max_error_hints must be >= 1")
        if max_compile_refinements < 1:
            raise ValueError("max_compile_refinements must be >= 1")
        self._planner = planner
        self._runner_factory = runner_factory
        self._max_retries = max_retries
        self._max_error_hints = max_error_hints
        self._max_compile_refinements = max_compile_refinements
        self._plan_compiler = plan_compiler or JavaPlanCompiler()
        self._plan_fixer = plan_fixer
        self._plan_fixer_max_attempts = max(0, plan_fixer_max_attempts)
        self._plan_logger = logging.getLogger(PLAN_LOGGER_NAME)
        if plan_artifact_root is not None:
            self._plan_artifact_root = Path(plan_artifact_root)
        elif plan_fixer is not None and hasattr(plan_fixer, "artifact_root"):
            self._plan_artifact_root = Path(plan_fixer.artifact_root)
        else:
            self._plan_artifact_root = Path("plans")

    def execute_with_retries(
        self,
        request: JavaPlanRequest,
        *,
        capture_trace: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        goal_summary: Optional[str] = None,
        deferred_metadata: Optional[Dict[str, Any]] = None,
        deferred_constraints: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Generate and execute a Java plan with bounded retries.

        Returns a dict with the final execution payload plus telemetry describing
        each attempt. The ``success`` key reflects the outcome of the last
        execution.
        """

        attempts: List[Dict[str, Any]] = []
        repair_hints: List[str] = []

        for attempt_idx in range(self._max_retries + 1):
            effective_request = self._augment_request(request, attempt_idx, repair_hints)
            plan_result = self._planner.generate_plan(effective_request)
            self._log_plan_attempt(attempt_idx + 1, plan_result)
            plan_result, compile_attempts = self._compile_with_refinement(
                effective_request,
                plan_result,
                attempt_number=attempt_idx + 1,
            )
            compile_success = not compile_attempts or compile_attempts[-1]["success"]
            if compile_success:
                runner = self._runner_factory()
                execution_result = runner.execute(
                    plan_result.plan_source,
                    capture_trace=capture_trace,
                    metadata=self._clone_dict(metadata),
                    goal_summary=goal_summary,
                    deferred_metadata=self._clone_dict(deferred_metadata),
                    deferred_constraints=list(deferred_constraints) if deferred_constraints else None,
                    tool_stub_class_name=effective_request.tool_stub_class_name,
                )
            else:
                execution_result = self._build_compile_failure_payload(compile_attempts[-1])
            attempt_record = {
                "attempt": attempt_idx + 1,
                "plan": plan_result,
                "plan_id": plan_result.plan_id,
                "plan_metadata": dict(plan_result.metadata or {}),
                "execution": execution_result,
                "repair_hints": list(repair_hints),
                "compile_attempts": compile_attempts,
            }
            attempt_record["summary"] = self._summarize_attempt(attempt_record)
            attempts.append(attempt_record)
            if execution_result.get("success"):
                telemetry = self._build_telemetry(attempts)
                return {
                    "success": True,
                    "final_plan": plan_result,
                    "final_execution": execution_result,
                    "attempts": attempts,
                    "telemetry": telemetry,
                    "summary": self._format_summary(telemetry),
                }
            failure_reason = format_execution_failure_reason(execution_result)
            log_plan_failure_summary(
                attempt_number=attempt_idx + 1,
                plan_id=plan_result.plan_id,
                reason=failure_reason,
                metadata={
                    "compile_success": compile_success,
                    "repair_hint_count": len(repair_hints),
                    "plan_metadata": dict(plan_result.metadata or {}),
                },
            )
            if attempt_idx == self._max_retries:
                break
            repair_hints = self._build_repair_hints(execution_result)

        telemetry = self._build_telemetry(attempts)
        return {
            "success": False,
            "final_plan": attempts[-1]["plan"] if attempts else None,
            "final_execution": attempts[-1]["execution"] if attempts else None,
            "attempts": attempts,
            "telemetry": telemetry,
            "summary": self._format_summary(telemetry),
        }

    def _log_plan_attempt(self, attempt_number: int, plan_result: JavaPlanResult) -> None:
        if not plan_result or not plan_result.plan_source:
            return
        self._plan_logger.info(
            "java_plan attempt=%s plan_id=%s metadata=%s\n%s",
            attempt_number,
            plan_result.plan_id,
            plan_result.metadata,
            plan_result.plan_source,
        )

    def _augment_request(
        self,
        original: JavaPlanRequest,
        attempt_idx: int,
        repair_hints: Sequence[str],
    ) -> JavaPlanRequest:
        if attempt_idx == 0 and not repair_hints:
            return original

        merged_constraints = list(original.additional_constraints or [])
        merged_constraints.extend(repair_hints)

        metadata = dict(original.metadata or {})
        metadata["attempt_index"] = attempt_idx + 1
        if repair_hints:
            metadata["repair_hints"] = list(repair_hints)

        return replace(
            original,
            additional_constraints=merged_constraints,
            metadata=metadata,
        )

    def _compile_with_refinement(
        self,
        base_request: JavaPlanRequest,
        initial_plan: JavaPlanResult,
        *,
        attempt_number: int,
    ) -> Tuple[JavaPlanResult, List[Dict[str, Any]]]:
        plan = initial_plan
        attempts: List[Dict[str, Any]] = []
        fixer_attempts = 0
        previous_error_count: Optional[int] = None
        self._persist_plan_inputs(plan, base_request, attempt_number)

        for iteration in range(1, self._max_compile_refinements + 1):
            compile_result = self._plan_compiler.compile(
                plan.plan_source,
                tool_stub_source=base_request.tool_stub_source,
                tool_stub_class_name=base_request.tool_stub_class_name,
            )
            attempt_payload = self._format_compile_attempt(compile_result, iteration)
            attempts.append(attempt_payload)
            error_count = len(attempt_payload.get("errors") or [])
            if compile_result.success:
                return plan, attempts

            can_fix = (
                self._plan_fixer is not None
                and fixer_attempts < self._plan_fixer_max_attempts
                and (previous_error_count is None or error_count <= previous_error_count)
            )
            if can_fix:
                fixer_attempts += 1
                previous_error_count = error_count
                prompt_hash = (
                    plan.prompt_hash
                    or plan.metadata.get("prompt_hash")
                    or (
                        _compute_prompt_hash(plan.prompt_messages)
                        if plan.prompt_messages
                        else plan.plan_id
                    )
                )
                fix_request = PlanFixerRequest(
                    plan_id=plan.plan_id,
                    plan_source=plan.plan_source,
                    prompt_hash=prompt_hash,
                    compile_errors=compile_result.errors,
                    tool_stub_source=base_request.tool_stub_source,
                    tool_stub_class_name=base_request.tool_stub_class_name,
                )
                fix_result = self._plan_fixer.fix_plan(fix_request, attempt=fixer_attempts)
                plan = self._apply_plan_fix(plan, fix_result)
                self._persist_plan_inputs(plan, base_request, attempt_number)
                continue

            break

        return plan, attempts

    @staticmethod
    def _format_location_fragment(error: Any) -> str:
        parts: List[str] = []
        file_name = PlanOrchestrator._get_error_attr(error, "file")
        line = PlanOrchestrator._get_error_attr(error, "line")
        column = PlanOrchestrator._get_error_attr(error, "column")
        if file_name:
            parts.append(str(file_name))
        if line is not None:
            segment = f"line {line}"
            if column is not None:
                segment += f", column {column}"
            parts.append(segment)
        if not parts:
            return ""
        return " (" + "; ".join(parts) + ")"

    @staticmethod
    def _extract_error_message(error: Any) -> str:
        message = PlanOrchestrator._get_error_attr(error, "message")
        if not message and isinstance(error, dict):
            message = error.get("message")
        return str(message or "Unknown compilation error.")

    @staticmethod
    def _get_error_attr(error: Any, name: str) -> Any:
        if isinstance(error, dict):
            return error.get(name)
        return getattr(error, name, None)

    def _format_compile_attempt(
        self,
        compile_result: JavaCompilationResult,
        iteration: int,
    ) -> Dict[str, Any]:
        return {
            "iteration": iteration,
            "success": compile_result.success,
            "errors": self._convert_compile_errors(compile_result.errors),
            "stderr": compile_result.stderr,
            "stdout": compile_result.stdout,
            "command": list(compile_result.command),
        }

    def _convert_compile_errors(self, errors: Sequence[Any]) -> List[Dict[str, Any]]:
        converted: List[Dict[str, Any]] = []
        for error in errors:
            if isinstance(error, dict):
                payload = dict(error)
            else:
                payload = {
                    "file": getattr(error, "file", None),
                    "line": getattr(error, "line", None),
                    "column": getattr(error, "column", None),
                    "message": getattr(error, "message", None),
                }
            payload.setdefault("type", "compile_error")
            converted.append(payload)
        return converted

    @staticmethod
    def _build_compile_failure_payload(attempt_summary: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": False,
            "errors": list(attempt_summary.get("errors") or []),
            "metadata": {"stage": "compile"},
            "trace": [],
        }

    @staticmethod
    def _apply_plan_fix(plan: JavaPlanResult, fix_result: PlanFixerResult) -> JavaPlanResult:
        plan.plan_source = fix_result.plan_source
        if fix_result.notes:
            notes = plan.metadata.setdefault("plan_fixer_notes", [])
            notes.append(fix_result.notes)
        return plan

    def _build_repair_hints(self, execution_result: Dict[str, Any]) -> List[str]:
        errors = execution_result.get("errors") or []
        hints: List[str] = []
        for error in errors[: self._max_error_hints]:
            hint = self._format_error_hint(error)
            if hint:
                hints.append(hint)
        if not hints:
            hints.append(
                "Previous attempt failed without structured errors. Ensure the plan parses and all referenced functions exist."
            )
        return hints

    def _persist_plan_inputs(
        self,
        plan: JavaPlanResult,
        request: JavaPlanRequest,
        attempt_number: int,
    ) -> None:
        prompt_hash = (
            plan.prompt_hash
            or plan.metadata.get("prompt_hash")
            or (_compute_prompt_hash(plan.prompt_messages) if plan.prompt_messages else None)
            or plan.plan_id
        )
        if attempt_number == 1:
            prompt_dir = reset_prompt_artifact_dir(self._plan_artifact_root, prompt_hash, plan.plan_id)
        else:
            try:
                prompt_dir = ensure_prompt_artifact_dir(
                    self._plan_artifact_root,
                    prompt_hash,
                    plan.plan_id,
                )
            except RuntimeError:
                prompt_dir = reset_prompt_artifact_dir(
                    self._plan_artifact_root,
                    prompt_hash,
                    plan.plan_id,
                )
        attempt_label = format_artifact_attempt_label(attempt_number)
        base_dir = prompt_dir / attempt_label
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            plan_path = base_dir / "Plan.java"
            plan_path.write_text(plan.plan_source.strip() + "\n", encoding="utf-8")
            if request.tool_stub_source:
                stub_class = request.tool_stub_class_name or "PlanningToolStubs"
                stub_path = base_dir / f"{stub_class}.java"
                stub_path.write_text(request.tool_stub_source.strip() + "\n", encoding="utf-8")
        except OSError as exc:  # pragma: no cover - filesystem issues are logged
            self._plan_logger.warning(
                "plan_artifact_write_failed plan_id=%s attempt=%s error=%s",
                plan.plan_id,
                attempt_number,
                exc,
            )

    @staticmethod
    def _format_error_hint(error: Dict[str, Any]) -> Optional[str]:
        if not isinstance(error, dict):
            return None
        err_type = error.get("type") or "unknown_error"
        message = error.get("message") or "No diagnostic message provided."
        function = error.get("function")
        location_parts: List[str] = []
        if function:
            location_parts.append(f"function {function}")
        line = error.get("line")
        column = error.get("column")
        if line is not None:
            location = f"line {line}"
            if column is not None:
                location += f", column {column}"
            location_parts.append(location)
        location_suffix = f" ({'; '.join(location_parts)})" if location_parts else ""
        return f"Repair hint: Address {err_type} - {message}{location_suffix}."

    def _summarize_attempt(self, attempt_record: Dict[str, Any]) -> Dict[str, Any]:
        execution = attempt_record["execution"]
        metadata = execution.get("metadata") or {}
        errors = execution.get("errors") or []
        first_error = errors[0] if errors else None
        tool_usage = self._extract_tool_usage(execution.get("trace"))
        trace_excerpt = self._trim_trace(execution.get("trace"))
        return {
            "attempt": attempt_record["attempt"],
            "plan_id": attempt_record.get("plan_id"),
            "status": "success" if execution.get("success") else "failure",
            "functions": metadata.get("functions"),
            "errors": errors,
            "error_message": (first_error or {}).get("message"),
            "error_type": (first_error or {}).get("type"),
            "tool_usage": tool_usage,
            "repair_hints": attempt_record.get("repair_hints", []),
            "plan_metadata": attempt_record.get("plan_metadata") or {},
            "trace_excerpt": trace_excerpt,
        }

    def _build_telemetry(self, attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
        attempt_summaries = [record.get("summary", {}) for record in attempts]
        aggregate_usage: Dict[str, int] = {}
        for summary in attempt_summaries:
            for name, count in summary.get("tool_usage", {}).items():
                aggregate_usage[name] = aggregate_usage.get(name, 0) + count
        final_summary = attempt_summaries[-1] if attempt_summaries else {}
        return {
            "attempt_count": len(attempt_summaries),
            "attempt_summaries": attempt_summaries,
            "tool_usage": aggregate_usage,
            "trace_excerpt": final_summary.get("trace_excerpt", []),
            "success": bool(final_summary.get("status") == "success"),
        }

    @staticmethod
    def _extract_tool_usage(trace: Optional[List[Dict[str, Any]]]) -> Dict[str, int]:
        usage: Dict[str, int] = {}
        if not isinstance(trace, list):
            return usage
        for event in trace:
            if not isinstance(event, dict):
                continue
            if event.get("type") != "syscall_start":
                continue
            name = event.get("name")
            if not name:
                continue
            usage[name] = usage.get(name, 0) + 1
        return usage

    @staticmethod
    def _trim_trace(trace: Optional[List[Dict[str, Any]]], limit: int = 20) -> List[Dict[str, Any]]:
        if not isinstance(trace, list):
            return []
        return trace[:limit]

    @staticmethod
    def _format_summary(telemetry: Dict[str, Any]) -> str:
        attempt_count = telemetry.get("attempt_count", 0)
        success = telemetry.get("success", False)
        status_symbol = "✅" if success else "❌"
        lines = [f"{status_symbol} Java plan run – {attempt_count} attempt(s)"]
        for summary in telemetry.get("attempt_summaries", []):
            status_icon = "✅" if summary.get("status") == "success" else "❌"
            attempt_num = summary.get("attempt")
            functions = summary.get("functions")
            error_message = summary.get("error_message") or "none"
            tool_usage = summary.get("tool_usage", {})
            tool_text = ", ".join(f"{name}×{count}" for name, count in tool_usage.items()) or "none"
            lines.append(
                f"  - Attempt {attempt_num}: {status_icon} {summary.get('status')} | functions={functions} | errors={error_message}"
            )
            lines.append(f"    Tool usage: {tool_text}")
        if telemetry.get("tool_usage"):
            aggregate_text = ", ".join(
                f"{name}×{count}" for name, count in sorted(telemetry["tool_usage"].items())
            )
            lines.append(f"Aggregate tool usage: {aggregate_text}")
        return "\n".join(lines)

    @staticmethod
    def _clone_dict(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if payload is None:
            return None
        return dict(payload)


__all__ = ["PlanOrchestrator"]
