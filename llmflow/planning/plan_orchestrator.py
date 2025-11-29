"""High-level orchestration with retry/repair loops for Java plans."""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Sequence

from llmflow.logging_utils import PLAN_LOGGER_NAME

from .java_planner import JavaPlanRequest, JavaPlanResult, JavaPlanner
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
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if max_error_hints < 1:
            raise ValueError("max_error_hints must be >= 1")
        self._planner = planner
        self._runner_factory = runner_factory
        self._max_retries = max_retries
        self._max_error_hints = max_error_hints
        self._plan_logger = logging.getLogger(PLAN_LOGGER_NAME)

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
        """Generate and execute a CPL plan with bounded retries.

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
            runner = self._runner_factory()
            execution_result = runner.execute(
                plan_result.plan_source,
                capture_trace=capture_trace,
                metadata=self._clone_dict(metadata),
                goal_summary=goal_summary,
                deferred_metadata=self._clone_dict(deferred_metadata),
                deferred_constraints=list(deferred_constraints) if deferred_constraints else None,
            )
            attempt_record = {
                "attempt": attempt_idx + 1,
                "plan": plan_result,
                "plan_id": plan_result.plan_id,
                "plan_metadata": dict(plan_result.metadata or {}),
                "execution": execution_result,
                "repair_hints": list(repair_hints),
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
            "cpl_plan attempt=%s plan_id=%s metadata=%s\n%s",
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
