"""High-level execution harness for Java-based plans."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

from llmflow.runtime.errors import ToolError
from llmflow.runtime.syscall_registry import SyscallRegistry

from .deferred_planner import DeferredFunctionPrompt
from .runtime.ast import DeferredExecutionOptions
from .runtime.interpreter import (
    ExecutionTracer,
    PlanInterpreter,
    PlanRuntimeError,
)
from .runtime.parser import PlanParseError, parse_java_plan
from .runtime.validator import ValidationError


@dataclass
class ExecutionError:
    type: str
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    function: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "type": self.type,
            "message": self.message,
        }
        if self.line is not None:
            data["line"] = self.line
        if self.column is not None:
            data["column"] = self.column
        if self.function is not None:
            data["function"] = self.function
        return data


@dataclass
class PlanExecutionResult:
    success: bool
    return_value: Any = None
    errors: List[ExecutionError] = field(default_factory=list)
    trace: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "return_value": self.return_value,
            "errors": [err.to_dict() for err in self.errors],
            "trace": self.trace,
            "metadata": self.metadata,
        }


class PlanExecutor:
    """Convenience wrapper that parses, validates, and executes Java plans."""

    def __init__(
        self,
        registry: SyscallRegistry,
        *,
        deferred_planner: Optional[Callable[[DeferredFunctionPrompt], str]] = None,
        deferred_options: Optional[DeferredExecutionOptions] = None,
        specification: Optional[str] = None,
    ):
        self.registry = registry
        self.deferred_planner = deferred_planner
        self.deferred_options = deferred_options or DeferredExecutionOptions()
        self.specification = specification or ""

    def execute_from_string(
        self,
        source: str,
        *,
        capture_trace: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        result = self._execute(source, capture_trace=capture_trace, extra_metadata=metadata or {})
        return result.to_dict()

    def execute_from_file(
        self,
        path: str,
        *,
        capture_trace: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        return self.execute_from_string(source, capture_trace=capture_trace, metadata=metadata)

    def _execute(
        self,
        source: str,
        *,
        capture_trace: bool,
        extra_metadata: Dict[str, Any],
    ) -> PlanExecutionResult:
        tracer = ExecutionTracer(enabled=capture_trace)
        try:
            plan = parse_java_plan(source)
            interpreter = PlanInterpreter(
                plan,
                registry=self.registry,
                tracer=tracer,
                deferred_planner=self.deferred_planner,
                deferred_options=self.deferred_options,
                spec_text=self.specification,
            )
            return_value = interpreter.run()
            trace_payload = tracer.as_list() if capture_trace else None
            metadata = {
                "functions": len(plan.functions),
                "has_trace": capture_trace,
            }
            metadata.update(extra_metadata)
            return PlanExecutionResult(
                success=True,
                return_value=return_value,
                trace=trace_payload,
                metadata=metadata,
            )
        except ValidationError as exc:
            return self._error_result("validation_error", exc, tracer, capture_trace, extra_metadata)
        except PlanParseError as exc:
            return self._error_result("parse_error", exc, tracer, capture_trace, extra_metadata)
        except ToolError as exc:
            return self._error_result("tool_error", exc, tracer, capture_trace, extra_metadata)
        except PlanRuntimeError as exc:
            return self._error_result("runtime_error", exc, tracer, capture_trace, extra_metadata)
        except Exception as exc:  # pragma: no cover - safeguard
            return self._error_result("internal_error", exc, tracer, capture_trace, extra_metadata)

    def _error_result(
        self,
        error_type: str,
        exc: Exception,
        tracer: ExecutionTracer,
        capture_trace: bool,
        extra_metadata: Dict[str, Any],
    ) -> PlanExecutionResult:
        trace_payload = tracer.as_list() if capture_trace else None
        metadata = {
            "has_trace": capture_trace,
        }
        metadata.update(extra_metadata)
        errors = self._exception_to_errors(error_type, exc)
        return PlanExecutionResult(
            success=False,
            errors=errors,
            trace=trace_payload,
            metadata=metadata,
        )

    @staticmethod
    def _exception_to_errors(error_type: str, exc: Exception) -> List[ExecutionError]:
        if isinstance(exc, ValidationError):
            return [
                ExecutionError(
                    type=error_type,
                    message=issue.message,
                    line=issue.line,
                    column=issue.column,
                    function=issue.function,
                )
                for issue in exc.issues
            ]
        return [ExecutionError(type=error_type, message=str(exc))]


__all__ = [
    "ExecutionError",
    "PlanExecutionResult",
    "PlanExecutor",
]
