"""Static analysis entry points for Java plan programs."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import javalang

from .java_plan_analysis import JavaPlanGraph, analyze_java_plan
from .java_planner import JavaPlanningError


_EMBEDDED_SPEC = (
    "You coordinate goal-driven Java automation. Return exactly one top-level class named Planner "
    "that calls PlanningToolStubs helpers to perform every side effect. Prefer helper decomposition, "
    "avoid markdown, and emit compilable Java source."
)
_EMBEDDED_SPEC = (
    "You coordinate goal-driven Java automation. Return exactly one top-level class named Planner "
    "that calls PlanningToolStubs helpers to perform every side effect. Prefer helper decomposition, "
    "avoid markdown, and emit compilable Java source."
)

logger = logging.getLogger(__name__)


class PlanRunner:
    """Parse Java plans and emit workflow graphs."""

    def __init__(self, *, specification: Optional[str] = None) -> None:
        self._specification = (specification or self._load_specification()).strip()

    def execute(
        self,
        plan_source: str,
        *,
        capture_trace: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        goal_summary: Optional[str] = None,
        deferred_metadata: Optional[Dict[str, Any]] = None,
        deferred_constraints: Optional[Sequence[str]] = None,
        tool_stub_class_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        del capture_trace, goal_summary, deferred_metadata, deferred_constraints
        metadata_payload = dict(metadata or {})
        try:
            graph = analyze_java_plan(plan_source, tool_stub_class_name=tool_stub_class_name)
        except (javalang.parser.JavaSyntaxError, javalang.tokenizer.LexerError) as exc:
            error = _format_parse_error(exc)
            return {
                "success": False,
                "errors": [error],
                "graph": None,
                "metadata": metadata_payload,
                "trace": [],
            }
        except (ValueError, JavaPlanningError) as exc:
            return {
                "success": False,
                "errors": [
                    {
                        "type": "analysis_error",
                        "message": str(exc),
                    }
                ],
                "graph": None,
                "metadata": metadata_payload,
                "trace": [],
            }

        validation_errors = self._validate_graph(graph)
        function_names = [fn.name for fn in graph.functions]
        metadata_payload.setdefault("functions", len(function_names))
        metadata_payload["function_names"] = function_names
        metadata_payload["tool_call_count"] = sum(len(fn.tool_calls) for fn in graph.functions)
        return {
            "success": not validation_errors,
            "errors": validation_errors,
            "graph": graph.to_dict(),
            "metadata": metadata_payload,
            "trace": [],
        }

    def _validate_graph(self, graph: JavaPlanGraph) -> List[Dict[str, Any]]:
        errors: List[Dict[str, Any]] = []
        if not any(fn.name == "main" for fn in graph.functions):
            errors.append(
                {
                    "type": "validation_error",
                    "message": "Java plan must include a main() function.",
                    "function": None,
                }
            )
        return errors

    @staticmethod
    def _load_specification() -> str:
        logger.info("PlanRunner specification not provided; using embedded guidance.")
        return _EMBEDDED_SPEC


_LINE_PATTERN = re.compile(r"line\s+(?P<line>\d+)", re.IGNORECASE)
_COLUMN_PATTERN = re.compile(r"column\s+(?P<column>\d+)", re.IGNORECASE)


def _format_parse_error(error: Exception) -> Dict[str, Any]:
    line: Optional[int] = None
    column: Optional[int] = None
    position = getattr(error, "position", None)
    if position:
        line, column = position
    inferred_line, inferred_column = _infer_position_from_message(str(error))
    if line is None:
        line = inferred_line
    if column is None:
        column = inferred_column
    return {
        "type": "parse_error",
        "message": str(error),
        "line": line,
        "column": column,
    }


def _infer_position_from_message(message: str) -> Tuple[Optional[int], Optional[int]]:
    if not message:
        return None, None
    line = None
    column = None
    line_match = _LINE_PATTERN.search(message)
    if line_match:
        try:
            line = int(line_match.group("line"))
        except ValueError:
            line = None
    column_match = _COLUMN_PATTERN.search(message)
    if column_match:
        try:
            column = int(column_match.group("column"))
        except ValueError:
            column = None
    return line, column


__all__ = ["PlanRunner"]
