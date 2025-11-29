"""Utilities for prompting the LLM to emit Java plans."""
from __future__ import annotations

import json
import logging
import re
import textwrap
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, field_validator

try:  # pragma: no cover - guard against optional dependency changes
    from instructor.core.exceptions import InstructorRetryException
except ImportError:  # pragma: no cover
    InstructorRetryException = None  # type: ignore[assignment]

from llmflow.llm_client import LLMClient

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SPEC_PATH = _PROJECT_ROOT / "planning" / "java_planning.md"
_PLANNER_TOOL_NAME = "define_java_plan"

logger = logging.getLogger(__name__)

_CLASS_DECL_PATTERN = re.compile(r"^\s*(?:public\s+)?class\s+\w+", re.MULTILINE)
_FENCE_BLOCK_PATTERN = re.compile(
    r"^```[a-zA-Z0-9_-]*\s*\n(?P<body>[\s\S]*?)\n```$",
    re.MULTILINE,
)


def _strip_markdown_fences(source: str) -> str:
    """Remove a single leading/trailing markdown fence pair if present."""

    text = source.strip()
    match = _FENCE_BLOCK_PATTERN.match(text)
    if match:
        return match.group("body").strip()
    return text


def _normalize_java_source(source: str) -> str:
    """Normalize Java source and ensure it declares a top-level class."""

    stripped = _strip_markdown_fences(source)
    if not _CLASS_DECL_PATTERN.search(stripped):
        raise ValueError("Java payload must declare a top-level class.")
    return stripped.strip()


def _extract_tool_call_count(exc: Exception) -> Optional[int]:
    """Best-effort extraction of tool call count from Instructor retries."""

    if InstructorRetryException is None:
        return None
    if not isinstance(exc, InstructorRetryException):
        return None
    attempts = getattr(exc, "failed_attempts", None)
    if not attempts:
        return None
    for attempt in attempts:
        completion = getattr(attempt, "completion", None)
        if not completion:
            continue
        choices = getattr(completion, "choices", None)
        if not choices:
            continue
        message = getattr(choices[0], "message", None)
        if not message:
            continue
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls is None:
            return 0
        return len(tool_calls)
    return None


def _summarize_structured_failure(exc: Exception) -> Optional[str]:
    """Return a user-friendly explanation for structured generation failures."""

    count = _extract_tool_call_count(exc)
    if count == 0:
        return "Structured Java plan request produced no tool calls."
    if count and count != 1:
        return f"Structured Java plan request produced {count} tool calls; expected exactly one."

    message = str(exc)
    if "Instructor does not support multiple tool calls" in message:
        return "Structured Java plan request did not yield exactly one tool call."
    if message:
        return f"Structured Java plan request failed: {message}"
    return None


class JavaPlanningError(RuntimeError):
    """Raised when Java plan synthesis fails."""


@dataclass
class JavaPlanRequest:
    """Inputs that describe what the planner should generate."""

    task: str
    goals: Sequence[str] = field(default_factory=list)
    context: Optional[str] = None
    allowed_syscalls: Sequence[str] = field(default_factory=list)
    additional_constraints: Sequence[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    include_deferred_guidance: bool = True


@dataclass
class JavaPlanResult:
    """Structured result returned by :class:`JavaPlanner`."""

    plan_id: str
    plan_source: str
    raw_response: Dict[str, Any]
    prompt_messages: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class _PlannerToolPayload(BaseModel):
    notes: Optional[str] = Field(
        default=None,
        description="Optional commentary about assumptions, risks, or TODOs.",
    )
    java: str = Field(
        ..., description="Complete Java source code containing exactly one top-level class."
    )

    @field_validator("java")
    @classmethod
    def _ensure_plan_block(cls, value: str) -> str:
        try:
            return _normalize_java_source(value)
        except ValueError as exc:  # pragma: no cover - validated downstream
            raise ValueError(str(exc)) from exc

    @field_validator("notes", mode="before")
    @classmethod
    def _normalize_notes(cls, value: Optional[Any]) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, list):
            normalized = [str(item).strip() for item in value if str(item).strip()]
            if not normalized:
                return None
            return "\n\n".join(normalized)
        return str(value)


class JavaPlanner:
    """High-level helper that asks the LLM to emit a Java plan."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        specification: Optional[str] = None,
        specification_path: Optional[Path] = None,
    ):
        self._llm_client = llm_client
        self._specification = self._load_specification(specification, specification_path)
        self._planner_tool_schema = self._build_planner_tool_schema()
        self._planner_tool_choice = {
            "type": "function",
            "function": {"name": _PLANNER_TOOL_NAME},
        }

    def generate_plan(self, request: JavaPlanRequest) -> JavaPlanResult:
        messages = self._build_messages(request)
        plan_source: str
        raw_response: Dict[str, Any]
        notes: Optional[str] = None
        try:
            payload = self._llm_client.structured_generate(
                messages=messages,
                response_model=_PlannerToolPayload,
                tools=[self._planner_tool_schema],
                tool_choice=self._planner_tool_choice,
            )
        except Exception as exc:  # pragma: no cover - provider dependent
            friendly = _summarize_structured_failure(exc)
            if friendly:
                logger.warning("%s Falling back to plain-text parsing.", friendly)
            else:
                logger.warning(
                    "Structured Java plan generation failed; attempting plain-text fallback.",
                    exc_info=exc,
                )
            plan_source, raw_response, notes = self._generate_plain_plan(messages)
        else:
            plan_source = payload.java.strip()
            raw_response = payload.model_dump()
            if payload.notes:
                notes = payload.notes.strip()

        plan_id = str(request.metadata.get("plan_id") or uuid.uuid4())
        normalized_syscalls = self._normalize_allowed_syscalls(request.allowed_syscalls)
        metadata = dict(request.metadata)
        metadata.setdefault("allowed_syscalls", normalized_syscalls)
        if notes:
            metadata["planner_notes"] = notes
        return JavaPlanResult(
            plan_id=plan_id,
            plan_source=plan_source,
            raw_response=raw_response,
            prompt_messages=messages,
            metadata=metadata,
        )

    def _generate_plain_plan(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any], Optional[str]]:
        """Fallback path when structured tool calls are unavailable."""

        response = self._llm_client.generate(
            messages=messages,
            tools=[self._planner_tool_schema],
            tool_choice=self._planner_tool_choice,
        )
        if not isinstance(response, dict):
            raise JavaPlanningError("Planner returned an unexpected response format.")

        notes: Optional[str] = None
        tool_calls = response.get("tool_calls") or []
        if len(tool_calls) == 1:
            call = tool_calls[0]
            function_meta = call.get("function", {})
            if function_meta.get("name") == _PLANNER_TOOL_NAME:
                arguments = function_meta.get("arguments") or "{}"
                try:
                    payload_data = json.loads(arguments)
                except (TypeError, json.JSONDecodeError) as exc:
                    raise JavaPlanningError(
                        "Planner tool call arguments were invalid JSON."
                    ) from exc
                java_source = payload_data.get("java")
                if not java_source:
                    raise JavaPlanningError("Planner tool call did not include Java source.")
                try:
                    normalized = _normalize_java_source(java_source)
                except ValueError as exc:
                    raise JavaPlanningError(f"Planner returned invalid Java: {exc}") from exc
                raw_notes = payload_data.get("notes")
                if raw_notes is not None:
                    notes = _PlannerToolPayload._normalize_notes(raw_notes)
                return normalized, response, notes

        content = response.get("content")
        if content is None or not str(content).strip():
            raise JavaPlanningError("Planner returned empty content.")
        try:
            normalized = _normalize_java_source(str(content))
        except ValueError as exc:
            raise JavaPlanningError(f"Planner returned invalid Java: {exc}") from exc
        return normalized, response, None

    def _build_messages(self, request: JavaPlanRequest) -> List[Dict[str, Any]]:
        normalized_syscalls = self._normalize_allowed_syscalls(request.allowed_syscalls)
        constraint_lines = self._build_constraints(request, normalized_syscalls)
        system_content = self._build_system_message()
        user_content = self._build_user_message(request, normalized_syscalls, constraint_lines)
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _build_system_message(self) -> str:
        header = (
            "You are the Java plan synthesizer."
            " Produce a single Java class that fully solves the user's task."
            " Refer to the define_java_plan tool description for the full specification,"
            " calling that tool exactly once when your model supports tool calls."
            " If tools are unavailable, respond with the raw Java source only and do not"
            " emit explanations."
        )
        tools_block = json.dumps(
            {
                "available_tools": [self._planner_tool_schema["function"]],
                "instructions": (
                    "Call define_java_plan exactly once when possible; otherwise return the"
                    " Java source as plain assistant text."
                ),
            },
            indent=2,
        )
        return f"{header}\n\n<available_tools>\n{tools_block}\n</available_tools>".strip()

    def _build_user_message(
        self,
        request: JavaPlanRequest,
        normalized_syscalls: Sequence[str],
        constraint_lines: Sequence[str],
    ) -> str:
        lines: List[str] = []
        lines.append("Task:")
        lines.append(textwrap.dedent(request.task).strip())
        lines.append("")

        if request.goals:
            lines.append("Goals:")
            for idx, goal in enumerate(request.goals, start=1):
                lines.append(f"{idx}. {goal}")
            lines.append("")

        if request.context:
            lines.append("Context:")
            lines.append(textwrap.dedent(request.context).strip())
            lines.append("")

        lines.append("Allowed syscalls:")
        if normalized_syscalls:
            for name in normalized_syscalls:
                lines.append(f"- {name}")
        else:
            lines.append("- (none registered)")
        lines.append("")

        lines.append("Constraints:")
        for rule in constraint_lines:
            lines.append(f"- {rule}")
        if request.additional_constraints:
            for rule in request.additional_constraints:
                lines.append(f"- {rule}")
        lines.append("")

        lines.append(
            "Output requirements: respond with only the Java source, preferably via the"
            f" {_PLANNER_TOOL_NAME} function when tool calls are supported."
        )
        return "\n".join(lines).strip()

    def _build_constraints(
        self,
        request: JavaPlanRequest,
        normalized_syscalls: Sequence[str],
    ) -> List[str]:
        constraints = [
            "Emit exactly one top-level Java class (any name) with helper methods and a main() entrypoint when needed.",
            "Use only the provided syscall names via the `syscall.<name>(...)` helper, keeping tool calls at the leaves.",
            "Limit every helper body to seven statements and ensure each helper is more specific than its caller.",
            "Stick to the allowed statement types (variable declarations, assignments, helper/syscall calls, if/else, enhanced for, try/catch, returns).",
            "Do not wrap the output in markdown; Java comments and imports are allowed but avoid prose explanations.",
        ]
        if not normalized_syscalls:
            constraints.append(
                "If no syscalls are available, describe diagnostic steps using logging and TODOs."
            )
        return constraints

    @staticmethod
    def _normalize_allowed_syscalls(allowed: Sequence[str] | None) -> List[str]:
        if not allowed:
            return []
        deduped = {name.strip() for name in allowed if name and name.strip()}
        return sorted(deduped)

    @staticmethod
    def _load_specification(
        override_content: Optional[str],
        override_path: Optional[Path],
    ) -> str:
        if override_content:
            return override_content.strip()
        path = override_path or _DEFAULT_SPEC_PATH
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise JavaPlanningError(f"Failed to load plan specification from {path}") from exc

    def _build_planner_tool_schema(self) -> Dict[str, Any]:
        description_lines = [
            "Return the final Java plan along with any helpful notes.",
            "Every response must comply with this specification:",
            self._specification,
        ]
        return {
            "type": "function",
            "function": {
                "name": _PLANNER_TOOL_NAME,
                "description": "\n\n".join(description_lines).strip(),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "notes": {
                            "type": "string",
                            "description": "Optional commentary about assumptions, risks, or TODOs.",
                        },
                        "java": {
                            "type": "string",
                            "description": "Complete Java source code containing a single top-level class.",
                        },
                    },
                    "required": ["java"],
                },
            },
        }


__all__ = [
    "JavaPlanRequest",
    "JavaPlanResult",
    "JavaPlanner",
    "JavaPlanningError",
]
