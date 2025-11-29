"""Utilities for prompting the LLM to emit Java plans."""
from __future__ import annotations

import json
import textwrap
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field, ValidationError, field_validator

from llmflow.llm_client import LLMClient

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SPEC_PATH = _PROJECT_ROOT / "planning" / "java_planning.md"
_PLANNER_TOOL_NAME = "define_java_plan"


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
        ..., description="Complete Java source code containing a public class Plan."
    )

    @field_validator("java")
    @classmethod
    def _ensure_plan_block(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed.lower().startswith("public class"):
            raise ValueError("Java payload must start with a public class definition.")
        return trimmed

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
        try:
            payload = self._llm_client.structured_generate(
                messages=messages,
                response_model=_PlannerToolPayload,
                tools=[self._planner_tool_schema],
                tool_choice=self._planner_tool_choice,
            )
        except ValidationError as exc:
            raise JavaPlanningError("Planner returned an invalid structured payload.") from exc

        plan_source = payload.java.strip()
        plan_id = str(request.metadata.get("plan_id") or uuid.uuid4())
        normalized_syscalls = self._normalize_allowed_syscalls(request.allowed_syscalls)
        metadata = dict(request.metadata)
        metadata.setdefault("allowed_syscalls", normalized_syscalls)
        if payload.notes:
            metadata["planner_notes"] = payload.notes.strip()
        return JavaPlanResult(
            plan_id=plan_id,
            plan_source=plan_source,
            raw_response=payload.model_dump(),
            prompt_messages=messages,
            metadata=metadata,
        )

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
            " Produce a single Java class named Plan that fully solves the user's task."
            " Refer to the define_java_plan tool description for the full specification"
            " and do not emit explanations."
        )
        tools_block = json.dumps(
            {
                "available_tools": [self._planner_tool_schema["function"]],
                "instructions": "Always call define_java_plan exactly once to return the final plan.",
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
            "Output requirements: respond with only the Java source returned via the"
            f" {_PLANNER_TOOL_NAME} function."
        )
        return "\n".join(lines).strip()

    def _build_constraints(
        self,
        request: JavaPlanRequest,
        normalized_syscalls: Sequence[str],
    ) -> List[str]:
        constraints = [
            "Implement public class Plan with helper methods plus a main() entrypoint.",
            "Use only the provided syscall names via the `syscall.<name>(...)` helper.",
            "Keep each method focused on one sub-goal and under 7 statements.",
            "Annotate methods that require runtime synthesis with @Deferred.",
            "Do not write prose or explanations; emit compilable Java only.",
        ]
        if not normalized_syscalls:
            constraints.append(
                "If no syscalls are available, describe diagnostic steps using logging and TODOs."
            )
        if not request.include_deferred_guidance:
            constraints = [rule for rule in constraints if "@Deferred" not in rule]
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
                            "description": "Complete Java source code containing a public class Plan.",
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
