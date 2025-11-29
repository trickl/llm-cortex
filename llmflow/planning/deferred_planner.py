"""Helpers for prompting deferred Java function generation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


def _serialize_value(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return repr(value)


@dataclass
class DeferredParameter:
    name: str
    type: str


@dataclass
class DeferredFunctionContext:
    function_name: str
    return_type: str
    parameters: List[DeferredParameter]
    argument_values: Dict[str, Any]
    call_stack: List[str] = field(default_factory=list)
    goal_summary: Optional[str] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeferredFunctionPrompt:
    prompt: str
    context: DeferredFunctionContext
    allowed_syscalls: List[str]


_DEFAULT_CONSTRAINTS = (
    "Generate only the Java function body block (starting and ending with braces).",
    "Use only the syscalls listed below; regular functions cannot call arbitrary tools.",
    "Do not defer the immediate next action inside this function.",
    "Prefer deterministic, minimal logic that can run without additional clarification.",
)


def build_deferred_prompt(
    context: DeferredFunctionContext,
    specification: str,
    allowed_syscalls: Iterable[str],
    extra_constraints: Optional[Iterable[str]] = None,
) -> str:
    """Create a natural-language prompt for synthesizing a deferred Java function body."""

    lines: List[str] = []
    lines.append("You are filling in a deferred Java plan function body.")
    signature = _format_signature(context)
    lines.append("")
    lines.append(f"Function signature:\n{signature}")
    lines.append("")
    if context.argument_values:
        lines.append("Inputs:")
        for name, value in context.argument_values.items():
            lines.append(f"- {name}: {_serialize_value(value)}")
        lines.append("")
    if context.goal_summary:
        lines.append(f"Goal summary:\n{context.goal_summary}\n")
    if context.call_stack:
        lines.append(
            "Current call path: " + " -> ".join(context.call_stack + [context.function_name])
        )
        lines.append("")
    if context.extra_metadata:
        lines.append("Metadata:")
        for key, value in context.extra_metadata.items():
            lines.append(f"- {key}: {_serialize_value(value)}")
        lines.append("")

    if specification:
        lines.append("Plan specification:")
        lines.append(specification.strip())
        lines.append("")
    constraints = list(_DEFAULT_CONSTRAINTS)
    if extra_constraints:
        constraints.extend(extra_constraints)
    lines.append("Constraints:")
    for idx, rule in enumerate(constraints, start=1):
        lines.append(f"{idx}. {rule}")
    lines.append("")
    syscalls = list(dict.fromkeys(allowed_syscalls))
    lines.append("Allowed syscalls:")
    if syscalls:
        for name in syscalls:
            lines.append(f"- {name}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append(
        "Respond ONLY with the Java function body block (including braces) that satisfies these requirements."
    )
    return "\n".join(lines).strip()


def _format_signature(context: DeferredFunctionContext) -> str:
    params = ", ".join(f"{param.type} {param.name}" for param in context.parameters)
    return f"{context.return_type} {context.function_name}({params})"


def prepare_deferred_prompt(
    context: DeferredFunctionContext,
    specification: str,
    allowed_syscalls: Iterable[str],
    extra_constraints: Optional[Iterable[str]] = None,
) -> DeferredFunctionPrompt:
    prompt = build_deferred_prompt(context, specification, allowed_syscalls, extra_constraints)
    return DeferredFunctionPrompt(prompt=prompt, context=context, allowed_syscalls=list(allowed_syscalls))


__all__ = [
    "DeferredFunctionContext",
    "DeferredFunctionPrompt",
    "DeferredParameter",
    "build_deferred_prompt",
    "prepare_deferred_prompt",
]
