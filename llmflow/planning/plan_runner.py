"""Execution helpers for Java plan programs."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from llmflow.runtime.syscall_registry import SyscallRegistry
from llmflow.runtime.syscalls import build_default_syscall_registry

from .deferred_planner import DeferredFunctionPrompt
from .executor import PlanExecutor
from .java_planner import JavaPlanningError
from .runtime.ast import DeferredExecutionOptions


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SPEC_PATH = _PROJECT_ROOT / "planning" / "java_planning.md"


class DeferredBodyPlanner:
    """Adapter that uses an LLM to synthesize deferred Java function bodies."""

    _SYSTEM_PROMPT = (
        "You generate Java plan function bodies."
        " Respond with only the function body block (including braces) that satisfies"
        " the user's request."
    )

    def __init__(self, llm_client):
        self._llm_client = llm_client

    def __call__(self, prompt: DeferredFunctionPrompt) -> str:
        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user", "content": prompt.prompt},
        ]
        response = self._llm_client.generate(messages=messages, tools=None)
        body = response.get("content") if isinstance(response, dict) else None
        if not isinstance(body, str):
            raise JavaPlanningError("Deferred planner did not return textual content.")
        normalized = body.strip()
        if not normalized:
            raise JavaPlanningError("Deferred planner returned an empty body.")
        return normalized


class PlanRunner:
    """High-level wrapper that executes Java plans using :class:`PlanExecutor`."""

    def __init__(
        self,
        *,
        registry_factory: Optional[Callable[[], SyscallRegistry]] = None,
        deferred_planner: Optional[Callable[[DeferredFunctionPrompt], str]] = None,
        specification: Optional[str] = None,
    ):
        self._registry_factory = registry_factory or build_default_syscall_registry
        self._deferred_planner = deferred_planner
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
    ) -> Dict[str, Any]:
        registry = self._registry_factory()
        deferred_options = None
        if self._deferred_planner is not None:
            deferred_options = DeferredExecutionOptions(
                goal_summary=goal_summary,
                metadata=dict(deferred_metadata or {}),
                extra_constraints=list(deferred_constraints or []),
            )
        executor = PlanExecutor(
            registry,
            deferred_planner=self._deferred_planner,
            deferred_options=deferred_options,
            specification=self._specification,
        )
        extra_metadata = dict(metadata) if metadata else None
        return executor.execute_from_string(
            plan_source,
            capture_trace=capture_trace,
            metadata=extra_metadata,
        )

    @staticmethod
    def _load_specification() -> str:
        try:
            return _DEFAULT_SPEC_PATH.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - depends on filesystem
            raise JavaPlanningError(
                f"Unable to load plan specification from '{_DEFAULT_SPEC_PATH}'."
            ) from exc


__all__ = ["PlanRunner", "DeferredBodyPlanner"]
