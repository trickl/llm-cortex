"""Java-plan Agent orchestrator.

This module provides a single entry point for building agents that rely on the
Cortex Java planning workflow. Instead of iteratively calling tools directly
from LLM responses, the agent now:

1. Synthesizes a Java plan with :class:`~llmflow.planning.JavaPlanner`.
2. Executes the plan via :class:`~llmflow.planning.plan_runner.PlanRunner`.
3. Surfaces the orchestrator summary (or plan return value) back to the user.

Tool exposure is controlled through syscall whitelists derived from the default
syscall registry or filtered via tool tags. Conversation memory is retained for
traceability, but the execution flow is entirely plan-driven.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from llmflow.runtime.syscalls import build_default_syscall_registry

from llmflow.llm_client import LLMClient
from llmflow.logging_utils import RunArtifactManager, RunLogContext
from llmflow.planning import JavaPlanner, JavaPlanRequest, PlanOrchestrator
from llmflow.planning.plan_runner import PlanRunner, DeferredBodyPlanner
from llmflow.telemetry.mermaid_recorder import MermaidSequenceRecorder
from llmflow.tools import get_module_for_tool_name, load_tool_module
from llmflow.tools.tool_registry import get_tool_schema, get_tool_tags

from .agent_instrumentation import AgentInstrumentationMixin
from .goals import GoalManager
from .memory import Memory


_JAVA_PLANNING_GUIDANCE = (
    "When responding you must first synthesize a complete Java plan (public class Plan) "
    "that conforms to the `define_java_plan` tool specification. Do not execute tools "
    "directly; emit only Java source for the runtime to execute."
)
_SYSTEM_PROMPT_TEMPLATE = (
    "{{ base_prompt }}\n\n"
    "{{ planning_guidance }}"
)
_SYSTEM_PROMPT_PREVIEW = (
    "{{ base_prompt }}\n\n"
    "{{ planning_guidance }}"
)


_SYS_CALL_TOOL_MAP: Dict[str, Optional[str]] = {
    "log": None,
    "listFilesInTree": "list_files_in_tree",
    "readTextFile": "read_text_file",
    "overwriteTextFile": "overwrite_text_file",
    "applyTextRewrite": "apply_text_rewrite",
    "cloneRepo": "git_clone_repository",
    "createBranch": "git_create_branch",
    "suggestBranchName": "git_suggest_branch_name",
    "switchBranch": "git_switch_branch",
    "stagePaths": "git_stage_paths",
    "commitChanges": "git_commit_changes",
    "getUncommittedChanges": "git_get_uncommitted_changes",
    "pushBranch": "git_push_branch",
    "createPullRequest": "git_create_pull_request",
    "qltyListIssues": "qlty_list_issues",
    "qltyGetFirstIssue": "qlty_get_first_issue",
    "runSubgoal": "run_subgoal",
    "runFetchIssueSubgoal": "run_fetch_issue_subgoal",
    "runPrepareWorkspaceSubgoal": "run_prepare_workspace_subgoal",
    "runUnderstandIssueSubgoal": "run_understand_issue_subgoal",
    "runPatchIssueSubgoal": "run_patch_issue_subgoal",
    "runFinalizeIssueSubgoal": "run_finalize_issue_subgoal",
}


class Agent(AgentInstrumentationMixin):
    """Goal-aware Java plan orchestrator with syscall filtering."""

    _MAX_CONTEXT_TRACE = 100

    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str = (
            "You are a helpful AI assistant. You coordinate structured Java plans to solve user"
            " requests using the available syscalls."
        ),
        initial_goals: Optional[List[Dict[str, Any]]] = None,
        available_tool_tags: Optional[List[str]] = None,
        match_all_tags: bool = True,
        allowed_syscalls: Optional[Sequence[str]] = None,
        registry_factory: Optional[Callable[[], Any]] = None,
        runner_factory: Optional[Callable[[], PlanRunner]] = None,
        planner: Optional[JavaPlanner] = None,
        plan_max_retries: int = 1,
        capture_trace: bool = False,
        verbose: bool = True,
        enable_run_logging: bool = True,
    ):
        self.llm_client = llm_client
        self.goal_manager = GoalManager(initial_goals=initial_goals)
        self.system_prompt, self._system_prompt_template_preview = self._render_system_prompt(
            system_prompt
        )
        self.memory = Memory(system_prompt=self.system_prompt)
        self.available_tool_tags = available_tool_tags
        self.match_all_tags_for_tools = match_all_tags
        self.verbose = verbose
        self.active_tools_schemas: List[Dict[str, Any]] = []
        self.context_trace: List[Dict[str, Any]] = []
        self.current_iteration = 0
        self._capture_trace = capture_trace
        self.plan_max_retries = max(plan_max_retries, 0)
        self._registry_factory = registry_factory or build_default_syscall_registry
        self._planner = planner or JavaPlanner(llm_client)
        self._deferred_planner = DeferredBodyPlanner(llm_client)
        if runner_factory is not None:
            self._runner_factory = runner_factory
        else:
            self._runner_factory = self._build_default_runner_factory
        self._orchestrator = PlanOrchestrator(
            self._planner,
            self._runner_factory,
            max_retries=self.plan_max_retries,
        )

        available_syscalls = self._discover_syscalls()
        self.allowed_syscalls = self._resolve_allowed_syscalls(
            available_syscalls,
            allowed_syscalls,
            available_tool_tags,
            match_all_tags,
        )
        self.active_tools_schemas = self._build_tool_schemas(self.allowed_syscalls)

        self.enable_run_logging = enable_run_logging
        self._run_log_context: Optional[RunLogContext] = None
        self._run_artifact_manager: Optional[RunArtifactManager] = None
        self._mermaid_recorder: Optional[MermaidSequenceRecorder] = None
        self._run_failed: bool = False
        self._last_prompt_summary: Optional[str] = None
        self._owns_run_directory = False
        self._last_run_summary: Optional[str] = None

        if self.verbose:
            print("Agent initialized for Java planning.")
            print("System Prompt Template (spec omitted in logs):")
            print(self._system_prompt_template_preview)
            print(
                f"Allowed syscalls ({len(self.allowed_syscalls)}): "
                + ", ".join(self.allowed_syscalls)
            )
            print(f"Plan retries: {self.plan_max_retries}")

    # ------------------------------------------------------------------
    # Public API

    def add_user_message_and_run(self, user_input: str) -> Optional[str]:
        """Record ``user_input`` and execute a Java plan once."""

        if not user_input or not user_input.strip():
            raise ValueError("user_input must be a non-empty string")

        self._start_run_instrumentation()
        self.current_iteration = 1
        try:
            self.memory.add_user_message(user_input)
            self._record_context_snapshot("pre_plan_request")
            plan_request = self._build_plan_request(user_input)
            if self.verbose:
                print("Submitting Java plan request...")
            result = self._orchestrator.execute_with_retries(
                plan_request,
                capture_trace=self._capture_trace,
                metadata={"allowed_syscalls": list(self.allowed_syscalls)},
                goal_summary=self.goal_manager.get_goals_for_prompt(),
            )
            final_message = self._finalize_plan_result(result)
            if final_message:
                self.memory.add_assistant_message(final_message)
            self._append_context_trace(result)
            self._last_run_summary = result.get("summary")
            if not result.get("success", False):
                self._mark_run_failure()
            return final_message
        finally:
            self._finalize_run_instrumentation()

    def get_context_trace(self) -> List[Dict[str, Any]]:
        """Expose the captured execution context."""

        return list(self.context_trace)

    # ------------------------------------------------------------------
    # Internal helpers

    def _build_default_runner_factory(self) -> PlanRunner:
        return PlanRunner(
            registry_factory=self._registry_factory,
            deferred_planner=self._deferred_planner,
        )

    def _discover_syscalls(self) -> List[str]:
        registry = self._registry_factory()
        names = sorted(registry.to_dict().keys())
        if not names:
            raise RuntimeError("No syscalls registered in the provided registry factory.")
        return names

    def _resolve_allowed_syscalls(
        self,
        available: List[str],
        explicit: Optional[Sequence[str]],
        tags: Optional[List[str]],
        match_all: bool,
    ) -> List[str]:
        if explicit:
            normalized = {name.strip(): None for name in explicit if name and name.strip()}
            missing = [name for name in normalized if name not in available]
            if missing:
                raise ValueError(f"Unknown syscalls requested: {', '.join(missing)}")
            return sorted(normalized)

        if tags:
            filtered = self._filter_syscalls_by_tags(available, tags, match_all)
            if not filtered:
                raise ValueError(
                    f"No syscalls matched tags {tags} (match_all={match_all})."
                )
            return filtered

        return list(available)

    def _filter_syscalls_by_tags(
        self,
        available: Sequence[str],
        tags: Sequence[str],
        match_all: bool,
    ) -> List[str]:
        desired = {tag.strip().lower() for tag in tags if tag and tag.strip()}
        if not desired:
            return list(available)

        matched: List[str] = []
        for name in available:
            tool_tags = self._tags_for_syscall(name)
            if not tool_tags:
                # Built-ins like log remain available regardless of tag filters.
                matched.append(name)
                continue
            if match_all:
                if desired.issubset(tool_tags):
                    matched.append(name)
            elif tool_tags.intersection(desired):
                matched.append(name)
        return matched

    def _tags_for_syscall(self, syscall_name: str) -> Set[str]:
        tool_name = _SYS_CALL_TOOL_MAP.get(syscall_name)
        if tool_name is None:
            return {"utility"}
        tool_tags = get_tool_tags(tool_name) or []
        return {tag.lower() for tag in tool_tags}

    def _build_tool_schemas(self, syscalls: Sequence[str]) -> List[Dict[str, Any]]:
        seen: Set[str] = set()
        schemas: List[Dict[str, Any]] = []
        for syscall in syscalls:
            tool_name = _SYS_CALL_TOOL_MAP.get(syscall)
            if not tool_name or tool_name in seen:
                continue
            seen.add(tool_name)
            schema = get_tool_schema(tool_name)
            if schema is None:
                module_name = get_module_for_tool_name(tool_name)
                if module_name:
                    load_tool_module(module_name, warn=self.verbose)
                    schema = get_tool_schema(tool_name)
            if schema:
                schemas.append(schema)
        return schemas

    def _build_plan_request(self, user_input: str) -> JavaPlanRequest:
        goals = [goal.description for goal in self.goal_manager.goals]
        context_sections = [f"System prompt:\n{self.system_prompt.strip()}".strip()]
        if self._last_run_summary:
            context_sections.append(
                f"Previous plan summary:\n{self._last_run_summary.strip()}"
            )
        context_sections.append(self._format_recent_memory())
        context = "\n\n".join(section for section in context_sections if section)
        return JavaPlanRequest(
            task=user_input.strip(),
            goals=goals,
            context=context,
            allowed_syscalls=self.allowed_syscalls,
            metadata={
                "goal_count": len(goals),
                "source": "llmflow.core.agent",
            },
        )

    def _format_recent_memory(self, limit: int = 6) -> str:
        recent = self.memory.get_last_n_messages(limit, as_dicts=True)
        if not recent:
            return "No prior conversation."
        lines: List[str] = ["Recent conversation:"]
        for message in recent:
            role = message.get("role", "unknown")
            content = message.get("content")
            if isinstance(content, str):
                snippet = content.strip()
            else:
                snippet = json.dumps(content, ensure_ascii=False)
            lines.append(f"- {role}: {snippet}")
        return "\n".join(lines)

    def _finalize_plan_result(self, result: Dict[str, Any]) -> str:
        summary = result.get("summary")
        execution = result.get("final_execution") or {}
        if execution.get("success"):
            return self._format_success_message(execution, summary)
        errors = execution.get("errors") or []
        if errors:
            return self._format_failure_message(errors, summary)
        if summary:
            return summary
        return "Plan run finished without additional details."

    def _format_success_message(
        self,
        execution: Dict[str, Any],
        summary: Optional[str],
    ) -> str:
        return_value = execution.get("return_value")
        if isinstance(return_value, str) and return_value.strip():
            return return_value.strip()
        if return_value not in (None, ""):
            try:
                return json.dumps(return_value, ensure_ascii=False, indent=2)
            except TypeError:
                return str(return_value)
        return summary or "✅ Java plan run completed successfully."

    def _format_failure_message(
        self,
        errors: Sequence[Dict[str, Any]],
        summary: Optional[str],
    ) -> str:
        first = errors[0] if errors else {}
        err_type = first.get("type") or "execution_error"
        message = first.get("message") or "Plan execution failed."
        location = []
        if first.get("function"):
            location.append(f"function {first['function']}")
        if first.get("line") is not None:
            loc = f"line {first['line']}"
            if first.get("column") is not None:
                loc += f", column {first['column']}"
            location.append(loc)
        suffix = f" ({'; '.join(location)})" if location else ""
        base = f"❌ {err_type}: {message}{suffix}"
        if summary:
            return f"{base}\n{summary}"
        return base

    def _append_context_trace(self, result: Dict[str, Any]) -> None:
        entry = {
            "stage": "plan_run",
            "success": result.get("success", False),
            "attempts": len(result.get("attempts") or []),
            "summary": result.get("summary"),
        }
        self.context_trace.append(entry)
        if len(self.context_trace) > self._MAX_CONTEXT_TRACE:
            self.context_trace.pop(0)
        self._record_context_snapshot("post_plan_execution")

    def _pending_goal_count(self) -> int:
        return sum(1 for goal in self.goal_manager.goals if not goal.completed)

    def _render_system_prompt(self, base_prompt: str) -> tuple[str, str]:
        prompt = (base_prompt or "").strip()
        if not prompt:
            prompt = "You are a helpful AI assistant that coordinates Java plans."
        replacements = {
            "{{ base_prompt }}": prompt,
            "{{ planning_guidance }}": _JAVA_PLANNING_GUIDANCE,
        }
        rendered = _SYSTEM_PROMPT_TEMPLATE
        preview = _SYSTEM_PROMPT_PREVIEW
        for placeholder, value in replacements.items():
            rendered = rendered.replace(placeholder, value)
            preview = preview.replace(placeholder, value)
        return rendered.strip(), preview.strip()

