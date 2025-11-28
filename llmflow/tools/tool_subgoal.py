"""Scoped sub-goal execution tool.

This tool creates an isolated agent instance with a trimmed context window so
complex tasks can be decomposed into focused sub-goals. Each invocation
constructs a small task packet, launches a nested LLMFlow agent with only the
allowed tools, and returns the resulting summary plus execution metadata.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from llmflow.core.agent import Agent
from llmflow.llm_client import LLMClient
from llmflow.tools import get_module_for_tool_name, load_tool_module
from llmflow.tools.tool_decorator import register_tool

_SUBGOAL_TAGS = ["subgoal", "workflow", "control"]
_DEFAULT_SYSTEM_PROMPT = (
    "You are a scoped assistant tasked with completing a single sub-goal. "
    "Rely exclusively on the provided context packet and granted tools. "
    "Keep responses concise and actionable so the parent agent can stitch your "
    "results back into the broader plan."
)

_UNDERSTAND_INSTRUCTIONS = (
    "You review lint issues and locate the exact code that violates the rule. "
    "Use only read/inspection tools. Summarize findings with: (1) files and line "
    "ranges inspected, (2) root-cause explanation, (3) recommended files to "
    "edit. Finish with a compact JSON object under the heading 'Diagnosis'."
)

_PATCH_INSTRUCTIONS = (
    "You own implementation and verification for a single lint fix. Plan the "
    "minimal patches, apply them carefully, list impacted files, and run the "
    "specified tests. Report results under headings 'Patch Plan', 'Applied "
    "Changes', and 'Test Results'."
)

_FETCH_INSTRUCTIONS = (
    "You query the Qlty API for a single issue that matches the provided filters. "
    "Always call 'qlty_get_first_issue' with the supplied owner/project keys and filters. "
    "Summarize the returned issue with ID, slug, title, category, rule, and any "
    "candidate files. If no issue is found, clearly state that outcome. End with a "
    "JSON object under the heading 'Issue Intake' that includes issue_id, slug, "
    "summary, lint_rule, candidate_files, and filters_summary."
)

_PREPARE_WORKSPACE_INSTRUCTIONS = (
    "You are responsible for preparing the workspace before any code edits happen. "
    "Clone the repository into PROJECT_REPO_ROOT (or create it when missing), record the "
    "absolute repo_path reported by git_clone_repository, and reuse it for later git calls. "
    "Use git_suggest_branch_name to derive a unique working branch from the issue reference, "
    "then create/check out that branch via git_create_branch (falling back to git_switch_branch "
    "when it already exists). If a clone already exists at the target location, reuse it instead "
    "of recloning. Produce a JSON blob labelled 'Workspace Prep' containing repo_path, branch_name, "
    "base_branch, default_branch, and any follow-up notes."
)

_FINALIZE_INSTRUCTIONS = (
    "You finalize the fix once patches and tests are complete. Stage any remaining changes, craft "
    "a concise commit message referencing the issue, and push the branch. When possible, gather the "
    "latest diff via git_get_uncommitted_changes to summarize what changed. If pushing succeeds and a "
    "GitHub token is available, prepare a pull request draft via git_create_pull_request. Summaries should "
    "include 'Commit Summary', 'Test Evidence', and a JSON block named 'Delivery' with commit_sha, branch, "
    "tests_status, and pr_url (if created)."
)


def _serialize_context(context: Optional[Dict[str, Any]]) -> str:
    if not context:
        return "No additional context was supplied."
    try:
        return json.dumps(context, indent=2, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return repr(context)


def _load_explicit_tools(tool_names: Optional[List[str]]) -> List[str]:
    missing: List[str] = []
    for name in tool_names or []:
        module_name = get_module_for_tool_name(name)
        if not module_name:
            missing.append(name)
            continue
        load_tool_module(module_name, warn=False)
    return missing


@register_tool(tags=_SUBGOAL_TAGS)
def run_subgoal(
    goal_name: str,
    objective: str,
    instructions: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    allowed_tool_tags: Optional[List[str]] = None,
    explicit_tool_names: Optional[List[str]] = None,
    match_all_tags: bool = True,
    user_prompt: Optional[str] = None,
    max_iterations: int = 5,
    llm_config_path: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Execute a scoped sub-goal through a nested LLMFlow agent.

    Args:
        goal_name: Identifier for the sub-goal (e.g., "understand_issue").
        objective: Natural language description of the desired outcome.
        instructions: Optional system prompt override for the sub-agent.
        context: Structured data packet supplied to the sub-goal.
        allowed_tool_tags: Optional list of tool tags to expose to the sub-agent.
        explicit_tool_names: Optional list of tool names that must be loaded.
        match_all_tags: Whether all tags must match when filtering tools.
        user_prompt: Optional override for the user message sent to the sub-agent.
        max_iterations: Iteration cap for the nested agent loop.
        llm_config_path: Optional custom LLM config path; defaults to llm_config.yaml.
        verbose: Whether to print verbose diagnostics from the nested agent.

    Returns:
        Dict containing success flag, final message, and execution metadata.
    """

    if not goal_name.strip():
        return {
            "success": False,
            "error": "goal_name must be a non-empty string",
            "retryable": False,
        }
    if not objective.strip():
        return {
            "success": False,
            "error": "objective must be a non-empty string",
            "retryable": False,
        }

    config_path = Path(
        llm_config_path
        or os.getenv("LLMFLOW_SUBGOAL_LLM_CONFIG")
        or Path("llm_config.yaml").resolve()
    )

    try:
        llm_client = LLMClient(config_file=str(config_path))
    except Exception as exc:  # pragma: no cover - config errors caught early
        return {
            "success": False,
            "error": f"Failed to initialize LLMClient: {exc}",
            "retryable": False,
        }

    missing_tools = _load_explicit_tools(explicit_tool_names)

    system_prompt = instructions.strip() if instructions else _DEFAULT_SYSTEM_PROMPT
    packet = {
        "goal_name": goal_name,
        "objective": objective,
        "context": context or {},
        "missing_tools": missing_tools,
    }
    synthesized_prompt = user_prompt or (
        f"You must accomplish the sub-goal '{goal_name}'.\\n"
        f"Primary objective: {objective}.\\n\\n"
        f"Context packet:\\n{_serialize_context(context)}"
    )

    agent = Agent(
        llm_client=llm_client,
        system_prompt=system_prompt,
        available_tool_tags=allowed_tool_tags,
        match_all_tags=match_all_tags,
        max_iterations=max_iterations,
        verbose=verbose,
    )

    final_message = agent.add_user_message_and_run(synthesized_prompt)

    return {
        "success": bool(final_message and final_message.strip()),
        "goal_name": goal_name,
        "objective": objective,
        "final_message": final_message,
        "context_trace": agent.get_context_trace(),
        "packet": packet,
        "missing_tools": missing_tools,
    }


@register_tool(tags=_SUBGOAL_TAGS)
def run_fetch_issue_subgoal(
    owner_key_or_id: str,
    project_key_or_id: str,
    categories: Optional[List[str]] = None,
    statuses: Optional[List[str]] = None,
    levels: Optional[List[str]] = None,
    tools_filter: Optional[List[str]] = None,
    additional_context: Optional[Dict[str, Any]] = None,
    max_iterations: int = 3,
    verbose: bool = False,
    llm_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve the highest-priority lint issue using the Qlty tools."""

    context_packet: Dict[str, Any] = {
        "owner_key_or_id": owner_key_or_id,
        "project_key_or_id": project_key_or_id,
        "categories": categories or ["lint"],
        "statuses": statuses or ["open"],
        "levels": levels or None,
        "tools": tools_filter or None,
    }
    if additional_context:
        context_packet["additional_context"] = additional_context

    user_prompt = (
        "Use the Qlty single-issue tools to fetch the first matching issue. "
        "Return only serialized metadata that downstream sub-goals can consume."
    )

    return run_subgoal(
        goal_name="fetch_issue",
        objective="Identify the next lint issue to work on via the Qlty API.",
        instructions=_FETCH_INSTRUCTIONS,
        context=context_packet,
        allowed_tool_tags=["qlty_single_issue"],
        match_all_tags=False,
        user_prompt=user_prompt,
        max_iterations=max_iterations,
        llm_config_path=llm_config_path,
        verbose=verbose,
    )


@register_tool(tags=_SUBGOAL_TAGS)
def run_prepare_workspace_subgoal(
    repo_url: str,
    issue_reference: Optional[str] = None,
    default_branch: str = "main",
    branch_prefix: str = "fix/issue-",
    prefer_reuse_checkout: bool = True,
    project_repo_root_env: str = "PROJECT_REPO_ROOT",
    additional_context: Optional[Dict[str, Any]] = None,
    max_iterations: int = 5,
    verbose: bool = False,
    llm_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Clone the repository (or reuse an existing checkout) and create a feature branch."""

    context_packet: Dict[str, Any] = {
        "repo_url": repo_url,
        "issue_reference": issue_reference,
        "default_branch": default_branch,
        "branch_prefix": branch_prefix,
        "prefer_reuse_checkout": prefer_reuse_checkout,
        "project_repo_root_env": project_repo_root_env,
        "project_repo_root": os.getenv(project_repo_root_env),
    }
    if additional_context:
        context_packet["additional_context"] = additional_context

    user_prompt = (
        "Clone (or reuse) the repository under the configured PROJECT_REPO_ROOT, store the repo_path, "
        "and create/switch to a unique branch derived from the issue reference."
    )

    return run_subgoal(
        goal_name="prepare_workspace",
        objective="Produce a ready-to-edit checkout with a unique working branch.",
        instructions=_PREPARE_WORKSPACE_INSTRUCTIONS,
        context=context_packet,
        allowed_tool_tags=["git", "file_system", "shell"],
        match_all_tags=False,
        user_prompt=user_prompt,
        max_iterations=max_iterations,
        llm_config_path=llm_config_path,
        verbose=verbose,
    )


@register_tool(tags=_SUBGOAL_TAGS)
def run_understand_issue_subgoal(
    issue_id: str,
    repo_path: str,
    issue_summary: Optional[str] = None,
    candidate_files: Optional[List[str]] = None,
    lint_rule: Optional[str] = None,
    additional_context: Optional[Dict[str, Any]] = None,
    max_iterations: int = 4,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Analyze a lint issue and return a concise diagnosis.

    The sub-goal focuses on file discovery and reading; it should not edit the
    repository. Provide enough detail for a follow-up patch sub-goal to take
    action.
    """

    context_packet: Dict[str, Any] = {
        "issue_id": issue_id,
        "issue_summary": issue_summary or "Summary not provided",
        "repo_path": repo_path,
        "candidate_files": candidate_files or [],
        "lint_rule": lint_rule,
    }
    if additional_context:
        context_packet["additional_context"] = additional_context

    user_prompt = (
        "Inspect the repository for the supplied lint issue. Identify the exact "
        "file/lines responsible and explain why they violate the rule. If the "
        "issue text lists files, open those first before expanding the search."
    )

    return run_subgoal(
        goal_name="understand_issue",
        objective="Diagnose the root cause of the lint report and list target files.",
        instructions=_UNDERSTAND_INSTRUCTIONS,
        context=context_packet,
        allowed_tool_tags=["file_system", "text_search"],
        match_all_tags=False,
        user_prompt=user_prompt,
        max_iterations=max_iterations,
        verbose=verbose,
    )


@register_tool(tags=_SUBGOAL_TAGS)
def run_patch_issue_subgoal(
    repo_path: str,
    issue_id: str,
    diagnosis_summary: Optional[str] = None,
    target_files: Optional[List[str]] = None,
    tests_to_run: Optional[List[str]] = None,
    desired_outcome: Optional[str] = None,
    additional_context: Optional[Dict[str, Any]] = None,
    max_iterations: int = 6,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Plan and apply patches for a diagnosed lint issue.

    The sub-goal may edit files, run targeted commands, and should return patch
    notes plus test outcomes suitable for commit and PR messaging.
    """

    context_packet: Dict[str, Any] = {
        "repo_path": repo_path,
        "issue_id": issue_id,
        "diagnosis_summary": diagnosis_summary or "Diagnosis not provided",
        "target_files": target_files or [],
        "tests_to_run": tests_to_run or [],
        "desired_outcome": desired_outcome,
    }
    if additional_context:
        context_packet["additional_context"] = additional_context

    user_prompt = (
        "Using the provided diagnosis, craft the minimal code changes that fix "
        "the lint issue. Document each patch, apply it, and run the specified "
        "tests or linters."
    )

    return run_subgoal(
        goal_name="patch_issue",
        objective="Implement and validate the fix for the diagnosed lint issue.",
        instructions=_PATCH_INSTRUCTIONS,
        context=context_packet,
        allowed_tool_tags=["file_system", "git", "shell"],
        match_all_tags=False,
        user_prompt=user_prompt,
        max_iterations=max_iterations,
        verbose=verbose,
    )


@register_tool(tags=_SUBGOAL_TAGS)
def run_finalize_issue_subgoal(
    repo_path: str,
    branch_name: Optional[str] = None,
    base_branch: str = "main",
    issue_id: Optional[str] = None,
    issue_summary: Optional[str] = None,
    diagnosis_summary: Optional[str] = None,
    patch_summary: Optional[str] = None,
    tests_summary: Optional[str] = None,
    desired_outcome: Optional[str] = None,
    additional_context: Optional[Dict[str, Any]] = None,
    max_iterations: int = 4,
    verbose: bool = False,
    llm_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Stage, commit, push, and summarize the final delivery details for the fix."""

    context_packet: Dict[str, Any] = {
        "repo_path": repo_path,
        "branch_name": branch_name,
        "base_branch": base_branch,
        "issue_id": issue_id,
        "issue_summary": issue_summary,
        "diagnosis_summary": diagnosis_summary,
        "patch_summary": patch_summary,
        "tests_summary": tests_summary,
        "desired_outcome": desired_outcome,
    }
    if additional_context:
        context_packet["additional_context"] = additional_context

    user_prompt = (
        "Use the git toolchain to stage remaining files, craft a commit, push the branch, and "
        "prepare a PR-ready summary that references the provided issue context."
    )

    return run_subgoal(
        goal_name="finalize_issue",
        objective="Deliver the fix with a clean commit, tests evidence, and PR summary.",
        instructions=_FINALIZE_INSTRUCTIONS,
        context=context_packet,
        allowed_tool_tags=["git", "file_system", "shell"],
        match_all_tags=False,
        user_prompt=user_prompt,
        max_iterations=max_iterations,
        llm_config_path=llm_config_path,
        verbose=verbose,
    )
