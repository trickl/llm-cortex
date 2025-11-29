"""Sub-goal orchestration syscalls bridged from the tool_subgoal module."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseSyscallModule, ensure_tool_success


class SubgoalSyscalls(BaseSyscallModule):
    """Expose scoped sub-goal helpers to plan programs."""

    def __init__(self, *, subgoal_module=None):
        self._subgoals = subgoal_module

    def _module(self):
        if self._subgoals is None:
            from llmflow.tools import tool_subgoal as loaded_subgoal_module

            self._subgoals = loaded_subgoal_module
        return self._subgoals

    def _call(self, syscall_name: str, func, **kwargs):
        payload = func(**kwargs)
        return ensure_tool_success(syscall_name, payload)

    def run_subgoal(
        self,
        goal_name: str,
        objective: str,
        instructions: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        allowed_tool_tags: Optional[List[str]] = None,
        explicit_tool_names: Optional[List[str]] = None,
        match_all_tags: bool = True,
        user_prompt: Optional[str] = None,
        plan_max_retries: int = 1,
        llm_config_path: Optional[str] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        module = self._module()
        return self._call(
            "runSubgoal",
            module.run_subgoal,
            goal_name=goal_name,
            objective=objective,
            instructions=instructions,
            context=context,
            allowed_tool_tags=allowed_tool_tags,
            explicit_tool_names=explicit_tool_names,
            match_all_tags=match_all_tags,
            user_prompt=user_prompt,
            plan_max_retries=plan_max_retries,
            llm_config_path=llm_config_path,
            verbose=verbose,
        )

    def run_fetch_issue_subgoal(
        self,
        owner_key_or_id: str,
        project_key_or_id: str,
        repo_url: Optional[str] = None,
        categories: Optional[List[str]] = None,
        statuses: Optional[List[str]] = None,
        levels: Optional[List[str]] = None,
        tools_filter: Optional[List[str]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        plan_max_retries: int = 1,
        verbose: bool = False,
        llm_config_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        module = self._module()
        return self._call(
            "runFetchIssueSubgoal",
            module.run_fetch_issue_subgoal,
            owner_key_or_id=owner_key_or_id,
            project_key_or_id=project_key_or_id,
            repo_url=repo_url,
            categories=categories,
            statuses=statuses,
            levels=levels,
            tools_filter=tools_filter,
            additional_context=additional_context,
            plan_max_retries=plan_max_retries,
            verbose=verbose,
            llm_config_path=llm_config_path,
        )

    def run_prepare_workspace_subgoal(
        self,
        repo_url: Optional[str] = None,
        issue_reference: Optional[str] = None,
        default_branch: str = "main",
        branch_prefix: str = "fix/issue-",
        prefer_reuse_checkout: bool = True,
        project_repo_root_env: str = "PROJECT_REPO_ROOT",
        fallback_repo_url_envs: Optional[List[str]] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        plan_max_retries: int = 1,
        verbose: bool = False,
        llm_config_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        module = self._module()
        return self._call(
            "runPrepareWorkspaceSubgoal",
            module.run_prepare_workspace_subgoal,
            repo_url=repo_url,
            issue_reference=issue_reference,
            default_branch=default_branch,
            branch_prefix=branch_prefix,
            prefer_reuse_checkout=prefer_reuse_checkout,
            project_repo_root_env=project_repo_root_env,
            fallback_repo_url_envs=fallback_repo_url_envs,
            additional_context=additional_context,
            plan_max_retries=plan_max_retries,
            verbose=verbose,
            llm_config_path=llm_config_path,
        )

    def run_understand_issue_subgoal(
        self,
        issue_id: str,
        repo_path: str,
        issue_summary: Optional[str] = None,
        candidate_files: Optional[List[str]] = None,
        lint_rule: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        plan_max_retries: int = 1,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        module = self._module()
        return self._call(
            "runUnderstandIssueSubgoal",
            module.run_understand_issue_subgoal,
            issue_id=issue_id,
            repo_path=repo_path,
            issue_summary=issue_summary,
            candidate_files=candidate_files,
            lint_rule=lint_rule,
            additional_context=additional_context,
            plan_max_retries=plan_max_retries,
            verbose=verbose,
        )

    def run_patch_issue_subgoal(
        self,
        repo_path: str,
        issue_id: str,
        diagnosis_summary: Optional[str] = None,
        target_files: Optional[List[str]] = None,
        tests_to_run: Optional[List[str]] = None,
        desired_outcome: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        plan_max_retries: int = 1,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        module = self._module()
        return self._call(
            "runPatchIssueSubgoal",
            module.run_patch_issue_subgoal,
            repo_path=repo_path,
            issue_id=issue_id,
            diagnosis_summary=diagnosis_summary,
            target_files=target_files,
            tests_to_run=tests_to_run,
            desired_outcome=desired_outcome,
            additional_context=additional_context,
            plan_max_retries=plan_max_retries,
            verbose=verbose,
        )

    def run_finalize_issue_subgoal(
        self,
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
        plan_max_retries: int = 1,
        verbose: bool = False,
        llm_config_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        module = self._module()
        return self._call(
            "runFinalizeIssueSubgoal",
            module.run_finalize_issue_subgoal,
            repo_path=repo_path,
            branch_name=branch_name,
            base_branch=base_branch,
            issue_id=issue_id,
            issue_summary=issue_summary,
            diagnosis_summary=diagnosis_summary,
            patch_summary=patch_summary,
            tests_summary=tests_summary,
            desired_outcome=desired_outcome,
            additional_context=additional_context,
            plan_max_retries=plan_max_retries,
            verbose=verbose,
            llm_config_path=llm_config_path,
        )

    def get_syscalls(self):
        return {
            "runSubgoal": self.run_subgoal,
            "runFetchIssueSubgoal": self.run_fetch_issue_subgoal,
            "runPrepareWorkspaceSubgoal": self.run_prepare_workspace_subgoal,
            "runUnderstandIssueSubgoal": self.run_understand_issue_subgoal,
            "runPatchIssueSubgoal": self.run_patch_issue_subgoal,
            "runFinalizeIssueSubgoal": self.run_finalize_issue_subgoal,
        }


__all__ = ["SubgoalSyscalls"]
