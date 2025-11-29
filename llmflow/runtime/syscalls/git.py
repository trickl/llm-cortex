"""Git-related syscall module."""
from __future__ import annotations

from typing import Optional, Sequence

from llmflow.tools import tool_git as default_git_tools

from .base import BaseSyscallModule, ensure_tool_success


class GitSyscalls(BaseSyscallModule):
    """Expose git workflow helpers to plan programs."""

    def __init__(self, *, git_module=default_git_tools):
        self._git = git_module

    def _call(self, syscall_name: str, func, **kwargs):
        payload = func(**kwargs)
        return ensure_tool_success(syscall_name, payload)

    def clone_repo(
        self,
        repo_url: str,
        destination: Optional[str] = None,
        branch: Optional[str] = None,
        depth: Optional[int] = None,
    ):
        return self._call(
            "cloneRepo",
            self._git.git_clone_repository,
            repo_url=repo_url,
            destination=destination,
            branch=branch,
            depth=depth,
        )

    def create_branch(
        self,
        repo_path: str,
        branch_name: str,
        base_branch: str = "main",
    ):
        return self._call(
            "createBranch",
            self._git.git_create_branch,
            repo_path=repo_path,
            branch_name=branch_name,
            base_branch=base_branch,
        )

    def suggest_branch_name(
        self,
        repo_path: Optional[str] = None,
        issue_reference: Optional[str] = None,
        prefix: str = "fix/issue-",
        max_suffix_attempts: int = 50,
    ):
        return self._call(
            "suggestBranchName",
            self._git.git_suggest_branch_name,
            repo_path=repo_path,
            issue_reference=issue_reference,
            prefix=prefix,
            max_suffix_attempts=max_suffix_attempts,
        )

    def switch_branch(
        self,
        repo_path: str,
        branch_name: Optional[str] = None,
        branch: Optional[str] = None,
    ):
        return self._call(
            "switchBranch",
            self._git.git_switch_branch,
            repo_path=repo_path,
            branch_name=branch_name,
            branch=branch,
        )

    def stage_paths(
        self,
        repo_path: str,
        paths: Optional[Sequence[str]] = None,
    ):
        return self._call(
            "stagePaths",
            self._git.git_stage_paths,
            repo_path=repo_path,
            paths=paths,
        )

    def commit_changes(
        self,
        repo_path: str,
        message: str,
        author_name: Optional[str] = None,
        author_email: Optional[str] = None,
    ):
        return self._call(
            "commitChanges",
            self._git.git_commit_changes,
            repo_path=repo_path,
            message=message,
            author_name=author_name,
            author_email=author_email,
        )

    def get_uncommitted_changes(self, repo_path: str):
        return self._call(
            "getUncommittedChanges",
            self._git.git_get_uncommitted_changes,
            repo_path=repo_path,
        )

    def push_branch(
        self,
        repo_path: str,
        remote_name: str = "origin",
        branch_name: Optional[str] = None,
        set_upstream: bool = True,
    ):
        return self._call(
            "pushBranch",
            self._git.git_push_branch,
            repo_path=repo_path,
            remote_name=remote_name,
            branch_name=branch_name,
            set_upstream=set_upstream,
        )

    def create_pull_request(
        self,
        repository: str,
        head_branch: str,
        base_branch: str = "main",
        title: Optional[str] = None,
        body: Optional[str] = None,
        draft: bool = False,
        token_env_var: str = "GITHUB_TOKEN",
    ):
        return self._call(
            "createPullRequest",
            self._git.git_create_pull_request,
            repository=repository,
            head_branch=head_branch,
            base_branch=base_branch,
            title=title,
            body=body,
            draft=draft,
            token_env_var=token_env_var,
        )

    def get_syscalls(self):
        return {
            "cloneRepo": self.clone_repo,
            "createBranch": self.create_branch,
            "suggestBranchName": self.suggest_branch_name,
            "switchBranch": self.switch_branch,
            "stagePaths": self.stage_paths,
            "commitChanges": self.commit_changes,
            "getUncommittedChanges": self.get_uncommitted_changes,
            "pushBranch": self.push_branch,
            "createPullRequest": self.create_pull_request,
        }


__all__ = ["GitSyscalls"]
