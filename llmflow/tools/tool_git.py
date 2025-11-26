"""Git tooling for repository automation.

These helper tools expose a subset of common git workflows—cloning, branching,
staging/committing, retrieving diffs for commit messages, pushing branches, and
opening pull requests—so agents can manage source control tasks end-to-end.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:  # pragma: no cover - exercised indirectly in tests
    from git import Actor as GitActor, Repo as GitRepo

    _GIT_AVAILABLE = True
except ImportError:  # pragma: no cover - handled by tool responses
    GitActor = None  # type: ignore
    GitRepo = None  # type: ignore
    _GIT_AVAILABLE = False

try:  # pragma: no cover - optional dependency for PR creation
    from github import Github

    _GITHUB_AVAILABLE = True
except ImportError:  # pragma: no cover
    Github = None  # type: ignore
    _GITHUB_AVAILABLE = False

from llmflow.tools.tool_decorator import register_tool

_TOOL_TAGS = ["git", "devops", "version_control"]


def _failure(message: str) -> Dict[str, Any]:
    return {"success": False, "error": message}


def _require_git() -> None:
    if not _GIT_AVAILABLE:
        raise RuntimeError(
            "GitPython is not installed. Install the 'GitPython' dependency to use git tools."
        )


def _require_github() -> None:
    if not _GITHUB_AVAILABLE:
        raise RuntimeError(
            "PyGithub is not installed. Install the 'PyGithub' dependency to create pull requests."
        )


def _resolve_repo(repo_path: str):
    _require_git()
    path = Path(repo_path).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"Repository path does not exist: {path}")
    try:
        return GitRepo(path)
    except Exception as exc:  # pragma: no cover - Repo raises different subclasses
        raise RuntimeError(f"Failed to open git repository at {path}: {exc}") from exc


def _actor(name: Optional[str], email: Optional[str]):
    if not name or not email:
        return None
    if not _GIT_AVAILABLE:
        return None
    return GitActor(name, email)


@register_tool(tags=_TOOL_TAGS)
def git_clone_repository(
    repo_url: str,
    destination: Optional[str] = None,
    branch: Optional[str] = None,
    depth: Optional[int] = None,
) -> Dict[str, Any]:
    """Clone a git repository, optionally targeting a branch and depth.

    Args:
        repo_url: HTTPS/SSH path or local filesystem path to clone.
        destination: Optional directory to clone into. When omitted, a temporary
            directory is created and returned.
        branch: Branch or tag to check out after cloning.
        depth: Optional shallow clone depth.
    """

    try:
        _require_git()
        target_dir = (
            Path(destination).expanduser().resolve()
            if destination
            else Path(tempfile.mkdtemp(prefix="llmflow-git-"))
        )
        clone_kwargs = {}
        if depth:
            clone_kwargs["depth"] = depth
        repo = GitRepo.clone_from(repo_url, target_dir, **clone_kwargs)
        if branch:
            repo.git.checkout(branch)
        return {
            "success": True,
            "path": str(target_dir),
            "branch": branch or repo.active_branch.name if repo.head.is_valid() else None,
            "commit": repo.head.commit.hexsha if repo.head.is_valid() else None,
        }
    except Exception as exc:
        return _failure(str(exc))


@register_tool(tags=_TOOL_TAGS)
def git_create_branch(
    repo_path: str,
    branch_name: str,
    base_branch: str = "main",
) -> Dict[str, Any]:
    """Create a new branch from the specified base.

    If the branch already exists, it is simply checked out.
    """

    try:
        repo = _resolve_repo(repo_path)
        heads_by_name = {head.name: head for head in repo.branches}
        if branch_name in heads_by_name:
            repo.git.checkout(branch_name)
            return {
                "success": True,
                "branch": branch_name,
                "message": "Branch already existed; checked out.",
            }

        base = heads_by_name.get(base_branch, repo.active_branch)
        new_branch = repo.create_head(branch_name, commit=base.commit)
        repo.git.checkout(branch_name)
        return {
            "success": True,
            "branch": branch_name,
            "base": base.name,
            "commit": new_branch.commit.hexsha,
        }
    except Exception as exc:
        return _failure(str(exc))


@register_tool(tags=_TOOL_TAGS)
def git_switch_branch(repo_path: str, branch_name: str) -> Dict[str, Any]:
    """Check out the given branch."""

    try:
        repo = _resolve_repo(repo_path)
        repo.git.checkout(branch_name)
        return {"success": True, "branch": branch_name, "commit": repo.head.commit.hexsha}
    except Exception as exc:
        return _failure(str(exc))


@register_tool(tags=_TOOL_TAGS)
def git_stage_paths(repo_path: str, paths: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Stage files or directories (defaults to all changes)."""

    try:
        repo = _resolve_repo(repo_path)
        targets = list(paths) if paths else ["."]
        repo.git.add(*targets)
        return {"success": True, "staged": targets}
    except Exception as exc:
        return _failure(str(exc))


@register_tool(tags=_TOOL_TAGS)
def git_commit_changes(
    repo_path: str,
    message: str,
    author_name: Optional[str] = None,
    author_email: Optional[str] = None,
) -> Dict[str, Any]:
    """Commit staged changes with the provided message."""

    try:
        repo = _resolve_repo(repo_path)
        staged_diff = repo.git.diff("--cached")
        if not staged_diff.strip():
            return _failure("No staged changes to commit.")
        author = _actor(author_name, author_email)
        commit = repo.index.commit(message, author=author, committer=author)
        return {"success": True, "commit": commit.hexsha, "message": message}
    except Exception as exc:
        return _failure(str(exc))


@register_tool(tags=_TOOL_TAGS)
def git_get_uncommitted_changes(repo_path: str) -> Dict[str, Any]:
    """Return staged and unstaged diffs so an LLM can craft commit messages."""

    try:
        repo = _resolve_repo(repo_path)
        staged_diff = repo.git.diff("--cached")
        unstaged_diff = repo.git.diff()
        staged_names_raw = repo.git.diff("--name-only", "--cached").splitlines()
        unstaged_names_raw = repo.git.diff("--name-only").splitlines()
        staged_names = [name for name in staged_names_raw if name]
        unstaged_names = [name for name in unstaged_names_raw if name]
        untracked = [name for name in repo.git.ls_files("--others", "--exclude-standard").splitlines() if name]
        has_changes = bool(staged_names or unstaged_names or untracked or staged_diff.strip() or unstaged_diff.strip())

        return {
            "success": True,
            "has_changes": has_changes,
            "staged": {
                "files": staged_names,
                "diff": staged_diff,
            },
            "unstaged": {
                "files": unstaged_names,
                "diff": unstaged_diff,
            },
            "untracked_files": untracked,
        }
    except Exception as exc:
        return _failure(str(exc))


@register_tool(tags=_TOOL_TAGS)
def git_push_branch(
    repo_path: str,
    remote_name: str = "origin",
    branch_name: Optional[str] = None,
    set_upstream: bool = True,
) -> Dict[str, Any]:
    """Push the specified branch to a remote."""

    try:
        repo = _resolve_repo(repo_path)
        branch = branch_name or repo.active_branch.name
        args: List[str] = []
        if set_upstream:
            args.append("--set-upstream")
        args.extend([remote_name, f"{branch}:{branch}"])
        push_output = repo.git.push(*args)
        return {"success": True, "remote": remote_name, "branch": branch, "result": push_output.strip()}
    except Exception as exc:
        return _failure(str(exc))


@register_tool(tags=_TOOL_TAGS)
def git_create_pull_request(
    repository: str,
    head_branch: str,
    base_branch: str = "main",
    title: Optional[str] = None,
    body: Optional[str] = None,
    draft: bool = False,
    token_env_var: str = "GITHUB_TOKEN",
) -> Dict[str, Any]:
    """Create a GitHub pull request.

    Args:
        repository: Full repo name in the form "owner/name".
        head_branch: The branch containing your changes.
        base_branch: The branch you want to merge into.
        title: Optional PR title; defaults to "Merge <head> into <base>".
        body: Optional PR description.
        draft: Whether to open the PR as a draft.
        token_env_var: Environment variable containing a GitHub personal access token.
    """

    try:
        _require_github()
        token = os.getenv(token_env_var)
        if not token:
            return _failure(
                f"Environment variable '{token_env_var}' is not set. Provide a GitHub token to create PRs."
            )
        gh = Github(token)
        repo = gh.get_repo(repository)
        pr = repo.create_pull(
            title=title or f"Merge {head_branch} into {base_branch}",
            body=body or "",
            head=head_branch,
            base=base_branch,
            draft=draft,
        )
        return {
            "success": True,
            "url": pr.html_url,
            "number": pr.number,
            "title": pr.title,
            "state": pr.state,
        }
    except Exception as exc:
        return _failure(str(exc))