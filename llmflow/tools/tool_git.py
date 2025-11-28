"""Git tooling for repository automation.

These helper tools expose a subset of common git workflows—cloning, branching,
staging/committing, retrieving diffs for commit messages, pushing branches, and
opening pull requests—so agents can manage source control tasks end-to-end.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

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

_INVALID_BRANCH_CHARS = re.compile(r"[^0-9a-zA-Z._/-]+")
_DUPLICATE_SEPARATORS = re.compile(r"[._/-]{2,}")


def _extract_repo_slug(repo_url: str) -> Optional[str]:
    trimmed = repo_url.strip()
    if not trimmed:
        return None
    trimmed = trimmed.rstrip("/")
    if trimmed.endswith(".git"):
        trimmed = trimmed[:-4]
    if trimmed.startswith("git@") and ":" in trimmed:
        _, remainder = trimmed.split(":", 1)
        return remainder.strip("/") or None
    parsed = urlparse(trimmed)
    if parsed.scheme:
        path = parsed.path.strip("/")
        if path:
            return path
    if "/" in trimmed:
        return trimmed.strip("/")
    return trimmed or None


def _slug_to_subpath(slug: Optional[str]) -> Path:
    if not slug:
        return Path("repository")
    parts = [part for part in slug.split("/") if part and part not in (".", "..")]
    return Path(*parts) if parts else Path("repository")


def _slugify_branch_component(value: Optional[str]) -> str:
    if not value:
        return "update"
    text = str(value).strip().lower()
    if not text:
        return "update"
    text = text.replace(" ", "-")
    text = text.replace(":", "-")
    text = text.replace("/", "-")
    text = text.replace("\\", "-")
    text = _INVALID_BRANCH_CHARS.sub("-", text)
    text = _DUPLICATE_SEPARATORS.sub("-", text)
    text = text.strip("./-")
    return text or "update"


def _resolve_clone_destination(repo_url: str, destination: Optional[str]) -> Path:
    repo_root_env = os.getenv("PROJECT_REPO_ROOT")
    if repo_root_env:
        root_path = Path(repo_root_env).expanduser().resolve()
        root_path.mkdir(parents=True, exist_ok=True)
        if destination:
            dest_path = Path(destination)
            if dest_path.is_absolute():
                relative_subpath = _slug_to_subpath(_extract_repo_slug(repo_url))
            else:
                relative_subpath = dest_path
        else:
            relative_subpath = _slug_to_subpath(_extract_repo_slug(repo_url))
        target_dir = (root_path / relative_subpath).resolve()
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        return target_dir

    if destination:
        target_dir = Path(destination).expanduser().resolve()
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        return target_dir

    return Path(tempfile.mkdtemp(prefix="llmflow-git-"))


def _failure(message: str, retryable: bool = False) -> Dict[str, Any]:
    return {"success": False, "error": message, "retryable": retryable}


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
        target_dir = _resolve_clone_destination(repo_url, destination)
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
            return _failure(
                f"Branch '{branch_name}' already exists. Choose a different name."
            )

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
def git_suggest_branch_name(
    repo_path: Optional[str] = None,
    issue_reference: Optional[str] = None,
    prefix: str = "fix/issue-",
    max_suffix_attempts: int = 50,
) -> Dict[str, Any]:
    """Suggest a unique branch name based on the provided issue reference.

    The helper sanitizes the issue reference (slug/id) into a git-safe component,
    prefixes it (default ``fix/issue-``), and appends a numeric suffix when the
    name already exists in the repository.
    """

    if not repo_path:
        return _failure(
            "Provide the 'repo_path' pointing to your cloned repository when calling git_suggest_branch_name.",
            retryable=True,
        )

    try:
        repo = _resolve_repo(repo_path)
        existing = {head.name for head in repo.branches}
        slug = _slugify_branch_component(issue_reference)
        base_name = f"{prefix}{slug}"
        candidate = base_name
        suffix = 1
        while candidate in existing and suffix <= max_suffix_attempts:
            candidate = f"{base_name}-{suffix}"
            suffix += 1

        if candidate in existing:
            return _failure(
                "Unable to find a unique branch name. Consider adjusting the prefix or slug."
            )

        return {
            "success": True,
            "branch_name": candidate,
            "base_branch_name": base_name,
            "was_modified": candidate != base_name,
        }
    except RuntimeError as exc:
        message = (
            f"{exc}. Ensure the repository is cloned via git_clone_repository and reuse its 'path' value as 'repo_path' when calling git_suggest_branch_name."
        )
        return _failure(message, retryable=True)
    except Exception as exc:
        return _failure(str(exc))


@register_tool(tags=_TOOL_TAGS)
def git_switch_branch(
    repo_path: str,
    branch_name: Optional[str] = None,
    branch: Optional[str] = None,
) -> Dict[str, Any]:
    """Check out the given branch.

    Args:
        repo_path: Local repository path to operate on.
        branch_name: Target branch name (preferred parameter).
        branch: Backwards-compatible alias used by earlier prompts.
    """

    target_branch = branch_name or branch
    if not target_branch:
        return _failure(
            "Missing 'branch_name' (or 'branch') argument for git_switch_branch.",
            retryable=True,
        )

    try:
        repo = _resolve_repo(repo_path)
        repo.git.checkout(target_branch)
        return {"success": True, "branch": target_branch, "commit": repo.head.commit.hexsha}
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