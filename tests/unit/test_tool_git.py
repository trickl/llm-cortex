"""Unit tests for git tool helpers."""

from pathlib import Path

import pytest

from llmflow.tools import tool_git

try:  # pragma: no cover - helper import for git-specific tests
    from git import Repo

    _GIT_AVAILABLE = True
except ImportError:  # pragma: no cover
    Repo = None  # type: ignore
    _GIT_AVAILABLE = False


@pytest.fixture(autouse=True)
def clear_project_repo_root(monkeypatch):
    monkeypatch.delenv("PROJECT_REPO_ROOT", raising=False)


def test_resolve_clone_destination_defaults_to_repo_root(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo-root"
    monkeypatch.setenv("PROJECT_REPO_ROOT", str(repo_root))

    resolved = tool_git._resolve_clone_destination(
        "git@github.com:owner/repo.git",
        None,
    )

    expected = (repo_root / "owner" / "repo").resolve()
    assert resolved == expected
    assert resolved.parent.exists()


def test_resolve_clone_destination_respects_relative_paths(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo-root"
    monkeypatch.setenv("PROJECT_REPO_ROOT", str(repo_root))

    resolved = tool_git._resolve_clone_destination(
        "git@github.com:owner/repo.git",
        "custom/workdir",
    )

    expected = (repo_root / "custom" / "workdir").resolve()
    assert resolved == expected


def test_resolve_clone_destination_without_env(monkeypatch, tmp_path):
    target = tmp_path / "manual"

    resolved = tool_git._resolve_clone_destination(
        "git@github.com:owner/repo.git",
        str(target),
    )

    assert resolved == target.resolve()
    assert resolved.parent == target.parent.resolve()


def _init_git_repo(repo_path: Path) -> str:
    if not _GIT_AVAILABLE:  # pragma: no cover - safety for environments without gitpython
        pytest.skip("GitPython is required for git_suggest_branch_name tests")
    repo_path.mkdir(parents=True, exist_ok=True)
    repo = Repo.init(repo_path)
    readme = repo_path / "README.md"
    readme.write_text("test repo", encoding="utf-8")
    repo.index.add([str(readme)])
    repo.index.commit("initial commit")
    return str(repo_path)


def test_git_create_branch_fails_when_exists(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_path = _init_git_repo(repo_dir)
    repo = Repo(repo_path)
    repo.create_head("existing-branch")

    result = tool_git.git_create_branch(
        repo_path=repo_path,
        branch_name="existing-branch",
    )

    assert result["success"] is False
    assert "already exists" in result["error"].lower()


def test_git_suggest_branch_name_includes_issue_slug(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_path = _init_git_repo(repo_dir)

    result = tool_git.git_suggest_branch_name(
        repo_path=repo_path,
        issue_reference="Lint Issue ABC-123",
    )

    assert result["success"] is True
    assert result["branch_name"].startswith("fix/issue-")
    assert "lint-issue-abc-123" in result["branch_name"]
    assert result["was_modified"] is False


def test_git_suggest_branch_name_handles_collisions(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_path = _init_git_repo(repo_dir)
    repo = Repo(repo_path)

    initial = tool_git.git_suggest_branch_name(
        repo_path=repo_path,
        issue_reference="Lint Issue ABC-123",
    )
    assert initial["success"] is True
    repo.create_head(initial["branch_name"])  # occupy the base branch name

    second = tool_git.git_suggest_branch_name(
        repo_path=repo_path,
        issue_reference="Lint Issue ABC-123",
    )

    assert second["success"] is True
    assert second["branch_name"].endswith("-1")
    assert second["was_modified"] is True


def test_git_suggest_branch_name_requires_repo_path() -> None:
    result = tool_git.git_suggest_branch_name(issue_reference="lint-1")

    assert result["success"] is False
    assert "repo_path" in result["error"].lower()
    assert result["retryable"] is True


def test_git_suggest_branch_name_missing_repo_is_retryable(tmp_path):
    missing_repo = tmp_path / "does-not-exist"

    result = tool_git.git_suggest_branch_name(
        repo_path=str(missing_repo),
        issue_reference="lint-2",
    )

    assert result["success"] is False
    assert "git_clone_repository" in result["error"].lower()
    assert result["retryable"] is True


def test_git_switch_branch_accepts_branch_alias(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_path = _init_git_repo(repo_dir)
    repo = Repo(repo_path)
    repo.create_head("feature", commit=repo.head.commit)

    result = tool_git.git_switch_branch(repo_path=repo_path, branch="feature")

    assert result["success"] is True
    assert result["branch"] == "feature"
    assert repo.active_branch.name == "feature"


def test_git_switch_branch_requires_target(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_path = _init_git_repo(repo_dir)

    result = tool_git.git_switch_branch(repo_path=repo_path)

    assert result["success"] is False
    assert "branch" in result["error"].lower()
    assert result["retryable"] is True
