from __future__ import annotations

import pytest

from llmflow.runtime.errors import ToolError
from llmflow.runtime.syscall_registry import SyscallRegistry
from llmflow.runtime.syscalls import register_default_syscalls
from llmflow.runtime.syscalls.files import FileSyscalls
from llmflow.runtime.syscalls.git import GitSyscalls
from llmflow.runtime.syscalls.qlty import QltySyscalls


class _StubGitTools:
    def __init__(self):
        self.calls = {}

    def git_clone_repository(self, **kwargs):
        self.calls["clone"] = kwargs
        return {"success": True, "path": "/tmp/repo"}

    def git_create_branch(self, **kwargs):
        return {"success": True, **kwargs}

    def git_suggest_branch_name(self, **kwargs):
        return {"success": True, "branch_name": "fix/example", **kwargs}

    def git_switch_branch(self, **kwargs):
        return {"success": True, **kwargs}

    def git_stage_paths(self, **kwargs):
        return {"success": True, **kwargs}

    def git_commit_changes(self, **kwargs):
        return {"success": True, "commit": "deadbeef", **kwargs}

    def git_get_uncommitted_changes(self, **kwargs):
        return {"success": True, "staged": {}, "unstaged": {}, **kwargs}

    def git_push_branch(self, **kwargs):
        return {"success": True, **kwargs}

    def git_create_pull_request(self, **kwargs):
        return {"success": True, "url": "https://example/pr/1", **kwargs}


class _StubFileManager:
    def __init__(self):
        self.last_kwargs = None

    def list_files_in_tree(self, **kwargs):
        self.last_kwargs = kwargs
        return {"success": True, "files": []}

    def read_text_file(self, **kwargs):
        return {"success": True, "content": "hi"}


class _StubFileEditing:
    def overwrite_text_file(self, **kwargs):
        return {"success": True, **kwargs}

    def apply_text_rewrite(self, **kwargs):
        return {"success": True, **kwargs}


class _StubQltyTools:
    def __init__(self):
        self.calls = {}

    def qlty_list_issues(self, **kwargs):
        self.calls["list"] = kwargs
        return {"success": True, "data": []}

    def qlty_get_first_issue(self, **kwargs):
        self.calls["first"] = kwargs
        return {"success": True, "issue_found": False}


class _FailingGitTools(_StubGitTools):
    def git_clone_repository(self, **kwargs):  # type: ignore[override]
        return {"success": False, "error": "boom"}


def test_register_default_syscalls_registers_expected_names():
    registry = SyscallRegistry()
    git_tools = _StubGitTools()
    file_manager = _StubFileManager()
    file_editing = _StubFileEditing()
    qlty_tools = _StubQltyTools()

    register_default_syscalls(
        registry,
        logger=lambda msg: None,
        git_module=git_tools,
        file_manager_module=file_manager,
        file_editing_module=file_editing,
        qlty_module=qlty_tools,
    )

    expected = {
        "log",
        "cloneRepo",
        "createBranch",
        "suggestBranchName",
        "switchBranch",
        "stagePaths",
        "commitChanges",
        "getUncommittedChanges",
        "pushBranch",
        "createPullRequest",
        "listFilesInTree",
        "readTextFile",
        "overwriteTextFile",
        "applyTextRewrite",
        "qltyListIssues",
        "qltyGetFirstIssue",
    }

    assert expected.issubset(set(registry.to_dict().keys()))


def test_git_syscall_failure_raises_tool_error():
    registry = SyscallRegistry()
    module = GitSyscalls(git_module=_FailingGitTools())
    module.register(registry)

    with pytest.raises(ToolError):
        registry.get("cloneRepo")("https://example.com/repo.git")


def test_file_syscalls_forward_arguments():
    registry = SyscallRegistry()
    file_manager = _StubFileManager()
    module = FileSyscalls(file_manager_module=file_manager, file_editing_module=_StubFileEditing())
    module.register(registry)

    registry.get("listFilesInTree")("/repo", pattern="**/*.py", include_hidden=True)

    assert file_manager.last_kwargs == {
        "root_path": "/repo",
        "pattern": "**/*.py",
        "max_results": 200,
        "include_hidden": True,
        "follow_symlinks": False,
    }


def test_qlty_syscalls_propagate_payload():
    registry = SyscallRegistry()
    qlty_tools = _StubQltyTools()
    module = QltySyscalls(qlty_module=qlty_tools)
    module.register(registry)

    result = registry.get("qltyListIssues")("workspace", "project", max_pages=2)

    assert result["success"] is True
    assert qlty_tools.calls["list"]["max_pages"] == 2
