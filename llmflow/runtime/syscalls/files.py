"""File system and editing related syscalls."""
from __future__ import annotations

from typing import Optional

from llmflow.tools import tool_file_editing as default_file_editing
from llmflow.tools import tool_file_manager as default_file_manager

from .base import BaseSyscallModule, ensure_tool_success


class FileSyscalls(BaseSyscallModule):
    """Expose repository file helpers as plan syscalls."""

    def __init__(
        self,
        *,
        file_manager_module=default_file_manager,
        file_editing_module=default_file_editing,
    ):
        self._file_manager = file_manager_module
        self._file_editing = file_editing_module

    def _call(self, syscall_name: str, func, **kwargs):
        payload = func(**kwargs)
        return ensure_tool_success(syscall_name, payload)

    def list_files_in_tree(
        self,
        root_path: str,
        pattern: Optional[str] = None,
        max_results: int = 200,
        include_hidden: bool = False,
        follow_symlinks: bool = False,
    ):
        return self._call(
            "listFilesInTree",
            self._file_manager.list_files_in_tree,
            root_path=root_path,
            pattern=pattern,
            max_results=max_results,
            include_hidden=include_hidden,
            follow_symlinks=follow_symlinks,
        )

    def read_text_file(
        self,
        file_path: str,
        encoding: str = "utf-8",
        max_characters: Optional[int] = None,
    ):
        return self._call(
            "readTextFile",
            self._file_manager.read_text_file,
            file_path=file_path,
            encoding=encoding,
            max_characters=max_characters,
        )

    def overwrite_text_file(
        self,
        file_path: str,
        content: str,
        encoding: str = "utf-8",
        create_directories: bool = False,
        ensure_trailing_newline: bool = True,
    ):
        return self._call(
            "overwriteTextFile",
            self._file_editing.overwrite_text_file,
            file_path=file_path,
            content=content,
            encoding=encoding,
            create_directories=create_directories,
            ensure_trailing_newline=ensure_trailing_newline,
        )

    def apply_text_rewrite(
        self,
        file_path: str,
        original_snippet: str,
        new_snippet: str,
        occurrence: Optional[int] = None,
        encoding: str = "utf-8",
    ):
        return self._call(
            "applyTextRewrite",
            self._file_editing.apply_text_rewrite,
            file_path=file_path,
            original_snippet=original_snippet,
            new_snippet=new_snippet,
            occurrence=occurrence,
            encoding=encoding,
        )

    def get_syscalls(self):
        return {
            "listFilesInTree": self.list_files_in_tree,
            "readTextFile": self.read_text_file,
            "overwriteTextFile": self.overwrite_text_file,
            "applyTextRewrite": self.apply_text_rewrite,
        }


__all__ = ["FileSyscalls"]
