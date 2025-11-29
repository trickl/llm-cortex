"""Qlty API syscall module."""
from __future__ import annotations

from typing import Optional, Sequence

from llmflow.tools import tool_qlty as default_qlty_tools

from .base import BaseSyscallModule, ensure_tool_success


class QltySyscalls(BaseSyscallModule):
    """Wrap Qlty REST API helpers for plan programs."""

    def __init__(self, *, qlty_module=default_qlty_tools):
        self._qlty = qlty_module

    def _call(self, syscall_name: str, func, **kwargs):
        payload = func(**kwargs)
        return ensure_tool_success(syscall_name, payload)

    def list_issues(
        self,
        owner_key_or_id: str,
        project_key_or_id: str,
        categories: Optional[Sequence[str]] = None,
        levels: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[str]] = None,
        tools: Optional[Sequence[str]] = None,
        page_limit: Optional[int] = None,
        page_offset: Optional[int] = None,
        auto_paginate: bool = False,
        max_pages: int = 5,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        token_env_var: str = "QLTY_API_TOKEN",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        include_raw_pages: bool = False,
    ):
        return self._call(
            "qltyListIssues",
            self._qlty.qlty_list_issues,
            owner_key_or_id=owner_key_or_id,
            project_key_or_id=project_key_or_id,
            categories=categories,
            levels=levels,
            statuses=statuses,
            tools=tools,
            page_limit=page_limit,
            page_offset=page_offset,
            auto_paginate=auto_paginate,
            max_pages=max_pages,
            base_url=base_url,
            token=token,
            token_env_var=token_env_var,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            include_raw_pages=include_raw_pages,
        )

    def get_first_issue(
        self,
        owner_key_or_id: str,
        project_key_or_id: str,
        categories: Optional[Sequence[str]] = None,
        statuses: Optional[Sequence[str]] = None,
        levels: Optional[Sequence[str]] = None,
        tools: Optional[Sequence[str]] = None,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        token_env_var: str = "QLTY_API_TOKEN",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        return self._call(
            "qltyGetFirstIssue",
            self._qlty.qlty_get_first_issue,
            owner_key_or_id=owner_key_or_id,
            project_key_or_id=project_key_or_id,
            categories=categories,
            statuses=statuses,
            levels=levels,
            tools=tools,
            base_url=base_url,
            token=token,
            token_env_var=token_env_var,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    def get_syscalls(self):
        return {
            "qltyListIssues": self.list_issues,
            "qltyGetFirstIssue": self.get_first_issue,
        }


__all__ = ["QltySyscalls"]
