"""Shared runtime exceptions."""
from __future__ import annotations


class ToolError(Exception):
    """Exception type raised when a syscall reports a failure."""


__all__ = ["ToolError"]
