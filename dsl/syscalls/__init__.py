"""Structured syscall collections for CPL plans."""
from __future__ import annotations

from typing import Iterable, Optional

from ..syscall_registry import SyscallRegistry

from .base import BaseSyscallModule, ToolResult, register_modules
from .files import FileSyscalls
from .git import GitSyscalls
from .qlty import QltySyscalls
from .subgoal import SubgoalSyscalls
from .utility import UtilitySyscalls


def _maybe_kwargs(**candidates):
    return {key: value for key, value in candidates.items() if value is not None}


def register_default_syscalls(
    registry: SyscallRegistry,
    *,
    logger=None,
    git_module=None,
    file_manager_module=None,
    file_editing_module=None,
    qlty_module=None,
    subgoal_module=None,
) -> SyscallRegistry:
    """Register the standard syscall set on ``registry``."""

    modules: Iterable[BaseSyscallModule] = (
        UtilitySyscalls(logger=logger),
        FileSyscalls(**_maybe_kwargs(
            file_manager_module=file_manager_module,
            file_editing_module=file_editing_module,
        )),
        GitSyscalls(**_maybe_kwargs(git_module=git_module)),
        QltySyscalls(**_maybe_kwargs(qlty_module=qlty_module)),
        SubgoalSyscalls(**_maybe_kwargs(subgoal_module=subgoal_module)),
    )
    return register_modules(registry, modules)


def build_default_syscall_registry(**kwargs) -> SyscallRegistry:
    """Create a :class:`SyscallRegistry` populated with default modules."""

    registry = SyscallRegistry()
    return register_default_syscalls(registry, **kwargs)


__all__ = [
    "BaseSyscallModule",
    "ToolResult",
    "UtilitySyscalls",
    "FileSyscalls",
    "GitSyscalls",
    "QltySyscalls",
    "SubgoalSyscalls",
    "register_default_syscalls",
    "build_default_syscall_registry",
]
