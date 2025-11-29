"""Shared helpers for implementing structured plan syscalls."""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping

from ..errors import ToolError
from ..syscall_registry import SyscallRegistry

ToolResult = Dict[str, Any]


def ensure_tool_success(syscall_name: str, payload: Any) -> Any:
    """Raise :class:`ToolError` when a tool payload reports failure."""

    if isinstance(payload, Mapping):
        success = payload.get("success")
        if success is False or (success is None and "error" in payload):
            message = payload.get("error") or f"{syscall_name} failed"
            raise ToolError(message)
    return payload


class BaseSyscallModule:
    """Base class for grouping syscall implementations."""

    def get_syscalls(self) -> Dict[str, Callable[..., Any]]:
        raise NotImplementedError

    def register(self, registry: SyscallRegistry) -> None:
        for name, fn in self.get_syscalls().items():
            registry.register(name, fn)


def register_modules(registry: SyscallRegistry, modules: Iterable[BaseSyscallModule]) -> SyscallRegistry:
    for module in modules:
        module.register(registry)
    return registry


__all__ = ["BaseSyscallModule", "ToolResult", "ensure_tool_success", "register_modules"]
