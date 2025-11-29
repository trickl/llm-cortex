"""Utility helpers for registering and retrieving plan syscalls."""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping, Optional


class SyscallRegistry:
    """Central registry for syscall implementations used by the runtime."""

    def __init__(self):
        self._syscalls: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]):
        if not callable(fn):
            raise TypeError(f"Syscall '{name}' must be callable")
        if name in self._syscalls:
            raise ValueError(f"Syscall '{name}' already registered")
        self._syscalls[name] = fn

    def register_many(self, mapping: Mapping[str, Callable[..., Any]]):
        for name, fn in mapping.items():
            self.register(name, fn)

    def register_module(
        self,
        module: Any,
        prefix: Optional[str] = None,
        include_private: bool = False,
    ):
        """Register all callable attributes from a module-like object."""

        names: Iterable[str] = dir(module)
        for attr_name in names:
            if not include_private and attr_name.startswith("_"):
                continue
            attr = getattr(module, attr_name)
            if callable(attr):
                key = f"{prefix}.{attr_name}" if prefix else attr_name
                self.register(key, attr)

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._syscalls[name]
        except KeyError as exc:
            raise KeyError(f"Syscall '{name}' is not registered") from exc

    def has(self, name: str) -> bool:
        return name in self._syscalls

    def to_dict(self) -> Dict[str, Callable[..., Any]]:
        return dict(self._syscalls)

    def clear(self):
        self._syscalls.clear()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Callable[..., Any]]) -> "SyscallRegistry":
        registry = cls()
        registry.register_many(mapping)
        return registry


__all__ = ["SyscallRegistry"]
