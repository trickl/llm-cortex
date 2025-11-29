"""Utility syscalls that do not depend on external services."""
from __future__ import annotations

from typing import Callable, Optional

from .base import BaseSyscallModule


class UtilitySyscalls(BaseSyscallModule):
    """Provide simple logging helpers to plan programs."""

    def __init__(self, logger: Optional[Callable[[str], None]] = None):
        self._logger = logger or print

    def log(self, message: str) -> None:
        self._logger(str(message))
        return None

    def get_syscalls(self):
        return {"log": self.log}


__all__ = ["UtilitySyscalls"]
