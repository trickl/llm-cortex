import types

import pytest

from llmflow.runtime.syscall_registry import SyscallRegistry


def test_register_and_get_syscall():
    registry = SyscallRegistry()
    registry.register("echo", lambda value: value)

    fn = registry.get("echo")
    assert fn("hi") == "hi"


def test_register_module_skips_private_attributes():
    module = types.SimpleNamespace(visible=lambda: "ok", _hidden=lambda: "nope")
    registry = SyscallRegistry()

    registry.register_module(module)

    assert registry.has("visible")
    assert not registry.has("_hidden")


def test_from_mapping_rejects_duplicates():
    registry = SyscallRegistry.from_mapping({"one": lambda: 1})

    with pytest.raises(ValueError):
        registry.register("one", lambda: 2)

    assert registry.get("one")() == 1
