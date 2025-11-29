"""Tests for the ActionHandler metadata and failure classification helpers."""

from __future__ import annotations

import json
from typing import Dict, Any

import pytest

pytest.skip("ActionHandler module has been retired in favor of Java planning", allow_module_level=True)

from llmflow.core.action_handler import ActionHandler
from llmflow.tools.tool_decorator import register_tool


@register_tool(tags=["tests", "action_handler"])
def action_handler_unit_test_tool(text: str) -> Dict[str, Any]:
    """Echo back the provided text wrapped in a success dict for testing."""
    return {"success": True, "echo": text}


@register_tool(tags=["tests", "action_handler"])
def action_handler_unit_test_missing_dep() -> Dict[str, Any]:
    """Simulate a missing dependency by raising ModuleNotFoundError."""
    raise ModuleNotFoundError("GitPython is not installed")


def _build_tool_call(name: str, arguments: Dict[str, Any], call_id: str = "call_1") -> Dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def test_execute_tool_calls_includes_metadata_for_success():
    handler = ActionHandler()
    tool_calls = [_build_tool_call("action_handler_unit_test_tool", {"text": "ping"}, call_id="call_success")]

    results = handler.execute_tool_calls(tool_calls)

    assert len(results) == 1
    result = results[0]
    assert result["metadata"]["success"] is True
    assert result["metadata"]["fatal"] is False
    assert json.loads(result["content"]) == {"success": True, "echo": "ping"}


def test_execute_tool_calls_flags_fatal_missing_dependency():
    handler = ActionHandler()
    tool_calls = [_build_tool_call("action_handler_unit_test_missing_dep", {}, call_id="call_missing_dep")]

    results = handler.execute_tool_calls(tool_calls)

    assert len(results) == 1
    result = results[0]
    metadata = result["metadata"]
    assert metadata["success"] is False
    assert metadata["fatal"] is True
    assert metadata["retryable"] is False
    assert "GitPython" in metadata["error"]