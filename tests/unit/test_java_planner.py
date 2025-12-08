import io
import json
import logging
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from llmflow.logging_utils import LLM_LOGGER_NAME, PLAN_LOGGER_NAME
from llmflow.planning import JavaPlanRequest, JavaPlanner, JavaPlanningError
from llmflow.planning import java_planner as jp


JAVA_PLAN = """
public class Plan {
    public static void main(String[] args) throws Exception {
        PlanningToolStubs.log("hello");
    }
}
""".strip()


class DummyLLMClient:
    def __init__(self, payload):
        self.payload = payload
        self.messages = None
        self.kwargs = None

    def generate(self, **kwargs):
        self.messages = kwargs.get("messages")
        self.kwargs = kwargs
        arguments = json.dumps(self.payload)
        return {
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "define_java_plan",
                        "arguments": arguments,
                    },
                }
            ]
        }


class PlainFallbackRetryClient:
    def __init__(self, fallback_plan=JAVA_PLAN):
        self.fallback_plan = fallback_plan
        self.generate_calls: List[List[Dict[str, Any]]] = []

    def generate(self, *, messages, **kwargs):
        self.generate_calls.append(messages)
        if len(self.generate_calls) == 1:
            return {"content": ""}
        return {
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "define_java_plan",
                        "arguments": json.dumps({"java": self.fallback_plan}),
                    },
                }
            ]
        }


class PlainOnlyLLMClient:
    def __init__(self, fallback_plan=JAVA_PLAN):
        self.fallback_plan = fallback_plan
        self.generate_calls = 0
        self.messages = None

    def generate(self, *, messages, **kwargs):
        self.generate_calls += 1
        self.messages = messages
        return {"content": self.fallback_plan}


def _format_log_call(call):
    template = call.args[0]
    params = call.args[1:]
    try:
        return template % params if params else template
    except TypeError:
        return str(template)


def test_planner_builds_prompt_and_returns_plan():
    client = DummyLLMClient({"java": JAVA_PLAN})
    planner = JavaPlanner(client, specification="SPEC CONTENT")
    request = JavaPlanRequest(
        task="Fix the reported lint issue",
        context="Repository: example, Branch: main",
        tool_names=["log", "cloneRepo"],
        tool_stub_class_name="PlanningToolStubs",
    )

    result = planner.generate_plan(request)

    assert result.plan_source.startswith("public class Plan")
    assert result.metadata["allowed_tools"] == ["cloneRepo", "log"]
    assert "define_java_plan" in client.messages[0]["content"]
    user_prompt = client.messages[1]["content"]
    assert "Use the static methods on PlanningToolStubs" in user_prompt
    tool_choice = client.kwargs.get("tool_choice")
    assert tool_choice["function"]["name"] == "define_java_plan"
    tools = client.kwargs.get("tools")
    assert tools and tools[0]["function"]["name"] == "define_java_plan"


def test_planner_includes_planner_notes():
    client = DummyLLMClient({"java": JAVA_PLAN, "notes": "ok"})
    planner = JavaPlanner(client, specification="SPEC CONTENT")

    result = planner.generate_plan(JavaPlanRequest(task="Do work"))

    assert result.plan_source == JAVA_PLAN
    assert result.metadata.get("planner_notes") == "ok"


def test_planner_flattens_list_notes():
    client = DummyLLMClient({"java": JAVA_PLAN, "notes": ["first", "second"]})
    planner = JavaPlanner(client, specification="SPEC CONTENT")

    result = planner.generate_plan(JavaPlanRequest(task="Do work"))

    assert result.metadata.get("planner_notes") == "first\n\nsecond"


def test_planner_raises_when_java_missing():
    client = DummyLLMClient({"java": ""})
    planner = JavaPlanner(client, specification="SPEC CONTENT")

    with pytest.raises(JavaPlanningError):
        planner.generate_plan(JavaPlanRequest(task="Summarize"))


def test_planner_includes_tool_stubs_in_system_prompt():
    client = DummyLLMClient({"java": JAVA_PLAN})
    planner = JavaPlanner(client, specification="SPEC CONTENT")
    stub_source = "public final class DemoTools { public static void call() {} }"

    planner.generate_plan(
        JavaPlanRequest(
            task="Do work",
            tool_stub_source=stub_source,
            tool_stub_class_name="DemoTools",
        )
    )

    system_message = client.messages[0]["content"]
    assert "<tool_stubs>" in system_message
    assert "DemoTools" in system_message
    assert stub_source in system_message


def test_planner_includes_refinement_context_in_user_prompt():
    client = DummyLLMClient({"java": JAVA_PLAN})
    planner = JavaPlanner(client, specification="SPEC CONTENT")
    request = JavaPlanRequest(
        task="Do work",
        prior_plan_source="public class Bad { void main() {} }",
        compile_error_report="1. Missing syscall",
    )

    planner.generate_plan(request)

    user_message = client.messages[1]["content"]
    assert "Previous plan attempt" in user_message
    assert "Compile diagnostics" in user_message
    assert "Missing syscall" in user_message


def test_log_files_capture_request_and_plan(monkeypatch):
    monkeypatch.setattr(jp, "_current_timestamp_for_llm_log", lambda: "2025-01-01T00:00:00Z")

    plan_logger = logging.getLogger(PLAN_LOGGER_NAME)
    llm_logger = logging.getLogger(LLM_LOGGER_NAME)
    plan_stream = io.StringIO()
    llm_stream = io.StringIO()
    plan_handler = logging.StreamHandler(plan_stream)
    llm_handler = logging.StreamHandler(llm_stream)
    plan_handler.setFormatter(logging.Formatter("%(message)s"))
    llm_handler.setFormatter(logging.Formatter("%(message)s"))
    old_plan_handlers = plan_logger.handlers[:]
    old_llm_handlers = llm_logger.handlers[:]
    old_plan_level = plan_logger.level
    old_llm_level = llm_logger.level
    plan_logger.handlers = [plan_handler]
    llm_logger.handlers = [llm_handler]
    plan_logger.setLevel(logging.INFO)
    llm_logger.setLevel(logging.INFO)

    client = DummyLLMClient({"java": JAVA_PLAN})
    planner = JavaPlanner(client, specification="SPEC CONTENT")

    try:
        planner.generate_plan(JavaPlanRequest(task="Do work", metadata={"plan_id": "plan-logs"}))
    finally:
        plan_logger.handlers = old_plan_handlers
        llm_logger.handlers = old_llm_handlers
        plan_logger.setLevel(old_plan_level)
        llm_logger.setLevel(old_llm_level)

    plan_output = plan_stream.getvalue()
    assert "planner_plan_source plan_id=plan-logs" in plan_output
    assert "public class Plan" in plan_output

    llm_output = llm_stream.getvalue()
    assert "[2025-01-01T00:00:00Z] Request - Role System - Input Token Count" in llm_output
    assert "[2025-01-01T00:00:00Z] Request - Role User - Input Token Count" in llm_output
    assert "[2025-01-01T00:00:00Z] Request - Role Assistant - Output Token Count" in llm_output
    assert llm_output.count(">>>") == 3
    assert "public class Plan" in llm_output


def test_planner_defaults_to_plain_generation():
    client = PlainOnlyLLMClient()
    planner = JavaPlanner(client, specification="SPEC CONTENT")

    result = planner.generate_plan(JavaPlanRequest(task="Do work"))

    assert result.plan_source == JAVA_PLAN
    assert client.generate_calls == 1


def test_plain_fallback_sends_retry_prompt():
    client = PlainFallbackRetryClient()
    planner = JavaPlanner(client, specification="SPEC CONTENT")

    result = planner.generate_plan(JavaPlanRequest(task="Handle retries"))

    assert result.plan_source == JAVA_PLAN
    assert len(client.generate_calls) == 2
    first_followup = client.generate_calls[0][-1]["content"]
    second_followup = client.generate_calls[1][-1]["content"]
    assert "Output the Java class" in first_followup
    assert "Your prior response was empty or malformed" in second_followup


def test_plain_prompt_mentions_helper_focus():
    client = PlainOnlyLLMClient()
    planner = JavaPlanner(client, specification="SPEC CONTENT")
    helper_focus = {
        "function": "hasOpenIssues",
        "comment": "Stub: determine if repository has open issues",
        "message": "Helper 'hasOpenIssues' is a placeholder.",
    }
    request = JavaPlanRequest(
        task="Implement helper method 'hasOpenIssues'.",
        metadata={"helper_focus": helper_focus},
    )

    planner.generate_plan(request)

    assert client.messages and len(client.messages) >= 3
    plain_prompt = client.messages[-1]["content"]
    assert "hasOpenIssues" in plain_prompt
    assert "Keep main()" in plain_prompt
    assert "Stub: determine if repository has open issues" in plain_prompt


def test_planner_logs_llm_request_payload(monkeypatch):
    mock_llm_info = MagicMock()
    monkeypatch.setattr(jp.llm_logger, "info", mock_llm_info)

    client = DummyLLMClient({"java": JAVA_PLAN})
    planner = JavaPlanner(client, specification="SPEC CONTENT")

    planner.generate_plan(JavaPlanRequest(task="Capture request"))

    info_messages = [_format_log_call(call) for call in mock_llm_info.call_args_list]
    assert any("Role System - Input Token Count" in message for message in info_messages)
    assert any("Role User - Input Token Count" in message for message in info_messages)


def test_extract_plan_from_nested_tool_payload():
    payload = {"define_java_plan": json.dumps({"java": JAVA_PLAN})}

    result = jp._extract_plan_from_dict(payload)

    assert result == JAVA_PLAN
