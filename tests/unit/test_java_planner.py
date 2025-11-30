import io
import json
import logging
from unittest.mock import MagicMock

import pytest

from llmflow.planning import JavaPlanRequest, JavaPlanner, JavaPlanningError
from llmflow.logging_utils import LLM_LOGGER_NAME, PLAN_LOGGER_NAME


JAVA_PLAN = """
public class Plan {
    public void main() {
        PlanningToolStubs.log("hello");
        return;
    }
}
""".strip()


class DummyLLMClient:
    def __init__(self, payload):
        self.payload = payload
        self.messages = None
        self.kwargs = None
        self.response_model = None
        self.provider = type("Provider", (), {"max_retries": 0})()

    def structured_generate(self, *, messages, response_model, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        self.response_model = response_model
        return response_model(**self.payload)

    def generate(self, **kwargs):  # pragma: no cover - fallback stub
        raise JavaPlanningError("Planner returned invalid payload")


class FailingLLMClient:
    def __init__(self, exception, fallback_plan=JAVA_PLAN):
        self.exception = exception
        self.fallback_plan = fallback_plan
        self.provider = type("Provider", (), {"max_retries": 1})()
        self.kwargs = None

    def structured_generate(self, **kwargs):
        self.kwargs = kwargs
        raise self.exception

    def generate(self, **kwargs):
        return {"content": self.fallback_plan}


class FakeCompletion:
    def __init__(self, payload):
        self._payload = payload

    def model_dump_json(self, indent=2):
        return json.dumps(self._payload, indent=indent, ensure_ascii=False)


class FakeAttempt:
    def __init__(self, completion):
        self.completion = completion


class FakeInstructorException(Exception):
    def __init__(self, completions):
        super().__init__("validation error")
        self.failed_attempts = [FakeAttempt(completion) for completion in completions]
        self.last_completion = completions[-1] if completions else None


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
        goals=["Diagnose the lint failure", "Apply a minimal patch"],
        context="Repository: example, Branch: main",
        tool_names=["log", "cloneRepo"],
        tool_stub_class_name="PlanningToolStubs",
    )

    result = planner.generate_plan(request)

    assert result.plan_source.startswith("public class Plan")
    assert result.metadata["allowed_tools"] == ["cloneRepo", "log"]
    assert "available_tools" in client.messages[0]["content"]
    user_prompt = client.messages[1]["content"]
    assert "Use the static methods on PlanningToolStubs" in user_prompt
    assert "Available planning tools" not in user_prompt
    tool_choice = client.kwargs.get("tool_choice")
    assert tool_choice["function"]["name"] == "define_java_plan"
    tools = client.kwargs.get("tools")
    assert tools and tools[0]["function"]["name"] == "define_java_plan"
    assert client.kwargs.get("max_retries") == 2
    log_context = client.kwargs.get("log_context", {})
    assert log_context.get("prefix", "").startswith("plan_id=")
    assert callable(log_context.get("completion_logger"))


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


def test_planner_logs_instructor_failure(monkeypatch):
    from llmflow.planning import java_planner as jp

    monkeypatch.setattr(jp, "InstructorRetryException", FakeInstructorException)
    mock_plan_warning = MagicMock()
    mock_plan_info = MagicMock()
    mock_llm_error = MagicMock()
    monkeypatch.setattr(jp.plan_logger, "warning", mock_plan_warning)
    monkeypatch.setattr(jp.plan_logger, "info", mock_plan_info)
    monkeypatch.setattr(jp.llm_logger, "error", mock_llm_error)
    completion = FakeCompletion({"java": JAVA_PLAN, "notes": "bad"})
    exception = FakeInstructorException([completion])
    client = FailingLLMClient(exception)
    planner = JavaPlanner(client, specification="SPEC CONTENT")

    planner.generate_plan(JavaPlanRequest(task="Do work", metadata={"plan_id": "plan-123"}))

    plan_messages = [_format_log_call(call) for call in mock_plan_warning.call_args_list]
    assert any("planner_structured_payload plan_id=plan-123" in message for message in plan_messages)

    plan_info_messages = [_format_log_call(call) for call in mock_plan_info.call_args_list]
    assert any("planner_structured_plan_body plan_id=plan-123" in message for message in plan_info_messages)
    assert any("PlanningToolStubs.log" in message for message in plan_info_messages)

    llm_messages = [_format_log_call(call) for call in mock_llm_error.call_args_list]
    assert any("planner_structured_payload_dump plan_id=plan-123" in message for message in llm_messages)
    assert any('"notes": "bad"' in message for message in llm_messages)


def test_plan_logger_decodes_json_wrapped_plan(monkeypatch):
    from llmflow.planning import java_planner as jp

    mock_plan_info = MagicMock()
    mock_llm_error = MagicMock()
    monkeypatch.setattr(jp.plan_logger, "info", mock_plan_info)
    monkeypatch.setattr(jp.llm_logger, "error", mock_llm_error)

    completion = {"define_java_plan": "public class Planner {\\n    void run() {\\n        // noop\\n    }\\n}"}

    jp._log_structured_completion_payload(
        plan_id="plan-json",
        attempt_label="1",
        completion=completion,
        stage="parse_error",
    )

    plan_info_messages = [_format_log_call(call) for call in mock_plan_info.call_args_list]
    matching = [message for message in plan_info_messages if "planner_structured_plan_body" in message]
    assert matching, "Plan logger did not emit decoded plan body."
    header, _, body = matching[0].partition("\n")
    assert body.strip().startswith("public class Planner")
    assert "\\n" not in body
    assert "\n" in body


def test_plan_logger_formats_single_line_plan(monkeypatch):
    from llmflow.planning import java_planner as jp

    mock_plan_info = MagicMock()
    mock_llm_error = MagicMock()
    monkeypatch.setattr(jp.plan_logger, "info", mock_plan_info)
    monkeypatch.setattr(jp.llm_logger, "error", mock_llm_error)

    completion = {"java": "public class Planner { private void run() {} }"}

    jp._log_structured_completion_payload(
        plan_id="plan-single-line",
        attempt_label="1",
        completion=completion,
        stage="parse_error",
    )

    plan_info_messages = [_format_log_call(call) for call in mock_plan_info.call_args_list]
    matching = [message for message in plan_info_messages if "planner_structured_plan_body" in message]
    assert matching
    _, _, body = matching[0].partition("\n")
    assert "\n" in body, "Expected formatter to insert newlines"
    assert "public class Planner" in body


def test_plan_logger_ignores_non_plan_strings(monkeypatch):
    from llmflow.planning import java_planner as jp

    mock_plan_info = MagicMock()
    mock_llm_error = MagicMock()
    monkeypatch.setattr(jp.plan_logger, "info", mock_plan_info)
    monkeypatch.setattr(jp.llm_logger, "error", mock_llm_error)

    plan_payload = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            "define_java_plan": "public class Planner {\n    void run() {}\n}"
                        }
                    )
                },
            }
        ]
    }

    jp._log_structured_completion_payload(
        plan_id="plan-nested",
        attempt_label="1",
        completion=plan_payload,
        stage="parse_error",
    )

    plan_info_messages = [_format_log_call(call) for call in mock_plan_info.call_args_list]
    matching = [message for message in plan_info_messages if "planner_structured_plan_body" in message]
    assert matching, "Expected nested plan to be logged"
    _, _, body = matching[0].partition("\n")
    assert "finish_reason" not in body
    assert "public class Planner" in body


def test_log_files_capture_request_and_plan(monkeypatch):
    from llmflow.planning import java_planner as jp

    monkeypatch.setattr(jp, "InstructorRetryException", FakeInstructorException)
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

    completion = FakeCompletion({"java": JAVA_PLAN})
    exception = FakeInstructorException([completion])
    client = FailingLLMClient(exception)
    planner = JavaPlanner(client, specification="SPEC CONTENT")

    try:
        planner.generate_plan(JavaPlanRequest(task="Do work", metadata={"plan_id": "plan-logs"}))
    finally:
        plan_logger.handlers = old_plan_handlers
        llm_logger.handlers = old_llm_handlers
        plan_logger.setLevel(old_plan_level)
        llm_logger.setLevel(old_llm_level)

    plan_output = plan_stream.getvalue()
    assert "planner_structured_plan_body plan_id=plan-logs" in plan_output
    assert "public class Plan" in plan_output
    assert "PlanningToolStubs.log" in plan_output

    llm_output = llm_stream.getvalue()
    assert "[2025-01-01T00:00:00Z] Request - Role System - Input Token Count" in llm_output
    assert "[2025-01-01T00:00:00Z] Request - Role User - Input Token Count" in llm_output
    assert "[2025-01-01T00:00:00Z] Request - Role Assistant - Output Token Count" in llm_output
    assert llm_output.count(">>>") == 3
    assert "public class Plan" in llm_output


def test_planner_allows_retry_override():
    client = DummyLLMClient({"java": JAVA_PLAN})
    planner = JavaPlanner(client, specification="SPEC CONTENT", structured_max_retries=2)

    planner.generate_plan(JavaPlanRequest(task="Do work"))

    log_context = client.kwargs.get("log_context", {})
    assert client.kwargs.get("max_retries") == 2
    assert log_context.get("logger_name") == PLAN_LOGGER_NAME
    assert callable(log_context.get("completion_logger"))


def test_planner_env_var_controls_retry(monkeypatch):
    client = DummyLLMClient({"java": JAVA_PLAN})
    monkeypatch.setenv("LLMFLOW_PLANNER_STRUCTURED_MAX_RETRIES", "5")
    planner = JavaPlanner(client, specification="SPEC CONTENT")

    planner.generate_plan(JavaPlanRequest(task="Do work"))

    assert client.kwargs.get("max_retries") == 5


def test_planner_logs_llm_request_payload(monkeypatch):
    from llmflow.planning import java_planner as jp

    mock_llm_info = MagicMock()
    monkeypatch.setattr(jp.llm_logger, "info", mock_llm_info)

    client = DummyLLMClient({"java": JAVA_PLAN})
    planner = JavaPlanner(client, specification="SPEC CONTENT")

    planner.generate_plan(JavaPlanRequest(task="Capture request"))

    info_messages = [_format_log_call(call) for call in mock_llm_info.call_args_list]
    assert any("Role System - Input Token Count" in message for message in info_messages)
    assert any("Role User - Input Token Count" in message for message in info_messages)
