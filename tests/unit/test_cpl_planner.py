import json

import pytest

from llmflow.planning import CPLPlanRequest, CPLPlanner, CPLPlanningError


class DummyLLMClient:
    def __init__(self, response):
        if isinstance(response, str):
            response = {"role": "assistant", "content": response}
        self._response = response
        self.messages = None
        self.tools = None
        self.kwargs = None

    def generate(self, messages, tools=None, **kwargs):
        self.messages = messages
        self.tools = tools
        self.kwargs = kwargs
        return self._response


def test_planner_builds_prompt_and_returns_plan():
    client = DummyLLMClient(
        """plan {
            function main() : Void {
                syscall.log(\"hello\");
                return;
            }
        }
        """
    )
    planner = CPLPlanner(client, dsl_specification="SPEC CONTENT")
    request = CPLPlanRequest(
        task="Fix the reported lint issue",
        goals=["Diagnose the lint failure", "Apply a minimal patch"],
        context="Repository: example, Branch: main",
        allowed_syscalls=["log", "cloneRepo"],
    )

    result = planner.generate_plan(request)

    assert result.plan_source.lstrip().startswith("plan")
    assert result.metadata["allowed_syscalls"] == ["cloneRepo", "log"]
    assert client.messages[0]["role"] == "system"
    assert "cloneRepo" in client.messages[1]["content"]
    assert (
        client.tools is not None
        and client.tools[0]["function"]["name"] == "define_context_planning_language"
    )
    assert "SPEC CONTENT" in client.tools[0]["function"]["description"]
    expected_choice = {"type": "function", "function": {"name": "define_context_planning_language"}}
    assert client.kwargs.get("tool_choice") == expected_choice


def test_planner_extracts_plan_from_tool_call():
    plan_text = "plan { function main() : Void { syscall.log(\"hi\"); return; } }"
    response = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "define_context_planning_language",
                    "arguments": json.dumps({"cpl": plan_text, "notes": "ok"}),
                },
            }
        ],
    }
    client = DummyLLMClient(response)
    planner = CPLPlanner(client, dsl_specification="SPEC CONTENT")
    result = planner.generate_plan(CPLPlanRequest(task="Do work"))

    assert result.plan_source == plan_text
    assert result.metadata.get("planner_notes") == "ok"


def test_planner_rejects_invalid_response():
    client = DummyLLMClient("This is not a plan")
    planner = CPLPlanner(client, dsl_specification="SPEC")

    with pytest.raises(CPLPlanningError):
        planner.generate_plan(CPLPlanRequest(task="Summarize"))
