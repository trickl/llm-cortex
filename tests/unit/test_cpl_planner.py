import pytest

from llmflow.planning import JavaPlanRequest, JavaPlanner, JavaPlanningError


JAVA_PLAN = """
public class Plan {
    public void main() {
        syscall.log("hello");
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

    def structured_generate(self, *, messages, response_model, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        self.response_model = response_model
        return response_model(**self.payload)


def test_planner_builds_prompt_and_returns_plan():
    client = DummyLLMClient({"java": JAVA_PLAN})
    planner = JavaPlanner(client, specification="SPEC CONTENT")
    request = JavaPlanRequest(
        task="Fix the reported lint issue",
        goals=["Diagnose the lint failure", "Apply a minimal patch"],
        context="Repository: example, Branch: main",
        allowed_syscalls=["log", "cloneRepo"],
    )

    result = planner.generate_plan(request)

    assert result.plan_source.startswith("public class Plan")
    assert result.metadata["allowed_syscalls"] == ["cloneRepo", "log"]
    assert "available_tools" in client.messages[0]["content"]
    assert "cloneRepo" in client.messages[1]["content"]
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


def test_planner_raises_on_invalid_payload():
    client = DummyLLMClient({"java": "class Nope {}"})
    planner = JavaPlanner(client, specification="SPEC CONTENT")

    with pytest.raises(JavaPlanningError):
        planner.generate_plan(JavaPlanRequest(task="Summarize"))
