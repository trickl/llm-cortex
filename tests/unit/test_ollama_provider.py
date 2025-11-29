import pytest

from llmflow.providers.ollama_provider import OllamaProvider


class DummyOllamaProvider(OllamaProvider):
    """Subclass that skips network validation for unit tests."""

    def __init__(self):
        super().__init__(model_name="granite4:3b")

    def validate_tool_support(self) -> None:  # pragma: no cover - unused in test
        return


@pytest.fixture
def provider():
    instance = DummyOllamaProvider()
    return instance


def test_determine_tool_choice_respects_explicit(provider, monkeypatch):
    tools = [
        {
            "type": "function",
            "function": {"name": "define_context_planning_language"},
        }
    ]
    provider._explicit_tool_choice_supported = True
    choice = provider._determine_tool_choice(
        {"type": "function", "function": {"name": "define_context_planning_language"}},
        tools,
        False,
    )
    assert isinstance(choice, dict)
    assert choice["function"]["name"] == "define_context_planning_language"


def test_determine_tool_choice_falls_back_to_required(provider):
    tools = [
        {
            "type": "function",
            "function": {"name": "define_context_planning_language"},
        }
    ]
    provider._explicit_tool_choice_supported = False
    choice = provider._determine_tool_choice(
        {"type": "function", "function": {"name": "define_context_planning_language"}},
        tools,
        True,
    )
    assert choice == "required"
