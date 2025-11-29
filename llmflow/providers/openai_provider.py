"""OpenAI provider implementation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import LLMProviderInterface

try:
    from openai import OpenAI
    from openai.types.chat.chat_completion import ChatCompletion
    OPENAI_SDK_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - handled at runtime
    OPENAI_SDK_AVAILABLE = False
    OPENAI_IMPORT_ERROR = exc


class OpenAIProvider(LLMProviderInterface):
    """Provider for OpenAI Chat Completions API."""

    def __init__(self, api_key: str):
        if not OPENAI_SDK_AVAILABLE:
            raise ImportError(
                "OpenAI SDK not found. Install via 'pip install openai'."
            ) from OPENAI_IMPORT_ERROR
        if not api_key:
            raise ValueError("OpenAIProvider requires a non-empty API key.")

        self.client = OpenAI(api_key=api_key)

    def validate_tool_support(self) -> None:
        """OpenAI chat models natively support tool calls."""
        return

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 2048)
        top_p = kwargs.get("top_p", 1.0)

        request_params: Dict[str, Any] = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = kwargs.get("tool_choice", "auto")

        try:
            chat_completion: ChatCompletion = self.client.chat.completions.create(
                **request_params
            )
        except Exception as exc:  # pragma: no cover - network/API errors
            print(f"[OpenAIProvider] Error: {exc}")
            return {
                "role": "assistant",
                "content": f"Error: OpenAI API error: {exc}",
            }

        if not chat_completion.choices:
            return {
                "role": "assistant",
                "content": "Error: OpenAI API returned no choices.",
            }

        choice = chat_completion.choices[0]
        response_message = choice.message
        response_content = response_message.content
        response_tool_calls = None

        if response_message.tool_calls:
            response_tool_calls = []
            for tool_call in response_message.tool_calls:
                response_tool_calls.append(
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                )

        return {
            "role": "assistant",
            "content": response_content,
            "tool_calls": response_tool_calls,
        }
