"""Ollama provider with tool support verification."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import requests

from .base import LLMProviderInterface


class OllamaProvider(LLMProviderInterface):
    """Provider that communicates with a local Ollama server."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        default_options: Optional[Dict[str, Any]] = None,
        tool_check_timeout: int = 30,
    ):
        if not model_name:
            raise ValueError("OllamaProvider requires a model name.")

        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.chat_endpoint = f"{self.base_url}/api/chat"
        self.default_options = default_options or {}
        self.tool_check_timeout = tool_check_timeout

    def validate_tool_support(self) -> None:
        if not self._probe_tool_support():
            raise RuntimeError(
                "The configured Ollama model does not support tool/function calls. "
                "Choose or fine-tune a model with OpenAI-compatible tool calling."
            )

    def _probe_tool_support(self) -> bool:
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": "Call the function test_fn with no arguments.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "test_fn",
                        "description": "A dummy test function",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "stream": False,
        }

        try:
            response = requests.post(
                self.chat_endpoint, json=payload, timeout=self.tool_check_timeout
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Failed to verify Ollama tool support. Ensure the server is running and reachable."
            ) from exc

        tool_calls = data.get("message", {}).get("tool_calls")
        return isinstance(tool_calls, list) and len(tool_calls) > 0

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model or self.model_name,
            "messages": messages,
            "stream": False,
        }

        options = self.default_options.copy()
        if "temperature" in kwargs:
            options["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            options["num_predict"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            options["top_p"] = kwargs["top_p"]
        if options:
            payload["options"] = options

        tool_choice = kwargs.get("tool_choice")
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
            else:
                payload.setdefault("tool_choice", "auto")

        timeout = kwargs.get("timeout", 60)

        try:
            response = requests.post(
                self.chat_endpoint, json=payload, timeout=timeout
            )
            response.raise_for_status()
            response_data = response.json()
            print(f"[OllamaProvider] Raw response: {json.dumps(response_data, indent=2)}")
        except requests.RequestException as exc:
            error_message = f"Ollama API request failed: {exc}"
            if exc.response is not None:
                try:
                    error_payload = exc.response.json()
                    error_message += f" (Details: {error_payload})"
                except ValueError:
                    error_message += f" (Raw response: {exc.response.text})"
            print(f"[OllamaProvider] Error: {error_message}")
            return {"role": "assistant", "content": f"Error: {error_message}"}

        message_block = response_data.get("message", {})
        content = message_block.get("content")
        tool_calls = message_block.get("tool_calls")

        if not isinstance(content, (str, type(None))):
            content = str(content)
        if not isinstance(tool_calls, (list, type(None))):
            tool_calls = None

        return {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        }
