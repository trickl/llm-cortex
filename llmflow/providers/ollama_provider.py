"""Ollama provider with tool support verification."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

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
        self._explicit_tool_choice_supported = True
        self._tool_choice_warning_emitted = False

    def validate_tool_support(self) -> None:
        if not self._probe_tool_support():
            raise RuntimeError(
                "The configured Ollama model does not support tool/function calls. "
                "Choose or fine-tune a model with OpenAI-compatible tool calling."
            )
        # Determine whether the model honors explicit function targeting.
        self._explicit_tool_choice_supported = self._probe_tool_support(
            tool_choice={"type": "function", "function": {"name": "test_fn"}}
        )
        if not self._explicit_tool_choice_supported:
            print(
                "[OllamaProvider] Warning: model ignored explicit tool_choice requests; "
                "falling back to 'required' enforcement when only one tool is available."
            )

    def _probe_tool_support(self, tool_choice: Optional[Dict[str, Any]] = None) -> bool:
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

        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

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
        force_tool_call = kwargs.get("force_tool_call", False)
        if tools:
            payload["tools"] = tools
            normalized_choice = self._determine_tool_choice(
                tool_choice,
                tools,
                force_tool_call,
            )
            if normalized_choice is not None:
                payload["tool_choice"] = normalized_choice
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

    def _determine_tool_choice(
        self,
        requested_choice: Optional[Any],
        tools: Sequence[Dict[str, Any]],
        force_tool_call: bool,
    ) -> Optional[Any]:
        if requested_choice is None:
            if force_tool_call and len(tools) == 1:
                return "required"
            return None
        if isinstance(requested_choice, str):
            return requested_choice
        if isinstance(requested_choice, dict):
            if self._explicit_tool_choice_supported:
                return requested_choice
            if len(tools) == 1:
                return "required"
            if not self._tool_choice_warning_emitted:
                print(
                    "[OllamaProvider] Warning: Cannot target a specific tool when multiple options exist; "
                    "falling back to automatic selection."
                )
                self._tool_choice_warning_emitted = True
            return "auto"
        return None
