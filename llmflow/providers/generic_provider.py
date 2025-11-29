"""Generic HTTP provider implementation."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

import requests

from .base import LLMProviderInterface


PathComponent = Union[str, int]


class GenericProvider(LLMProviderInterface):
    """Configurable provider for custom HTTP-compatible LLM gateways."""

    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get("api_key")
        self.endpoint = config.get("endpoint")
        self.headers = config.get("headers", {}).copy()
        self.payload_template = config.get("payload_template", {}).copy()
        self.response_mapping = config.get(
            "response_mapping",
            {
                "content_path": ["choices", 0, "message", "content"],
                "error_path": ["error", "message"],
                "tool_calls_path": ["choices", 0, "message", "tool_calls"],
            },
        )
        self.tool_capable = config.get("tool_capable", False)

        if not self.endpoint:
            raise ValueError("GenericProvider requires an 'endpoint' in its config.")
        if not self.tool_capable:
            raise ValueError(
                "GenericProvider configs must set 'tool_capable: true' to affirm that "
                "the upstream API can emit OpenAI-style tool_calls."
            )

        if "Authorization" not in self.headers and self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def validate_tool_support(self) -> None:
        if not self.tool_capable:
            raise ValueError("Configured generic provider cannot emit tool calls.")

    def _get_value_from_path(
        self, data: Dict[str, Any], path: List[PathComponent]
    ) -> Any:
        current: Any = data
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif isinstance(current, list) and isinstance(key, int) and 0 <= key < len(current):
                current = current[key]
            else:
                return None
        return current

    def _deep_replace_placeholders(
        self, template_obj: Any, replacements: Dict[str, Any]
    ) -> Any:
        if isinstance(template_obj, dict):
            return {
                k: self._deep_replace_placeholders(v, replacements)
                for k, v in template_obj.items()
            }
        if isinstance(template_obj, list):
            return [
                self._deep_replace_placeholders(item, replacements)
                for item in template_obj
            ]
        if isinstance(template_obj, str):
            for placeholder, value in replacements.items():
                if template_obj == f"{{{{{placeholder}}}}}":
                    return value
            return template_obj
        return template_obj

    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        payload = self._deep_replace_placeholders(
            self.payload_template,
            {"messages": messages, "prompt": messages[-1]["content"] if messages else ""},
        )

        if payload.get("model") == "{{model}}":
            payload["model"] = model
        elif "model" not in payload and "model" in self.payload_template:
            payload["model"] = model

        if "generationConfig" in payload:
            gen_config = payload.get("generationConfig", {})
            gen_config["temperature"] = kwargs.get(
                "temperature", gen_config.get("temperature", 0.7)
            )
            gen_config["maxOutputTokens"] = kwargs.get(
                "max_tokens", gen_config.get("maxOutputTokens", 2048)
            )
            gen_config["topP"] = kwargs.get("top_p", gen_config.get("topP", 0.95))
            payload["generationConfig"] = gen_config
        else:
            payload["temperature"] = kwargs.get("temperature", payload.get("temperature", 0.7))
            payload["max_tokens"] = kwargs.get("max_tokens", payload.get("max_tokens", 2048))
            payload["top_p"] = kwargs.get("top_p", payload.get("top_p", 1.0))

        tool_choice = kwargs.get("tool_choice")
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or payload.get("tool_choice", "auto")

        print(f"[GenericProvider] Sending request to: {self.endpoint}")
        print(f"[GenericProvider] Payload: {json.dumps(payload, indent=2)}")
        print(f"[GenericProvider] Headers: {self.headers}")

        try:
            response = requests.post(
                self.endpoint, json=payload, headers=self.headers, timeout=kwargs.get("timeout", 60)
            )
            response.raise_for_status()
            response_data = response.json()
            print(f"[GenericProvider] Raw Response: {json.dumps(response_data, indent=2)}")
        except requests.RequestException as exc:
            error_message = f"Generic API request error: {exc}"
            if exc.response is not None:
                try:
                    error_details = exc.response.json()
                    api_error = self._get_value_from_path(
                        error_details, self.response_mapping.get("error_path", ["error"])
                    )
                    if api_error:
                        error_message += f" (Details: {api_error})"
                except json.JSONDecodeError:
                    error_message += f" (Raw response: {exc.response.text})"
            print(f"[GenericProvider] Error: {error_message}")
            return {"role": "assistant", "content": f"Error: {error_message}"}

        content = self._get_value_from_path(
            response_data, self.response_mapping.get("content_path", [])
        )
        tool_calls = self._get_value_from_path(
            response_data, self.response_mapping.get("tool_calls_path", [])
        )

        if not isinstance(content, (str, type(None))):
            content = str(content)
        if not isinstance(tool_calls, (list, type(None))):
            tool_calls = None

        return {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        }
