"""LLM client that loads providers defined in llmflow.providers."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import instructor
from instructor import Mode
import yaml
from pydantic import BaseModel

from llmflow.providers import (
    GenericProvider,
    LLMProviderInterface,
    OllamaProvider,
    OpenAIProvider,
)

class LLMClient:
    """Wrapper around the configured LLM provider."""

    def __init__(self, config_file: str = "llm_config.yaml"):
        self.config_file = config_file
        provider_config = self._load_config(config_file)
        self.model = provider_config.get("model")
        if not self.model:
            raise ValueError("llm_config.yaml must specify a 'model'.")

        provider_name = provider_config.get("provider", "").lower()
        self.provider_name = provider_name
        self.provider_config = dict(provider_config)
        self.provider = self._build_provider(provider_name, provider_config)
        self.default_request_timeout = provider_config.get("request_timeout")
        self.provider.validate_tool_support()
        self._structured_mode = self._resolve_structured_mode(provider_config)
        self._structured_client = None

    def _load_config(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, "r", encoding="utf-8") as config_file:
                data = yaml.safe_load(config_file) or {}
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"LLM configuration file '{file_path}' not found."
            ) from exc
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Invalid YAML in '{file_path}': {exc}"
            ) from exc

        provider_config = data.get("provider_config")
        if not provider_config:
            raise ValueError("'provider_config' section missing in llm_config.yaml.")
        return provider_config

    def _build_provider(
        self, provider_name: str, provider_config: Dict[str, Any]
    ) -> LLMProviderInterface:
        if provider_name == "openai":
            api_key = provider_config.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI provider selected but no API key found in config or OPENAI_API_KEY."
                )
            return OpenAIProvider(api_key=api_key)

        if provider_name == "ollama":
            base_url = provider_config.get("base_url", "http://localhost:11434")
            options = provider_config.get("options", {})
            return OllamaProvider(
                model_name=self.model,
                base_url=base_url,
                default_options=options,
            )

        if provider_name == "generic":
            return GenericProvider(provider_config)

        raise ValueError(
            f"Unsupported provider '{provider_name}'. Choose 'openai', 'ollama', or 'generic'."
        )

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        current_model = kwargs.pop("model", self.model)
        print(
            "[LLMClient] Generating response via %s (%s)",
            self.provider.__class__.__name__,
            current_model,
        )
        if "timeout" not in kwargs and self.default_request_timeout:
            kwargs["timeout"] = self.default_request_timeout
        return self.provider.generate_response(
            messages=messages,
            model=current_model,
            tools=tools,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Structured output helpers

    def structured_generate(
        self,
        *,
        messages: List[Dict[str, Any]],
        response_model: Any,
        **kwargs: Any,
    ) -> BaseModel:
        """Generate a structured response validated by ``response_model``."""

        params = dict(kwargs)
        params.pop("force_tool_call", None)
        if self._mode_supports_tool_schemas:
            params.setdefault("parallel_tool_calls", False)
        else:
            params.pop("parallel_tool_calls", None)
            params.pop("tools", None)
            params.pop("tool_choice", None)
        client = self._ensure_structured_client()
        return client.create(
            messages=messages,
            response_model=response_model,
            **params,
        )

    def _ensure_structured_client(self):
        if self._structured_client is None:
            self._structured_client = self._build_structured_client()
        return self._structured_client

    def _build_structured_client(self):
        if not self.provider_name:
            raise RuntimeError("Structured generation requires a configured provider.")

        provider_id = f"{self.provider_name}/{self.model}"
        kwargs = self._build_structured_kwargs()
        return instructor.from_provider(
            provider_id,
            mode=self._structured_mode,
            **kwargs,
        )

    def _build_structured_kwargs(self) -> Dict[str, Any]:
        provider = self.provider_name
        if provider == "openai":
            api_key = self.provider_config.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OpenAI structured generation requires an API key in the config or OPENAI_API_KEY."
                )
            return {"api_key": api_key}

        if provider == "ollama":
            base_url = self.provider_config.get("base_url") or "http://localhost:11434"
            normalized_url = base_url.rstrip("/")
            if not normalized_url.endswith("/v1"):
                normalized_url = f"{normalized_url}/v1"
            return {"base_url": normalized_url}

        raise RuntimeError(
            f"Structured generation is not supported for provider '{provider}'."
        )

    def _resolve_structured_mode(self, provider_config: Dict[str, Any]) -> Mode:
        requested = provider_config.get("structured_mode")
        if isinstance(requested, Mode):
            return requested
        if isinstance(requested, str) and requested.strip():
            normalized = requested.strip().upper().replace("-", "_")
            try:
                return Mode[normalized]
            except KeyError as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"Unsupported structured_mode '{requested}'. Refer to Instructor Mode enum."
                ) from exc
        if self.provider_name == "ollama":
            return Mode.JSON
        return Mode.TOOLS

    @property
    def structured_mode(self) -> Mode:
        return self._structured_mode

    @property
    def _mode_supports_tool_schemas(self) -> bool:
        mode = self._structured_mode
        return mode is Mode.FUNCTIONS or mode.name.endswith("TOOLS")