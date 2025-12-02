"""Tests for the LLMClient structured generation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest
from instructor import Mode
import yaml

from llmflow.llm_client import LLMClient
from llmflow.providers.ollama_provider import OllamaProvider


def _write_config(tmp_path: Path, provider_config: Dict[str, object]) -> Path:
    config_path = tmp_path / "llm_config.yaml"
    config_path.write_text(
        yaml.safe_dump({"provider_config": provider_config}),
        encoding="utf-8",
    )
    return config_path


def _disable_tool_validation(monkeypatch) -> None:
    monkeypatch.setattr(OllamaProvider, "validate_tool_support", lambda self: None)


def test_structured_kwargs_adds_v1_suffix(tmp_path, monkeypatch) -> None:
    _disable_tool_validation(monkeypatch)
    config_path = _write_config(
        tmp_path,
        {
            "provider": "ollama",
            "model": "qwen2.5-coder:7b",
            "base_url": "http://localhost:11434",
        },
    )

    client = LLMClient(config_file=str(config_path))

    kwargs = client._build_structured_kwargs()

    assert kwargs["base_url"] == "http://localhost:11434/v1"


def test_structured_kwargs_preserves_existing_v1(tmp_path, monkeypatch) -> None:
    _disable_tool_validation(monkeypatch)
    config_path = _write_config(
        tmp_path,
        {
            "provider": "ollama",
            "model": "qwen2.5-coder:7b",
            "base_url": "http://llm-proxy.internal/api/v1/",
        },
    )

    client = LLMClient(config_file=str(config_path))

    kwargs = client._build_structured_kwargs()

    assert kwargs["base_url"] == "http://llm-proxy.internal/api/v1"


def test_ollama_defaults_to_json_mode(tmp_path, monkeypatch) -> None:
    _disable_tool_validation(monkeypatch)
    config_path = _write_config(
        tmp_path,
        {
            "provider": "ollama",
            "model": "qwen2.5-coder:7b",
            "base_url": "http://localhost:11434",
        },
    )

    client = LLMClient(config_file=str(config_path))

    assert client.structured_mode == Mode.JSON


def test_structured_mode_override_respected(tmp_path, monkeypatch) -> None:
    _disable_tool_validation(monkeypatch)
    config_path = _write_config(
        tmp_path,
        {
            "provider": "ollama",
            "model": "qwen2.5-coder:7b",
            "base_url": "http://localhost:11434",
            "structured_mode": "md_json",
        },
    )

    client = LLMClient(config_file=str(config_path))

    assert client.structured_mode == Mode.MD_JSON


def test_invalid_structured_mode_raises(tmp_path, monkeypatch) -> None:
    _disable_tool_validation(monkeypatch)
    config_path = _write_config(
        tmp_path,
        {
            "provider": "ollama",
            "model": "qwen2.5-coder:7b",
            "base_url": "http://localhost:11434",
            "structured_mode": "invalid-mode",
        },
    )

    with pytest.raises(ValueError):
        LLMClient(config_file=str(config_path))


def test_tool_validation_runs_by_default(tmp_path, monkeypatch) -> None:
    call_count = {"value": 0}

    def _spy(self):  # noqa: ANN001 - bound via monkeypatch
        call_count["value"] += 1

    monkeypatch.setattr(OllamaProvider, "validate_tool_support", _spy)
    config_path = _write_config(
        tmp_path,
        {
            "provider": "ollama",
            "model": "qwen2.5-coder:7b",
            "base_url": "http://localhost:11434",
        },
    )

    LLMClient(config_file=str(config_path))

    assert call_count["value"] == 1


def test_tool_validation_can_be_disabled(tmp_path, monkeypatch) -> None:
    call_count = {"value": 0}

    def _spy(self):  # noqa: ANN001 - bound via monkeypatch
        call_count["value"] += 1

    monkeypatch.setattr(OllamaProvider, "validate_tool_support", _spy)
    config_path = _write_config(
        tmp_path,
        {
            "provider": "ollama",
            "model": "qwen2.5-coder:7b",
            "base_url": "http://localhost:11434",
            "require_tool_support": False,
        },
    )

    LLMClient(config_file=str(config_path))

    assert call_count["value"] == 0
