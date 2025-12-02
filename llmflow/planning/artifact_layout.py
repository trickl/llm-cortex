"""Helpers for organizing plan artifact directories."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

_PLAN_ID_SENTINEL = ".plan_id"


def _read_plan_id(prompt_dir: Path) -> Optional[str]:
    sentinel = prompt_dir / _PLAN_ID_SENTINEL
    if not sentinel.exists():
        return None
    try:
        return sentinel.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _write_plan_id(prompt_dir: Path, plan_id: str) -> None:
    sentinel = prompt_dir / _PLAN_ID_SENTINEL
    sentinel.write_text(plan_id.strip(), encoding="utf-8")


def reset_prompt_artifact_dir(root: Path, prompt_hash: str, plan_id: str) -> Path:
    """Ensure ``prompt_hash`` maps to ``plan_id`` by clearing any stale data."""

    prompt_dir = root / prompt_hash
    recorded = _read_plan_id(prompt_dir)
    if recorded and recorded == plan_id:
        return prompt_dir
    if prompt_dir.exists():
        shutil.rmtree(prompt_dir, ignore_errors=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    _write_plan_id(prompt_dir, plan_id)
    return prompt_dir


def ensure_prompt_artifact_dir(root: Path, prompt_hash: str, plan_id: str) -> Path:
    """Return the directory for ``prompt_hash`` without deleting existing artifacts."""

    prompt_dir = root / prompt_hash
    prompt_dir.mkdir(parents=True, exist_ok=True)
    recorded = _read_plan_id(prompt_dir)
    if recorded and recorded != plan_id:
        raise RuntimeError(
            f"Prompt hash {prompt_hash} already associated with plan_id {recorded};"
            f" refusing to reuse for {plan_id}."
        )
    if not recorded:
        _write_plan_id(prompt_dir, plan_id)
    return prompt_dir


def format_artifact_attempt_label(
    attempt_number: int,
    refinement_iteration: Optional[int] = None,
) -> str:
    """Format attempt/refinement numbers into a stable directory label."""

    if attempt_number <= 0:
        raise ValueError("attempt_number must be >= 1")
    if refinement_iteration is not None:
        if refinement_iteration <= 0:
            raise ValueError("refinement_iteration must be >= 1 when provided")
        return f"{attempt_number}.{refinement_iteration}"
    return str(attempt_number)
