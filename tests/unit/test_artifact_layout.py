from __future__ import annotations

import pytest

from llmflow.planning.artifact_layout import (
    ensure_prompt_artifact_dir,
    format_artifact_attempt_label,
    reset_prompt_artifact_dir,
)


def test_reset_prompt_artifact_dir_replaces_prior_plan(tmp_path) -> None:
    root = tmp_path / "plans"
    first = reset_prompt_artifact_dir(root, "hashA", "plan-one")
    (first / "Plan.java").write_text("class One {}", encoding="utf-8")

    second = reset_prompt_artifact_dir(root, "hashA", "plan-two")

    assert second == root / "hashA"
    assert not (second / "Plan.java").exists()
    recorded = (second / ".plan_id").read_text(encoding="utf-8").strip()
    assert recorded == "plan-two"


def test_ensure_prompt_artifact_dir_conflict(tmp_path) -> None:
    root = tmp_path / "plans"
    reset_prompt_artifact_dir(root, "hashB", "plan-A")

    with pytest.raises(RuntimeError):
        ensure_prompt_artifact_dir(root, "hashB", "plan-B")


def test_format_artifact_attempt_label() -> None:
    assert format_artifact_attempt_label(1) == "1"
    assert format_artifact_attempt_label(2, 3) == "2.3"
    with pytest.raises(ValueError):
        format_artifact_attempt_label(0)
    with pytest.raises(ValueError):
        format_artifact_attempt_label(1, 0)
