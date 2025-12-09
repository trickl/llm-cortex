from __future__ import annotations

from pathlib import Path

from llmflow.tools import tool_file_editing


def test_overwrite_text_file_creates_parent(tmp_path: Path) -> None:
    target = tmp_path / "src" / "module.py"

    result = tool_file_editing.overwrite_text_file(
        file_path=str(target),
        content="value = 1",
        create_directories=True,
    )

    assert result["success"] is True
    assert target.read_text(encoding="utf-8").strip() == "value = 1"


def test_apply_text_rewrite_single_occurrence(tmp_path: Path) -> None:
    file_path = tmp_path / "app.py"
    file_path.write_text("status = 'todo'\n", encoding="utf-8")

    result = tool_file_editing.apply_text_rewrite(
        file_path=str(file_path),
        original_snippet="'todo'",
        new_snippet="'done'",
        occurrence=1,
    )

    assert result["success"] is True
    assert "replacements" in result
    assert file_path.read_text(encoding="utf-8").strip() == "status = 'done'"


def test_apply_unified_diff_patch_updates_file(tmp_path: Path) -> None:
    file_path = tmp_path / "Planner.java"
    file_path.write_text("line1\nline2\n", encoding="utf-8")

    diff = "\n".join(
        [
            "--- a/Planner.java",
            "+++ b/Planner.java",
            "@@ -1,2 +1,2 @@",
            " line1",
            "-line2",
            "+line-two",
        ]
    )

    result = tool_file_editing.apply_unified_diff_patch(
        diff_text=diff,
        repo_root=str(tmp_path),
    )

    assert result["success"] is True
    assert file_path.read_text(encoding="utf-8") == "line1\nline-two\n"


def test_apply_unified_diff_patch_adds_and_deletes(tmp_path: Path) -> None:
    added = tmp_path / "src" / "NewFile.txt"
    removed = tmp_path / "Legacy.txt"
    removed.parent.mkdir(parents=True, exist_ok=True)
    removed.write_text("legacy\n", encoding="utf-8")

    diff = "\n".join(
        [
            "--- /dev/null",
            "+++ b/src/NewFile.txt",
            "@@ -0,0 +1,2 @@",
            "+alpha",
            "+beta",
            "--- a/Legacy.txt",
            "+++ /dev/null",
            "@@ -1,1 +0,0 @@",
            "-legacy",
        ]
    )

    result = tool_file_editing.apply_unified_diff_patch(diff_text=diff, repo_root=str(tmp_path))

    assert result["success"] is True
    assert added.read_text(encoding="utf-8") == "alpha\nbeta\n"
    assert not removed.exists()


def test_apply_unified_diff_patch_rejects_path_escape(tmp_path: Path) -> None:
    diff = "\n".join(
        [
            "--- a/../outside.txt",
            "+++ b/../outside.txt",
            "@@ -1,1 +1,1 @@",
            "-x",
            "+y",
        ]
    )

    result = tool_file_editing.apply_unified_diff_patch(diff_text=diff, repo_root=str(tmp_path))

    assert result["success"] is False
    assert "escapes root" in result["error"]
