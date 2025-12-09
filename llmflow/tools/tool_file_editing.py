"""File editing helpers so agents can safely mutate source files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from llmflow.tools.tool_decorator import register_tool
from llmflow.tools.unified_diff_patch import apply_unified_diff_patch as _apply_unified_diff_patch

_TOOL_TAGS = ["file_system", "file_editing", "file_management"]


def _resolve_path(file_path: str) -> Path:
    path = Path(file_path).expanduser().resolve()
    return path


def _ensure_parent(path: Path, create_directories: bool) -> None:
    parent = path.parent
    if not parent.exists():
        if not create_directories:
            raise FileNotFoundError(
                f"Directory '{parent}' does not exist. Pass create_directories=True to create it."
            )
        parent.mkdir(parents=True, exist_ok=True)


@register_tool(tags=_TOOL_TAGS)
def overwrite_text_file(
    file_path: str,
    content: str,
    encoding: str = "utf-8",
    create_directories: bool = False,
    ensure_trailing_newline: bool = True,
) -> Dict[str, Any]:
    """Replace or create a text file with the provided content."""

    try:
        path = _resolve_path(file_path)
        _ensure_parent(path, create_directories)
        final_content = content
        if ensure_trailing_newline and not final_content.endswith("\n"):
            final_content += "\n"
        path.write_text(final_content, encoding=encoding)
        return {
            "success": True,
            "path": str(path),
            "bytes_written": len(final_content.encode(encoding)),
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@register_tool(tags=_TOOL_TAGS)
def apply_text_rewrite(
    file_path: str,
    original_snippet: str,
    new_snippet: str,
    occurrence: Optional[int] = None,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """Apply a targetted text replacement inside an existing file."""

    try:
        path = _resolve_path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        text = path.read_text(encoding=encoding)
        if occurrence is None:
            replaced_text = text.replace(original_snippet, new_snippet)
            replacements = text.count(original_snippet)
        else:
            replacements = 0
            segments = []
            remaining = text
            while remaining:
                idx = remaining.find(original_snippet)
                if idx == -1:
                    segments.append(remaining)
                    break
                replacements += 1
                before, after = remaining[:idx], remaining[idx + len(original_snippet):]
                if replacements == occurrence:
                    segments.append(before + new_snippet)
                    remaining = after
                    break
                else:
                    segments.append(before + original_snippet)
                    remaining = after
            segments.append(remaining)
            replaced_text = "".join(segments)
        if replacements == 0:
            return {"success": False, "error": "Original snippet not found."}
        path.write_text(replaced_text, encoding=encoding)
        return {
            "success": True,
            "path": str(path),
            "replacements": replacements if occurrence is None else min(replacements, 1),
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@register_tool(tags=_TOOL_TAGS)
def apply_unified_diff_patch(
    diff_text: str,
    repo_root: Optional[str] = None,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """Apply a unified diff (Strict Unified Diff spec) to files on disk.

    Args:
        diff_text: Unified diff content containing one or more file sections.
        repo_root: Optional root directory that file paths are resolved against.
        encoding: Text encoding used when reading and writing files.
    """

    try:
        root = Path(repo_root).expanduser().resolve() if repo_root else Path.cwd()
        summary = _apply_unified_diff_patch(diff_text, root, encoding=encoding)
        return {"success": True, "root": str(root), **summary}
    except Exception as exc:
        return {"success": False, "error": str(exc)}
