"""Apply unified diffs using Aider's patch algorithm.

This module ports the core parsing and patch application routines from
``aider/coders/patch_coder.py`` (Apache 2.0). The original implementation is
adapted to operate inside LLMFlow tooling without the broader Aider runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


class DiffError(ValueError):
    """Raised when unified diff parsing or application fails."""


class ActionType(str, Enum):
    ADD = "Add"
    DELETE = "Delete"
    UPDATE = "Update"


@dataclass
class Chunk:
    orig_index: int = -1
    del_lines: List[str] = field(default_factory=list)
    ins_lines: List[str] = field(default_factory=list)


@dataclass
class PatchAction:
    type: ActionType
    path: str
    new_content: Optional[str] = None
    chunks: List[Chunk] = field(default_factory=list)
    move_path: Optional[str] = None


@dataclass
class Patch:
    actions: Dict[str, PatchAction] = field(default_factory=dict)
    fuzz: int = 0


def _norm(line: str) -> str:
    """Normalize patch lines for comparison."""

    return line.rstrip("\r")


def _normalize_patch_text(value: str) -> str:
    if not value:
        return ""
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _strip_code_fence_wrapper(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if not lines:
        return stripped
    opener = lines[0].strip()
    if not opener.startswith("```"):
        return stripped
    closing_index: Optional[int] = None
    for idx in range(len(lines) - 1, 0, -1):
        if lines[idx].strip() == "```":
            closing_index = idx
            break
    body = lines[1:closing_index] if closing_index is not None else lines[1:]
    return "\n".join(body).strip()


def _split_unified_diff_sections(text: str) -> List[str]:
    normalized = _normalize_patch_text(text.strip())
    normalized = _strip_code_fence_wrapper(normalized)
    if not normalized:
        return []
    lines = normalized.splitlines()
    sections: List[str] = []
    current: List[str] = []
    collecting = False
    for line in lines:
        if line.startswith("--- "):
            if current:
                sections.append("\n".join(current).strip())
                current = []
            collecting = True
        if not collecting:
            continue
        current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    return sections


def _normalize_diff_path(path: str) -> str:
    value = (path or "").strip()
    if not value:
        return value
    if value.startswith("a/") or value.startswith("b/"):
        value = value[2:]
    if value.startswith("./"):
        value = value[2:]
    value = value.strip('"')
    if "\t" in value:
        value = value.split("\t", 1)[0]
    return value


def _resolve_within_root(root: Path, relative_path: str) -> Path:
    candidate = (Path(relative_path) if Path(relative_path).is_absolute() else (root / relative_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:  # pragma: no cover - defensive
        raise DiffError(f"Patch path '{relative_path}' escapes root '{root}'.") from exc
    return candidate


@dataclass
class _UnifiedDiffEntry:
    action: ActionType
    path: str
    move_path: Optional[str] = None
    hunks: List[List[str]] = field(default_factory=list)
    add_lines: List[str] = field(default_factory=list)


def _parse_unified_diff_section(section: str) -> _UnifiedDiffEntry:
    text = _normalize_patch_text(section.strip())
    if not text:
        raise DiffError("Unified diff section was empty.")
    lines = [line.rstrip("\n") for line in text.splitlines() if line is not None]
    index = 0
    while index < len(lines) and not lines[index].startswith("--- "):
        index += 1
    if index == len(lines):
        raise DiffError("Unified diff chunk missing '--- <old>' header.")
    old_path = _normalize_diff_path(lines[index][4:])
    index += 1
    while index < len(lines) and not lines[index].startswith("+++ "):
        index += 1
    if index == len(lines):
        raise DiffError("Unified diff chunk missing '+++ <new>' header.")
    new_path = _normalize_diff_path(lines[index][4:])
    index += 1
    body = lines[index:]
    if any(line.startswith("Binary files ") or line == "GIT binary patch" for line in body):
        raise DiffError("Binary patches are not supported.")

    if not old_path and not new_path:
        raise DiffError("Unified diff chunk did not include file paths.")

    if old_path == "/dev/null":
        if not new_path:
            raise DiffError("Addition chunk missing new file path.")
        added_lines = [line[1:] for line in body if line.startswith("+")]
        if not added_lines:
            raise DiffError(f"Addition chunk for '{new_path}' did not include content.")
        return _UnifiedDiffEntry(action=ActionType.ADD, path=new_path, add_lines=added_lines)

    if new_path == "/dev/null":
        if not old_path:
            raise DiffError("Deletion chunk missing original path.")
        return _UnifiedDiffEntry(action=ActionType.DELETE, path=old_path)

    move_path = new_path if new_path != old_path else None
    hunks: List[List[str]] = []
    current: List[str] = []
    for line in body:
        if line.startswith("@@"):
            if current:
                hunks.append(current)
                current = []
            continue
        if line.startswith("diff --git") or line.startswith("index "):
            continue
        current.append(line)
    if current:
        hunks.append(current)
    return _UnifiedDiffEntry(action=ActionType.UPDATE, path=old_path, move_path=move_path, hunks=hunks)


def _entries_to_patch_text(entries: Sequence[_UnifiedDiffEntry]) -> str:
    if not entries:
        raise DiffError("Unified diff did not contain any file changes.")
    lines: List[str] = ["*** Begin Patch"]
    for entry in entries:
        if entry.action is ActionType.ADD:
            lines.append(f"*** Add File: {entry.path}")
            for payload in entry.add_lines:
                lines.append(f"+{payload}")
            continue
        if entry.action is ActionType.DELETE:
            lines.append(f"*** Delete File: {entry.path}")
            continue
        lines.append(f"*** Update File: {entry.path}")
        if entry.move_path:
            lines.append(f"*** Move to: {entry.move_path}")
        if not entry.hunks:
            lines.append("@@")
        for hunk in entry.hunks:
            lines.append("@@")
            for payload in hunk:
                if payload.startswith("\\ No newline"):
                    continue
                if payload and payload[0] in {" ", "+", "-"}:
                    lines.append(payload)
                else:
                    lines.append(f" {payload}")
    lines.append("*** End Patch")
    return "\n".join(lines).strip() + "\n"


def _identify_files_needed(patch_text: str) -> List[str]:
    lines = patch_text.splitlines()
    paths: List[str] = []
    for line in lines:
        normalized = _norm(line)
        if normalized.startswith("*** Update File: "):
            paths.append(normalized[len("*** Update File: ") :].strip())
        elif normalized.startswith("*** Delete File: "):
            paths.append(normalized[len("*** Delete File: ") :].strip())
    return paths


def _load_current_files(root: Path, patch_text: str, encoding: str) -> Dict[str, str]:
    files: Dict[str, str] = {}
    for rel_path in _identify_files_needed(patch_text):
        resolved = _resolve_within_root(root, rel_path)
        if not resolved.exists():
            raise DiffError(f"File '{rel_path}' referenced in patch was not found at '{resolved}'.")
        files[rel_path] = resolved.read_text(encoding=encoding)
    return files


def _parse_patch_text(lines: List[str], start_index: int, current_files: Dict[str, str]) -> Patch:
    patch = Patch()
    index = start_index
    fuzz_accumulator = 0
    while index < len(lines):
        line = lines[index]
        normalized = _norm(line)
        if normalized == "*** End Patch":
            index += 1
            break
        if normalized.startswith("*** Update File: "):
            path = normalized[len("*** Update File: ") :].strip()
            index += 1
            if not path:
                raise DiffError("Update action missing file path.")
            move_to = None
            if index < len(lines) and _norm(lines[index]).startswith("*** Move to: "):
                move_to = _norm(lines[index])[len("*** Move to: ") :].strip()
                index += 1
                if not move_to:
                    raise DiffError("Move to action missing target path.")
            if path not in current_files:
                raise DiffError(f"Missing content for file '{path}'.")
            file_content = current_files[path]
            existing = patch.actions.get(path)
            if existing is not None:
                if existing.type is not ActionType.UPDATE:
                    raise DiffError(f"Conflicting actions for file '{path}'.")
                new_action, index, fuzz = _parse_update_file_sections(lines, index, file_content)
                existing.chunks.extend(new_action.chunks)
                if move_to:
                    if existing.move_path and existing.move_path != move_to:
                        raise DiffError(f"Conflicting move targets for '{path}'.")
                    existing.move_path = move_to
                fuzz_accumulator += fuzz
            else:
                action, index, fuzz = _parse_update_file_sections(lines, index, file_content)
                action.path = path
                action.move_path = move_to
                patch.actions[path] = action
                fuzz_accumulator += fuzz
            continue
        if normalized.startswith("*** Delete File: "):
            path = normalized[len("*** Delete File: ") :].strip()
            index += 1
            if not path:
                raise DiffError("Delete action missing file path.")
            if path in patch.actions:
                raise DiffError(f"Duplicate actions declared for '{path}'.")
            patch.actions[path] = PatchAction(type=ActionType.DELETE, path=path)
            continue
        if normalized.startswith("*** Add File: "):
            path = normalized[len("*** Add File: ") :].strip()
            index += 1
            if not path:
                raise DiffError("Add action missing file path.")
            if path in patch.actions:
                raise DiffError(f"Duplicate actions declared for '{path}'.")
            action, index = _parse_add_file_content(lines, index)
            action.path = path
            patch.actions[path] = action
            continue
        if not normalized.strip():
            index += 1
            continue
        raise DiffError(f"Unknown line in patch: {line}")
    patch.fuzz = fuzz_accumulator
    return patch


def _parse_update_file_sections(
    lines: List[str],
    index: int,
    file_content: str,
) -> Tuple[PatchAction, int, int]:
    action = PatchAction(type=ActionType.UPDATE, path="")
    orig_lines = file_content.splitlines()
    current_file_index = 0
    total_fuzz = 0
    while index < len(lines):
        normalized = _norm(lines[index])
        if normalized.startswith(("*** End Patch", "*** Update File:", "*** Delete File:", "*** Add File:")):
            break
        scope_lines: List[str] = []
        while index < len(lines) and _norm(lines[index]).startswith("@@"):
            scope_text = lines[index][len("@@") :].strip()
            if scope_text:
                scope_lines.append(scope_text)
            index += 1
        if scope_lines:
            found_scope = False
            temp_index = current_file_index
            while temp_index < len(orig_lines):
                match = True
                for offset, scope in enumerate(scope_lines):
                    if temp_index + offset >= len(orig_lines) or _norm(orig_lines[temp_index + offset]).strip() != scope:
                        match = False
                        break
                if match:
                    current_file_index = temp_index + len(scope_lines)
                    found_scope = True
                    break
                temp_index += 1
            if not found_scope:
                raise DiffError(f"Could not locate scope context: {'; '.join(scope_lines)}")
            total_fuzz += 1
        context_block, chunks_in_section, next_index, is_eof = peek_next_section(lines, index)
        match_index, fuzz = find_context(orig_lines, context_block, current_file_index, is_eof)
        total_fuzz += fuzz
        if match_index == -1:
            raise DiffError("Could not match patch context block in file.")
        for chunk in chunks_in_section:
            chunk.orig_index += match_index
            action.chunks.append(chunk)
        current_file_index = match_index + len(context_block)
        index = next_index
    return action, index, total_fuzz


def _parse_add_file_content(lines: List[str], index: int) -> Tuple[PatchAction, int]:
    added_lines: List[str] = []
    while index < len(lines):
        line = lines[index]
        normalized = _norm(line)
        if normalized.startswith(("*** End Patch", "*** Update File:", "*** Delete File:", "*** Add File:")):
            break
        if normalized.startswith("@@"):
            index += 1
            continue
        if not line.startswith("+"):
            if not normalized:
                added_lines.append("")
                index += 1
                continue
            raise DiffError(f"Invalid add-file line: {line}")
        added_lines.append(line[1:])
        index += 1
    content = "\n".join(added_lines)
    return PatchAction(type=ActionType.ADD, path="", new_content=content), index


def find_context_core(lines: List[str], context: List[str], start: int) -> Tuple[int, int]:  # pragma: no cover - exercised via patch application
    if not context:
        return start, 0
    for idx in range(start, len(lines) - len(context) + 1):
        if lines[idx : idx + len(context)] == context:
            return idx, 0
    normalized = [entry.rstrip() for entry in context]
    for idx in range(start, len(lines) - len(context) + 1):
        if [entry.rstrip() for entry in lines[idx : idx + len(context)]] == normalized:
            return idx, 1
    stripped = [entry.strip() for entry in context]
    for idx in range(start, len(lines) - len(context) + 1):
        if [entry.strip() for entry in lines[idx : idx + len(context)]] == stripped:
            return idx, 100
    return -1, 0


def find_context(lines: List[str], context: List[str], start: int, eof: bool) -> Tuple[int, int]:  # pragma: no cover - exercised via patch application
    if eof:
        if len(lines) >= len(context):
            new_index, fuzz = find_context_core(lines, context, len(lines) - len(context))
            if new_index != -1:
                return new_index, fuzz
        new_index, fuzz = find_context_core(lines, context, start)
        return new_index, fuzz + 10_000
    return find_context_core(lines, context, start)


def peek_next_section(lines: List[str], index: int) -> Tuple[List[str], List[Chunk], int, bool]:  # pragma: no cover - exercised indirectly
    context_lines: List[str] = []
    del_lines: List[str] = []
    ins_lines: List[str] = []
    chunks: List[Chunk] = []
    mode = "keep"
    start_index = index
    while index < len(lines):
        line = lines[index]
        normalized = _norm(line)
        if normalized.startswith(("@@", "*** End Patch", "*** Update File:", "*** Delete File:", "*** Add File:", "*** End of File")):
            break
        if normalized == "***":
            break
        if normalized.startswith("***"):
            raise DiffError(f"Invalid line in update section: {line}")
        index += 1
        last_mode = mode
        if line.startswith("+"):
            mode = "add"
            payload = line[1:]
        elif line.startswith("-"):
            mode = "delete"
            payload = line[1:]
        elif line.startswith(" "):
            mode = "keep"
            payload = line[1:]
        elif line.strip() == "":
            mode = "keep"
            payload = ""
        else:
            raise DiffError(f"Invalid diff line: {line}")
        if mode == "keep" and last_mode != "keep":
            if del_lines or ins_lines:
                chunks.append(
                    Chunk(
                        orig_index=len(context_lines) - len(del_lines),
                        del_lines=del_lines,
                        ins_lines=ins_lines,
                    )
                )
            del_lines, ins_lines = [], []
        if mode == "delete":
            del_lines.append(payload)
            context_lines.append(payload)
        elif mode == "add":
            ins_lines.append(payload)
        else:
            context_lines.append(payload)
    if del_lines or ins_lines:
        chunks.append(
            Chunk(
                orig_index=len(context_lines) - len(del_lines),
                del_lines=del_lines,
                ins_lines=ins_lines,
            )
        )
    is_eof = False
    if index < len(lines) and _norm(lines[index]) == "*** End of File":
        index += 1
        is_eof = True
    if index == start_index and not is_eof:
        raise DiffError("Empty patch section encountered.")
    return context_lines, chunks, index, is_eof


def _apply_patch_actions(patch: Patch, root: Path, encoding: str) -> Dict[str, List[str]]:
    summary = {"added": [], "deleted": [], "updated": [], "moved": []}
    for rel_path, action in patch.actions.items():
        resolved = _resolve_within_root(root, rel_path)
        if action.type is ActionType.ADD:
            if resolved.exists():
                raise DiffError(f"File '{rel_path}' already exists; cannot add.")
            resolved.parent.mkdir(parents=True, exist_ok=True)
            content = action.new_content or ""
            if content and not content.endswith("\n"):
                content += "\n"
            resolved.write_text(content, encoding=encoding)
            summary["added"].append(rel_path)
            continue
        if action.type is ActionType.DELETE:
            if not resolved.exists():
                raise DiffError(f"File '{rel_path}' not found for deletion.")
            resolved.unlink()
            summary["deleted"].append(rel_path)
            continue
        if not resolved.exists():
            raise DiffError(f"File '{rel_path}' not found for update.")
        current_content = resolved.read_text(encoding=encoding)
        new_content = _apply_update(current_content, action, rel_path)
        target = _resolve_within_root(root, action.move_path) if action.move_path else resolved
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new_content, encoding=encoding)
        if action.move_path and target != resolved:
            if resolved.exists():
                resolved.unlink()
            summary["moved"].append({"from": rel_path, "to": action.move_path})
            summary["updated"].append(action.move_path)
        else:
            summary["updated"].append(rel_path)
    return summary


def _apply_update(text: str, action: PatchAction, path: str) -> str:
    if action.type is not ActionType.UPDATE:
        raise DiffError("_apply_update called with non-update action")
    orig_lines = text.splitlines()
    dest_lines: List[str] = []
    current_index = 0
    for chunk in sorted(action.chunks, key=lambda c: c.orig_index):
        start_index = chunk.orig_index
        if start_index < current_index:
            raise DiffError(f"{path}: overlapping patch chunk detected.")
        dest_lines.extend(orig_lines[current_index:start_index])
        num_deleted = len(chunk.del_lines)
        actual_deleted = orig_lines[start_index : start_index + num_deleted]
        norm_expected = [_norm(entry).strip() for entry in chunk.del_lines]
        norm_actual = [_norm(entry).strip() for entry in actual_deleted]
        if norm_expected != norm_actual:
            expected = "\n".join(f"- {entry}" for entry in chunk.del_lines)
            actual = "\n".join(f"  {entry}" for entry in actual_deleted)
            raise DiffError(
                f"{path}: patch mismatch near line {start_index + 1}.\n"
                f"Expected to remove:\n{expected}\nFound:\n{actual}"
            )
        dest_lines.extend(chunk.ins_lines)
        current_index = start_index + num_deleted
    dest_lines.extend(orig_lines[current_index:])
    result = "\n".join(dest_lines)
    if result or orig_lines:
        result += "\n"
    return result


def apply_unified_diff_patch(diff_text: str, root: Path, *, encoding: str = "utf-8") -> Dict[str, Any]:
    sections = _split_unified_diff_sections(diff_text)
    if not sections:
        raise DiffError("Unified diff did not contain any '---' file headers.")
    entries = [_parse_unified_diff_section(section) for section in sections]
    patch_text = _entries_to_patch_text(entries)
    current_files = _load_current_files(root, patch_text, encoding)
    lines = patch_text.splitlines()
    start_index = 1 if lines and lines[0].startswith("*** Begin Patch") else 0
    patch = _parse_patch_text(lines, start_index, current_files)
    summary = _apply_patch_actions(patch, root, encoding)
    summary["fuzz"] = patch.fuzz
    summary["files"] = sorted({*summary["added"], *summary["updated"], *summary["deleted"]})
    return summary
