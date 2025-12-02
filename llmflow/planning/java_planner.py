"""Utilities for prompting the LLM to emit Java plans."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import textwrap
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

try:  # pragma: no cover - guard against optional dependency changes
    from instructor.core.exceptions import InstructorRetryException
except ImportError:  # pragma: no cover
    InstructorRetryException = None  # type: ignore[assignment]

from llmflow.llm_client import LLMClient
from llmflow.logging_utils import LLM_LOGGER_NAME, PLAN_LOGGER_NAME

_EMBEDDED_SPEC = (
    "You coordinate goal-driven Java automation. Return exactly one top-level class named Planner "
    "that calls PlanningToolStubs helpers to perform every side effect. Prefer helper decomposition, "
    "avoid markdown, and emit compilable Java source."
)
_STRUCTURED_MAX_RETRIES_ENV = "LLMFLOW_PLANNER_STRUCTURED_MAX_RETRIES"
_PLAIN_FALLBACK_MAX_ATTEMPTS = 2
_PLANNER_TOOL_NAME = "define_java_plan"

def _read_structured_retry_limit() -> Optional[int]:
    value = os.getenv(_STRUCTURED_MAX_RETRIES_ENV)
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        logger.warning(
            "Invalid %s value '%s'; falling back to provider defaults",
            _STRUCTURED_MAX_RETRIES_ENV,
            value,
        )
        return None
    if parsed < 0:
        logger.warning(
            "%s must be >= 0; falling back to provider defaults",
            _STRUCTURED_MAX_RETRIES_ENV,
        )
        return None
    return parsed

logger = logging.getLogger(__name__)
plan_logger = logging.getLogger(PLAN_LOGGER_NAME)
llm_logger = logging.getLogger(LLM_LOGGER_NAME)

_CLASS_DECL_PATTERN = re.compile(r"^\s*(?:public\s+)?class\s+\w+", re.MULTILINE)
_CODE_FENCE_PATTERN = re.compile(r"```(?P<lang>[^\n`]*)\n", re.MULTILINE)
_MARKDOWN_PREFIX_PATTERN = re.compile(
    r"^(?:#{1,6}\s+|>{1,}\s+|[-*+]\s+|\d+\.\s+|`{1,3}$)",
)
_LIKELY_JAVA_PREFIX_PATTERN = re.compile(
    r"^(?:package\s+|import\s+|(?:public|protected|private|abstract|final|static)\b|class\s+|interface\s+|enum\s+|record\s+|@|/\*|//|\*)",
)
_TOKEN_COUNT_PATTERN = re.compile(r"\S+")


def _compute_prompt_hash(messages: Sequence[Dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for message in messages:
        role = str(message.get("role", ""))
        digest.update(role.encode("utf-8"))
        digest.update(b"\x00")
        content = message.get("content")
        if isinstance(content, str):
            payload = content
        else:
            payload = json.dumps(content, sort_keys=True, ensure_ascii=False)
        digest.update(payload.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _normalize_java_source(source: str) -> str:
    """Normalize Java source and ensure it declares a top-level class."""

    decoded = _decode_plan_string(source)
    if isinstance(decoded, str) and decoded.strip():
        working = decoded
    else:
        working = source

    stripped = working.strip()
    candidate = _extract_java_candidate(stripped)
    if not _CLASS_DECL_PATTERN.search(candidate):
        wrapped = _wrap_statements_in_class(candidate)
        if wrapped and _CLASS_DECL_PATTERN.search(wrapped):
            return wrapped.strip()
        raise ValueError("Java payload must declare a top-level class.")
    return candidate.strip()


def _extract_java_candidate(source: str) -> str:
    block = _extract_code_block(source)
    if block:
        cleaned = _strip_trailing_backticks(block)
        if cleaned:
            return cleaned

    trimmed = _trim_non_java_prefix(source)
    trimmed = _strip_trailing_backticks(trimmed)
    return trimmed or source


def _extract_code_block(source: str) -> Optional[str]:
    candidates: List[Dict[str, Any]] = []
    for match in _CODE_FENCE_PATTERN.finditer(source):
        lang = (match.group("lang") or "").strip().lower()
        block_start = match.end()
        block_end = source.find("```", block_start)
        closed = True
        if block_end == -1:
            block_end = len(source)
            closed = False
        body = source[block_start:block_end].strip()
        if not body:
            continue
        candidates.append(
            {
                "lang": lang,
                "body": body,
                "closed": closed,
                "start": match.start(),
            }
        )

    if not candidates:
        return None

    def _score(candidate: Dict[str, Any]) -> Tuple[int, int, int]:
        lang_score = 1 if "java" in candidate["lang"] else 0
        closed_score = 1 if candidate["closed"] else 0
        return (lang_score, closed_score, candidate["start"])

    best = max(candidates, key=_score)
    return best["body"]


def _trim_non_java_prefix(source: str) -> str:
    lines = source.splitlines()
    result: List[str] = []
    dropping = True
    for line in lines:
        stripped = line.lstrip()
        if dropping:
            if not stripped:
                continue
            if stripped.startswith("```"):
                continue
            if _MARKDOWN_PREFIX_PATTERN.match(stripped):
                continue
            if _LIKELY_JAVA_PREFIX_PATTERN.match(stripped):
                dropping = False
                result.append(line)
                continue
            # Skip any other non-Java preamble lines.
            continue
        result.append(line)
    return "\n".join(result).lstrip()


def _strip_trailing_backticks(source: str) -> str:
    lines = source.splitlines()
    while lines and lines[-1].strip() in {"```", "``", "`"}:
        lines.pop()
    return "\n".join(lines).rstrip()


def _wrap_statements_in_class(candidate: str) -> Optional[str]:
    decoded = _decode_plan_string(candidate)
    if isinstance(decoded, dict):
        rendered = _render_java_from_structured_payload(decoded)
        if rendered:
            return rendered
    elif isinstance(decoded, str) and decoded.strip():
        candidate = decoded.strip()
    lines = [line.rstrip() for line in candidate.splitlines() if line.strip()]
    if not lines:
        return None
    package_lines: List[str] = []
    import_lines: List[str] = []
    body_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("package "):
            normalized = stripped if stripped.endswith(";") else f"{stripped};"
            if normalized not in package_lines:
                package_lines.append(normalized)
            continue
        if stripped.startswith("import "):
            normalized = stripped if stripped.endswith(";") else f"{stripped};"
            if normalized not in import_lines:
                import_lines.append(normalized)
            continue
        body_lines.append(stripped)
    if not body_lines:
        return None
    if not any(";" in line or "PlanningToolStubs" in line or line.startswith(("if ", "for ", "while ", "return")) for line in body_lines):
        return None
    assembled: List[str] = []
    if package_lines:
        assembled.extend(package_lines)
    if import_lines:
        if assembled:
            assembled.append("")
        assembled.extend(import_lines)
    if assembled:
        assembled.append("")
    assembled.append("public class Planner {")
    assembled.append("    public static void main(String[] args) throws Exception {")
    for body in body_lines:
        assembled.append(f"        {body}")
    assembled.append("    }")
    assembled.append("}")
    return "\n".join(assembled).strip()


def _extract_tool_call_count(exc: Exception) -> Optional[int]:
    """Best-effort extraction of tool call count from Instructor retries."""

    if InstructorRetryException is None:
        return None
    if not isinstance(exc, InstructorRetryException):
        return None
    attempts = getattr(exc, "failed_attempts", None)
    if not attempts:
        return None
    for attempt in attempts:
        completion = getattr(attempt, "completion", None)
        if not completion:
            continue
        choices = getattr(completion, "choices", None)
        if not choices:
            continue
        message = getattr(choices[0], "message", None)
        if not message:
            continue
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls is None:
            return 0
        return len(tool_calls)
    return None


def _summarize_structured_failure(exc: Exception) -> Optional[str]:
    """Return a user-friendly explanation for structured generation failures."""

    count = _extract_tool_call_count(exc)
    if count == 0:
        return "Structured Java plan request produced no tool calls."
    if count and count != 1:
        return f"Structured Java plan request produced {count} tool calls; expected exactly one."

    message = str(exc)
    if "Instructor does not support multiple tool calls" in message:
        return "Structured Java plan request did not yield exactly one tool call."
    if message:
        return f"Structured Java plan request failed: {message}"
    return None


def _serialize_completion_for_logging(completion: Any) -> Optional[str]:
    if completion is None:
        return None
    serializer = getattr(completion, "model_dump_json", None)
    if callable(serializer):
        try:
            return serializer(indent=2)
        except TypeError:
            try:
                return serializer()
            except Exception:  # pragma: no cover - best effort only
                pass
    model_dump = getattr(completion, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
            return json.dumps(dumped, indent=2, ensure_ascii=False)
        except Exception:  # pragma: no cover - best effort only
            pass
    try:
        return json.dumps(completion, indent=2, ensure_ascii=False)
    except TypeError:
        return str(completion)


def _log_structured_failure_details(plan_id: str, exc: Exception) -> None:
    plan_logger.error(
        "planner_structured_exception plan_id=%s type=%s message=%s",
        plan_id,
        type(exc).__name__,
        exc,
    )
    if InstructorRetryException is None or not isinstance(exc, InstructorRetryException):
        return
    attempts = getattr(exc, "failed_attempts", None) or []
    if not attempts:
        completion = getattr(exc, "last_completion", None)
        _log_structured_completion_payload(
            plan_id=plan_id,
            attempt_label="last",
            completion=completion,
            stage="retry_failure",
        )
        return
    for idx, attempt in enumerate(attempts, start=1):
        completion = getattr(attempt, "completion", None) or getattr(attempt, "response", None)
        _log_structured_completion_payload(
            plan_id=plan_id,
            attempt_label=str(idx),
            completion=completion,
            stage="retry_failure",
        )


def _log_structured_attempt_completion(plan_id: str, attempt: int, completion: Any) -> None:
    _log_structured_completion_payload(
        plan_id=plan_id,
        attempt_label=str(attempt),
        completion=completion,
        stage="parse_error",
    )


def _log_structured_completion_payload(
    *,
    plan_id: str,
    attempt_label: str,
    completion: Any,
    stage: str,
) -> None:
    serialized = _serialize_completion_for_logging(completion)
    if not serialized:
        return
    plan_logger.warning(
        "planner_structured_payload plan_id=%s attempt=%s stage=%s detail=full_completion_logged_to=agentcortex.llm",
        plan_id,
        attempt_label,
        stage,
    )
    llm_logger.error(
        "planner_structured_payload_dump plan_id=%s attempt=%s stage=%s\n%s",
        plan_id,
        attempt_label,
        stage,
        serialized,
    )

    plan_text = _extract_plan_text_from_completion(completion, serialized)
    if plan_text:
        plan_logger.info(
            "planner_structured_plan_body plan_id=%s attempt=%s stage=%s\n%s",
            plan_id,
            attempt_label,
            stage,
            _format_plan_text(plan_text),
        )


def _extract_plan_text_from_completion(completion: Any, serialized: Optional[str]) -> Optional[str]:
    candidates: List[Any] = []
    if completion is not None:
        model_dump = getattr(completion, "model_dump", None)
        if callable(model_dump):
            try:
                candidates.append(model_dump())
            except Exception:  # pragma: no cover - best effort
                pass
        dict_attr = getattr(completion, "__dict__", None)
        if isinstance(dict_attr, dict):
            candidates.append(dict(dict_attr))
    if serialized:
        try:
            candidates.append(json.loads(serialized))
        except json.JSONDecodeError:
            pass
    candidates.append(completion)

    for candidate in candidates:
        text = _extract_plan_text(candidate)
        if text:
            return text
    return None


def _extract_plan_text(payload: Any) -> Optional[str]:
    if payload is None:
        return None
    if isinstance(payload, str):
        return _parse_plan_content_string(payload)
    if isinstance(payload, dict):
        direct = _extract_plan_from_dict(payload)
        if direct:
            return direct
        for key in ("choices", "message", "messages", "delta", "output"):
            value = payload.get(key)
            text = _extract_plan_text(value)
            if text:
                return text
        return None
    if isinstance(payload, Iterable) and not isinstance(payload, (bytes, bytearray)):
        for item in payload:
            text = _extract_plan_text(item)
            if text:
                return text
    return None


def _extract_plan_from_dict(payload: Dict[str, Any]) -> Optional[str]:
    tool_payload = payload.get(_PLANNER_TOOL_NAME)
    if isinstance(tool_payload, dict):
        rendered = _render_java_from_structured_payload(tool_payload)
        if rendered:
            return rendered
    elif isinstance(tool_payload, str) and tool_payload.strip():
        decoded = _decode_plan_string(tool_payload)
        if isinstance(decoded, dict):
            rendered = _render_java_from_structured_payload(decoded)
            if rendered:
                return rendered
        elif isinstance(decoded, str) and decoded.strip():
            return decoded.strip()

    ast_rendered = _render_java_from_structured_payload(payload)
    if ast_rendered:
        return ast_rendered
    for key in ("java", "java_code", "code", "source"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            decoded = _decode_plan_string(value)
            if isinstance(decoded, str) and decoded.strip():
                return decoded.strip()
    defines_block = payload.get("defines")
    if isinstance(defines_block, dict):
        for candidate in ("java", "java_code", "code", "source"):
            candidate_value = defines_block.get(candidate)
            if isinstance(candidate_value, str) and candidate_value.strip():
                decoded = _decode_plan_string(candidate_value)
                if isinstance(decoded, str) and decoded.strip():
                    return decoded.strip()
    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            parsed = _parse_plan_content_string(content)
            if parsed:
                return parsed
    return None


def _render_java_from_structured_payload(payload: Dict[str, Any]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    main_statements = _collect_statement_lines(payload.get("main"))
    if not main_statements:
        main_statements = _collect_statement_lines(payload.get("statements"))
    helper_blocks = _render_helper_blocks(payload.get("helpers") or payload.get("methods"))
    if not main_statements and not helper_blocks:
        return None

    class_name = str(payload.get("name") or "Planner").strip() or "Planner"
    package_decl = str(payload.get("package") or "").strip()
    imports = _normalize_import_entries(payload.get("imports") or payload.get("import"))

    lines: List[str] = []
    if package_decl:
        package_line = package_decl
        if not package_line.startswith("package "):
            package_line = f"package {package_line}"
        if not package_line.endswith(";"):
            package_line = f"{package_line};"
        lines.append(package_line)
        lines.append("")

    if imports:
        lines.extend(imports)
        lines.append("")

    lines.append(f"public class {class_name} {{")
    if main_statements:
        lines.append("    public static void main(String[] args) throws Exception {")
        for stmt in main_statements:
            lines.append(f"        {stmt}")
        lines.append("    }")
    else:
        lines.append("    public static void main(String[] args) throws Exception {")
        lines.append("        // TODO: implement main body")
        lines.append("    }")

    if helper_blocks:
        lines.extend(helper_blocks)

    lines.append("}")
    return "\n".join(line for line in lines if line is not None).strip()


def _collect_statement_lines(source: Any) -> List[str]:
    if source is None:
        return []
    if isinstance(source, str):
        text = source.strip()
        return [text] if text else []
    if isinstance(source, dict):
        for key in ("statements", "body", "lines"):
            if key in source:
                return _collect_statement_lines(source[key])
        fallback = source.get("statement") or source.get("code")
        if fallback:
            return _collect_statement_lines(fallback)
        return []
    if isinstance(source, Iterable) and not isinstance(source, (bytes, bytearray)):
        lines: List[str] = []
        for item in source:
            lines.extend(_collect_statement_lines(item))
        return lines
    return []


def _normalize_import_entries(entries: Any) -> List[str]:
    if entries is None:
        return []
    if isinstance(entries, str):
        entries = [entries]
    normalized: List[str] = []
    for raw in entries or []:
        text = str(raw).strip()
        if not text:
            continue
        if text.startswith("package "):
            continue
        if not text.startswith("import "):
            text = f"import {text}"
        if not text.endswith(";"):
            text = f"{text};"
        if text not in normalized:
            normalized.append(text)
    return normalized


def _render_helper_blocks(entries: Any) -> List[str]:
    if not entries:
        return []
    if isinstance(entries, dict):
        iterable: Iterable[Any] = [entries]
    elif isinstance(entries, Iterable) and not isinstance(entries, (bytes, bytearray, str)):
        iterable = entries
    else:
        return []

    blocks: List[str] = []
    helper_index = 1
    for entry in iterable:
        if isinstance(entry, str):
            body = entry.strip()
            if not body:
                continue
            blocks.extend(["", f"    {body}"])
            continue
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or f"helper{helper_index}").strip() or f"helper{helper_index}"
        helper_index += 1
        return_type = str(entry.get("return_type") or "void").strip() or "void"
        visibility = str(entry.get("visibility") or "private static").strip() or "private static"
        params = _format_helper_parameters(entry.get("parameters"))
        statements = _collect_statement_lines(entry)
        if not statements:
            statements = ["// TODO: implement"]
        blocks.append("")
        blocks.append(f"    {visibility} {return_type} {name}({params}) throws Exception {{")
        for stmt in statements:
            blocks.append(f"        {stmt}")
        blocks.append("    }")
    return blocks


def _format_helper_parameters(parameters: Any) -> str:
    if parameters is None:
        return ""
    if isinstance(parameters, str):
        return parameters.strip()
    if isinstance(parameters, dict):
        type_hint = parameters.get("type") or "Object"
        name = parameters.get("name") or "arg0"
        return f"{type_hint} {name}"
    if isinstance(parameters, Iterable):
        parts: List[str] = []
        for idx, entry in enumerate(parameters):
            if isinstance(entry, dict):
                entry_type = entry.get("type") or "Object"
                entry_name = entry.get("name") or f"arg{idx}"
                parts.append(f"{entry_type} {entry_name}")
            else:
                text = str(entry).strip()
                if text:
                    parts.append(text)
        return ", ".join(parts)
    return ""


def _decode_plan_string(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""
    if stripped.startswith(('{', '[')) or (
        stripped.startswith('"') and stripped.endswith('"')
    ):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    if any(seq in stripped for seq in ("\\n", "\\t", "\\r", '\\"')):
        try:
            return bytes(stripped, "utf-8").decode("unicode_escape")
        except Exception:  # pragma: no cover - best effort only
            return stripped
    return stripped


def _format_plan_text(text: str) -> str:
    normalized = text.strip()
    if "\n" in normalized:
        return normalized
    tokens = re.split(r"(\{|\}|;)", normalized)
    indent = 0
    lines: List[str] = []
    buffer: List[str] = []

    def flush_buffer() -> None:
        if buffer:
            line = "".join(buffer).strip()
            if line:
                lines.append((" " * 4 * indent) + line)
            buffer.clear()

    for token in tokens:
        if token is None:
            continue
        stripped = token.strip()
        if not stripped:
            continue
        if stripped == ";":
            buffer.append(";")
            flush_buffer()
        elif stripped == "{":
            flush_buffer()
            lines.append((" " * 4 * indent) + "{")
            indent += 1
        elif stripped == "}":
            flush_buffer()
            indent = max(indent - 1, 0)
            lines.append((" " * 4 * indent) + "}")
        else:
            if buffer and not buffer[-1].endswith(" "):
                buffer.append(" ")
            buffer.append(stripped)
    flush_buffer()
    return "\n".join(lines)


def _stringify_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (dict, list)):
        try:
            return json.dumps(content, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(content)
    return str(content)


def _estimate_token_count(text: str) -> int:
    if not text:
        return 0
    return len(_TOKEN_COUNT_PATTERN.findall(text))


def _current_timestamp_for_llm_log() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _format_llm_log_block(
    *,
    plan_id: str,
    phase: str,
    role: str,
    token_label: str,
    token_count: Optional[int],
    content: str,
) -> str:
    timestamp = _current_timestamp_for_llm_log()
    normalized_role = (role or "unknown").title()
    count_display = str(token_count) if token_count is not None else "n/a"
    header = (
        f"[{timestamp}] {phase} - Role {normalized_role} - {token_label} [{count_display}] "
        f"plan_id={plan_id}"
    )
    body = content.rstrip("\n")
    if not body:
        body = "(empty)"
    return f"{header}\n>>>\n{body}\n<<<"


def _log_llm_request(plan_id: str, messages: List[Dict[str, Any]], request: JavaPlanRequest) -> None:
    del request  # unused but kept for parity with future metadata hooks
    for message in messages:
        role = str(message.get("role", "unknown"))
        content = _stringify_message_content(message.get("content"))
        block = _format_llm_log_block(
            plan_id=plan_id,
            phase="Request",
            role=role,
            token_label="Input Token Count",
            token_count=_estimate_token_count(content),
            content=content,
        )
        llm_logger.info(block)


def _log_llm_response(plan_id: str, content: str, role: str = "assistant") -> None:
    block = _format_llm_log_block(
        plan_id=plan_id,
        phase="Request",
        role=role,
        token_label="Output Token Count",
        token_count=_estimate_token_count(content),
        content=content,
    )
    llm_logger.info(block)


def _parse_plan_content_string(text: str) -> Optional[str]:
    if not text:
        return None
    decoded = _decode_plan_string(text)
    if isinstance(decoded, dict):
        plan = _extract_plan_from_dict(decoded)
        if plan:
            return plan
        return None
    if isinstance(decoded, str):
        stripped = decoded.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
        if isinstance(parsed, dict):
            return _extract_plan_from_dict(parsed) or stripped
        return stripped
    return None


class JavaPlanningError(RuntimeError):
    """Raised when Java plan synthesis fails."""


@dataclass
class JavaPlanRequest:
    """Inputs that describe what the planner should generate."""

    task: str
    goals: Sequence[str] = field(default_factory=list)
    context: Optional[str] = None
    tool_names: Sequence[str] = field(default_factory=list)
    tool_schemas: Sequence[Dict[str, Any]] = field(default_factory=list)
    additional_constraints: Sequence[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    include_deferred_guidance: bool = True
    tool_stub_source: Optional[str] = None
    tool_stub_class_name: Optional[str] = None
    prior_plan_source: Optional[str] = None
    compile_error_report: Optional[str] = None
    refinement_iteration: int = 0


@dataclass
class JavaPlanResult:
    """Structured result returned by :class:`JavaPlanner`."""

    plan_id: str
    plan_source: str
    raw_response: Dict[str, Any]
    prompt_messages: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    prompt_hash: Optional[str] = None


class _PlannerToolPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _coerce_plain_java(cls, value: Any) -> Any:  # noqa: ANN401 - pydantic hook
        if isinstance(value, str):
            decoded = _decode_plan_string(value)
            if isinstance(decoded, dict):
                ast_rendered = _render_java_from_structured_payload(decoded)
                if ast_rendered:
                    coerced = dict(decoded)
                    coerced["java"] = ast_rendered
                    return coerced
                extracted = _extract_plan_text(decoded)
                if extracted:
                    coerced = dict(decoded)
                    coerced["java"] = extracted
                    return coerced
            elif isinstance(decoded, str):
                value = decoded
            return {"java": value}
        if isinstance(value, dict):
            ast_rendered = _render_java_from_structured_payload(value)
            if ast_rendered:
                coerced = dict(value)
                coerced["java"] = ast_rendered
                return coerced
            java_value = value.get("java")
            if isinstance(java_value, str) and java_value.strip():
                return value
            extracted = _extract_plan_text(value)
            if extracted:
                coerced = dict(value)
                coerced["java"] = extracted
                return coerced
        return value

    notes: Optional[str] = Field(
        default=None,
        description="Optional commentary about assumptions, risks, or TODOs.",
    )
    java: str = Field(
        ..., description="Complete Java source code containing exactly one top-level class."
    )

    @field_validator("java")
    @classmethod
    def _ensure_plan_block(cls, value: str) -> str:
        try:
            return _normalize_java_source(value)
        except ValueError as exc:  # pragma: no cover - validated downstream
            raise ValueError(str(exc)) from exc

    @field_validator("notes", mode="before")
    @classmethod
    def _normalize_notes(cls, value: Optional[Any]) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, list):
            normalized = [str(item).strip() for item in value if str(item).strip()]
            if not normalized:
                return None
            return "\n\n".join(normalized)
        return str(value)


class JavaPlanner:
    """High-level helper that asks the LLM to emit a Java plan."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        specification: Optional[str] = None,
        specification_path: Optional[Path] = None,
        structured_max_retries: Optional[int] = None,
    ):
        self._llm_client = llm_client
        self._specification = self._load_specification(specification, specification_path)
        self._planner_tool_schema = self._build_planner_tool_schema()
        self._planner_tool_choice = {
            "type": "function",
            "function": {"name": _PLANNER_TOOL_NAME},
        }
        self._structured_max_retries = self._resolve_structured_retry_limit(structured_max_retries)
        
    def _resolve_structured_retry_limit(self, explicit: Optional[int]) -> int:
        if explicit is not None:
            value = explicit
        else:
            env_limit = _read_structured_retry_limit()
            if env_limit is not None:
                value = env_limit
            else:
                provider_limit = (
                    getattr(self._llm_client.provider, "max_retries", None)
                    or getattr(self._llm_client.provider, "default_retries", None)
                )
                if provider_limit is None or provider_limit <= 0:
                    value = 2
                    logger.info(
                        "Planner structured retries defaulting to %s attempts (provider did not specify)",
                        value,
                    )
                else:
                    value = provider_limit
        if value < 0:
            raise ValueError("structured_max_retries must be >= 0")
        return value

    def generate_plan(self, request: JavaPlanRequest) -> JavaPlanResult:
        messages = self._build_messages(request)
        plan_source: str
        raw_response: Dict[str, Any]
        notes: Optional[str] = None
        plan_id = str(request.metadata.get("plan_id") or uuid.uuid4())
        request_start = time.perf_counter()
        retries = self._structured_max_retries
        status_prefix = "[JavaPlanner]"
        print(
            f"{status_prefix} Requesting structured Java plan (tools={request.tool_names or ['none']}, retries={retries})",
            flush=True,
        )
        logger.info(
            "%s Requesting structured Java plan (tools=%s, retries=%s)",
            status_prefix,
            request.tool_names or ["none"],
            retries,
        )
        plan_logger.info(
            "planner_request plan_id=%s tools=%s retries=%s goals=%s context=%s",
            plan_id,
            request.tool_names or ["none"],
            retries,
            len(request.goals or []),
            bool(request.context),
        )
        prompt_hash = _compute_prompt_hash(messages)
        _log_llm_request(plan_id, messages, request)
        try:
            payload = self._llm_client.structured_generate(
                messages=messages,
                response_model=_PlannerToolPayload,
                tools=[self._planner_tool_schema],
                tool_choice=self._planner_tool_choice,
                max_retries=self._structured_max_retries,
                log_context={
                    "logger_name": PLAN_LOGGER_NAME,
                    "prefix": f"plan_id={plan_id}",
                    "completion_logger": lambda attempt, completion: _log_structured_attempt_completion(
                        plan_id,
                        attempt,
                        completion,
                    ),
                },
            )
        except Exception as exc:  # pragma: no cover - provider dependent
            elapsed = time.perf_counter() - request_start
            print(
                f"{status_prefix} Structured plan request failed after {elapsed:.1f}s: {exc}",
                flush=True,
            )
            logger.warning(
                "%s Structured plan request failed after %.1fs: %s",
                status_prefix,
                elapsed,
                exc,
            )
            plan_logger.warning(
                "planner_structured_failure plan_id=%s elapsed=%.1fs error=%s",
                plan_id,
                elapsed,
                exc,
            )
            _log_structured_failure_details(plan_id, exc)
            friendly = _summarize_structured_failure(exc)
            if friendly:
                logger.warning("%s Falling back to plain-text parsing.", friendly)
                plan_logger.warning(
                    "planner_structured_reason plan_id=%s detail=%s",
                    plan_id,
                    friendly,
                )
            else:
                logger.warning(
                    "Structured Java plan generation failed; attempting plain-text fallback.",
                    exc_info=exc,
                )
            plan_logger.info("planner_plain_fallback plan_id=%s", plan_id)
            failure_reason = friendly or str(exc)
            plan_source, raw_response, notes = self._generate_plain_plan(
                messages,
                plan_id=plan_id,
                failure_reason=failure_reason,
            )
        else:
            elapsed = time.perf_counter() - request_start
            print(
                f"{status_prefix} Structured plan ready in {elapsed:.1f}s (notes={bool(payload.notes)})",
                flush=True,
            )
            logger.info(
                "%s Structured plan ready in %.1fs (notes=%s)",
                status_prefix,
                elapsed,
                bool(payload.notes),
            )
            plan_logger.info(
                "planner_structured_success plan_id=%s elapsed=%.1fs notes=%s",
                plan_id,
                elapsed,
                bool(payload.notes),
            )
            plan_source = payload.java.strip()
            raw_response = payload.model_dump()
            if payload.notes:
                notes = payload.notes.strip()

        metadata = dict(request.metadata)
        metadata.setdefault("allowed_tools", sorted(request.tool_names))
        metadata.setdefault("plan_id", plan_id)
        metadata.setdefault("prompt_hash", prompt_hash)
        if notes:
            metadata["planner_notes"] = notes

        plan_logger.info("planner_plan_source plan_id=%s\n%s", plan_id, plan_source)
        _log_llm_response(plan_id, plan_source)
        return JavaPlanResult(
            plan_id=plan_id,
            plan_source=plan_source,
            raw_response=raw_response,
            prompt_messages=messages,
            metadata=metadata,
            prompt_hash=prompt_hash,
        )

    def _generate_plain_plan(
        self,
        messages: List[Dict[str, Any]],
        *,
        plan_id: str,
        failure_reason: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any], Optional[str]]:
        """Fallback path when structured tool calls are unavailable."""

        current_messages = self._build_plain_fallback_messages(messages, failure_reason)
        attempts = _PLAIN_FALLBACK_MAX_ATTEMPTS
        last_error: Optional[JavaPlanningError] = None

        for attempt in range(1, attempts + 1):
            response = self._llm_client.generate(
                messages=current_messages,
                tools=[self._planner_tool_schema],
                tool_choice=self._planner_tool_choice,
            )
            try:
                return self._parse_plain_response(response)
            except JavaPlanningError as exc:
                last_error = exc
                logger.warning(
                    "Plain-text Java plan attempt %s/%s failed (plan_id=%s): %s",
                    attempt,
                    attempts,
                    plan_id,
                    exc,
                )
                plan_logger.warning(
                    "planner_plain_retry_failure plan_id=%s attempt=%s error=%s",
                    plan_id,
                    attempt,
                    exc,
                )
                if attempt == attempts:
                    break
                current_messages = self._append_plain_retry_prompt(
                    current_messages,
                    failure_reason=str(exc),
                    attempt_label=attempt + 1,
                    plan_id=plan_id,
                )

        raise last_error if last_error else JavaPlanningError("Planner returned invalid Java.")

    def _parse_plain_response(
        self, response: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any], Optional[str]]:
        if not isinstance(response, dict):
            raise JavaPlanningError("Planner returned an unexpected response format.")

        notes: Optional[str] = None
        tool_calls = response.get("tool_calls") or []
        if len(tool_calls) == 1:
            call = tool_calls[0]
            function_meta = call.get("function", {})
            if function_meta.get("name") == _PLANNER_TOOL_NAME:
                arguments = function_meta.get("arguments") or "{}"
                try:
                    payload_data = json.loads(arguments)
                except (TypeError, json.JSONDecodeError) as exc:
                    raise JavaPlanningError(
                        "Planner tool call arguments were invalid JSON."
                    ) from exc
                java_source = payload_data.get("java")
                if not java_source:
                    raise JavaPlanningError("Planner tool call did not include Java source.")
                try:
                    normalized = _normalize_java_source(java_source)
                except ValueError as exc:
                    raise JavaPlanningError(f"Planner returned invalid Java: {exc}") from exc
                raw_notes = payload_data.get("notes")
                if raw_notes is not None:
                    notes = _PlannerToolPayload._normalize_notes(raw_notes)
                return normalized, response, notes

        content = response.get("content")
        if content is None or not str(content).strip():
            raise JavaPlanningError("Planner returned empty content.")
        try:
            normalized = _normalize_java_source(str(content))
        except ValueError as exc:
            raise JavaPlanningError(f"Planner returned invalid Java: {exc}") from exc
        return normalized, response, None

    def _build_plain_fallback_messages(
        self,
        base_messages: List[Dict[str, Any]],
        failure_reason: Optional[str],
    ) -> List[Dict[str, Any]]:
        prompt = self._format_plain_followup_prompt(
            failure_reason,
            is_retry=False,
        )
        return list(base_messages) + [{"role": "user", "content": prompt}]

    def _append_plain_retry_prompt(
        self,
        current_messages: List[Dict[str, Any]],
        *,
        failure_reason: Optional[str],
        attempt_label: int,
        plan_id: str,
    ) -> List[Dict[str, Any]]:
        prompt = self._format_plain_followup_prompt(failure_reason, is_retry=True)
        plan_logger.info(
            "planner_plain_retry_prompt plan_id=%s attempt=%s reason=%s",
            plan_id,
            attempt_label,
            failure_reason,
        )
        return list(current_messages) + [{"role": "user", "content": prompt}]

    @staticmethod
    def _format_plain_followup_prompt(
        failure_reason: Optional[str],
        *,
        is_retry: bool,
    ) -> str:
        truncated = ""
        if failure_reason:
            truncated = textwrap.shorten(failure_reason.strip(), width=280, placeholder="…")
        if is_retry:
            header = "Your prior response was empty or malformed; output the Java class verbatim below."
        else:
            header = "Structured tool calls failed; output the Java class verbatim below."
        instructions = (
            "Return only the Java source for a single class named Planner, without JSON, markdown,"
            " or commentary. Continue calling PlanningToolStubs.<toolName>(...) exactly as defined."
        )
        if truncated:
            details = f"Previous error: {truncated}"
            return "\n".join([header, details, instructions]).strip()
        return "\n".join([header, instructions]).strip()

    def _build_messages(self, request: JavaPlanRequest) -> List[Dict[str, Any]]:
        has_tools = bool(request.tool_stub_source)
        constraint_lines = self._build_constraints(request, has_tools)
        system_content = self._build_system_message(
            request.tool_stub_source,
            request.tool_stub_class_name,
            request.tool_names,
        )
        user_content = self._build_user_message(request, constraint_lines)
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _build_system_message(
        self,
        tool_stub_source: Optional[str] = None,
        tool_stub_class_name: Optional[str] = None,
        tool_names: Sequence[str] = (),
    ) -> str:
        header = (
            "You are the experienced Java developer."
            " Produce a single Java class that fully solves the user's task by decomposing the problem into simpler parts."
            " It should consist of a single main method calls at most seven other functions."
            " The only helper classes available are those provided in the PlanningToolsStub code."
            " These functions can be called directly to perform specific subtasks. Do not invent new APIs."
            " If the problem cannot be solved directly with the available tools, stub out the functions called with comments, indicating what they should do."
        )
        schema_guidance = textwrap.dedent(
            """
            When the runtime requests structured output, you must emit a single JSON object matching:
            {
              "java": "public class Plan { ... }",
              "notes": "Optional commentary about risks or TODOs (omit or null when unused)"
            }
            Do not introduce additional keys, arrays, or wrapper text around this object.
            """
        ).strip()
        message = (
            f"{header}\n\n{schema_guidance}\n\n"
        ).strip()

        if tool_names:
            summary_payload = {
                "available_tools": sorted(tool_names),
                "tool_stub_class": tool_stub_class_name or "PlanningToolStubs",
            }
            tool_metadata = json.dumps(summary_payload, ensure_ascii=False, indent=2)
            message = (
                f"{message}\n\nMachine-readable tool summary:\n{tool_metadata}"
            ).strip()

        if tool_stub_source:
            stub_intro = [
                "Tool stub reference: Use the provided Java class when calling tools.",
            ]
            if tool_stub_class_name:
                stub_intro.append(
                    f"Call the static methods defined on `{tool_stub_class_name}` instead of inventing new signatures."
                )
            stub_intro_text = " ".join(stub_intro).strip()
            stub_block = textwrap.dedent(
                f"""
                {stub_intro_text}

                <tool_stubs>
                {tool_stub_source.strip()}
                </tool_stubs>
                """
            ).strip()
            message = f"{message}\n\n{stub_block}".strip()

        return message

    def _build_user_message(
        self,
        request: JavaPlanRequest,
        constraint_lines: Sequence[str],
    ) -> str:
        lines: List[str] = []
        lines.append("Task:")
        lines.append(textwrap.dedent(request.task).strip())
        lines.append("")

        if request.goals:
            lines.append("Goals:")
            for idx, goal in enumerate(request.goals, start=1):
                lines.append(f"{idx}. {goal}")
            lines.append("")

        if request.context:
            lines.append("Context:")
            lines.append(textwrap.dedent(request.context).strip())
            lines.append("")

        if request.tool_stub_class_name:
            lines.append(
                f"Use the static methods on {request.tool_stub_class_name} to invoke these tools; do not invent other APIs."
            )
            lines.append("")

        lines.append("Constraints:")
        for rule in constraint_lines:
            lines.append(f"- {rule}")
        if request.additional_constraints:
            for rule in request.additional_constraints:
                lines.append(f"- {rule}")
        lines.append("")

        if request.prior_plan_source:
            lines.append("Previous plan attempt:")
            lines.append("```java")
            lines.append(request.prior_plan_source.strip())
            lines.append("```")
            lines.append("")

        if request.compile_error_report:
            lines.append("Compile diagnostics:")
            lines.append(request.compile_error_report.strip())
            lines.append("")
            lines.append(
                "Revise the plan to satisfy the Java planning specification and resolve each diagnostic above."
            )
            lines.append("")

        lines.append(
            "Output requirements: respond with only the Java source, preferably via the"
            f" {_PLANNER_TOOL_NAME} function when tool calls are supported."
        )
        return "\n".join(lines).strip()

    def _build_constraints(
        self,
        request: JavaPlanRequest,
        has_tools: bool,
    ) -> List[str]:
        stub_name = request.tool_stub_class_name or "PlanningToolStubs"
        constraints = [
            "Emit exactly one top-level Java class (any name) with helper methods and a main() entrypoint when needed.",
            f"Call tools exclusively via the `{stub_name}.<name>(...)` static helpers; never invent new APIs.",
            "Limit every helper body to seven statements and ensure each helper is more specific than its caller.",
            "Stick to the allowed statement types (variable declarations, assignments, helper/tool calls, if/else, enhanced for, try/catch, returns).",
            "Do not wrap the output in markdown; Java comments and imports are allowed but avoid prose explanations.",
        ]
        if not has_tools:
            constraints.append(
                "If no planning tools are available, describe diagnostic steps using logging and TODOs."
            )
        return constraints

    @staticmethod
    def _load_specification(
        override_content: Optional[str],
        override_path: Optional[Path],
    ) -> str:
        if override_content:
            return override_content.strip()
        if override_path:
            try:
                return override_path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise JavaPlanningError(
                    f"Failed to load plan specification from {override_path}"
                ) from exc
        logger.info("Planner specification not provided; using embedded guidance.")
        return _EMBEDDED_SPEC

    def _build_planner_tool_schema(self) -> Dict[str, Any]:
        description_lines = [
            "Return the final Java plan along with any helpful notes.",
            "Every response must comply with this specification:",
            self._specification,
        ]
        return {
            "type": "function",
            "function": {
                "name": _PLANNER_TOOL_NAME,
                "description": "\n\n".join(description_lines).strip(),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "notes": {
                            "type": "string",
                            "description": "Optional commentary about assumptions, risks, or TODOs.",
                        },
                        "java": {
                            "type": "string",
                            "description": "Complete Java source code containing a single top-level class.",
                        },
                    },
                    "required": ["java"],
                },
            },
        }


__all__ = [
    "JavaPlanRequest",
    "JavaPlanResult",
    "JavaPlanner",
    "JavaPlanningError",
]
