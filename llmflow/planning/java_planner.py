"""Utilities for prompting the LLM to emit Java plans."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import textwrap
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Match, Optional, Sequence, Tuple

from llmflow.llm_client import LLMClient
from llmflow.logging_utils import LLM_LOGGER_NAME, PLAN_LOGGER_NAME

_EMBEDDED_SPEC = (
    "You coordinate goal-driven Java automation. Return exactly one top-level class named Planner "
    "that calls PlanningToolStubs helpers to perform every side effect. Prefer helper decomposition, "
    "avoid markdown, and emit compilable Java source."
)
_PLAIN_FALLBACK_MAX_ATTEMPTS = 2
_PLANNER_TOOL_NAME = "define_java_plan"

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
_MIN_MEANINGFUL_PLAN_LINES = 3
_ELLIPSIS_BLOCK_PATTERN = re.compile(r"\{\s*\.\.\.\s*\}")
_ESCAPE_SEQUENCE_PATTERN = re.compile(r"\\(u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2}|[\\\"'nrtbf])")


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
    normalized = candidate.strip()
    _validate_plan_structure(normalized)
    return normalized


def _validate_plan_structure(source: str) -> None:
    if _ELLIPSIS_BLOCK_PATTERN.search(source):
        raise ValueError("Java payload contains placeholder ellipses; provide actual code.")
    if _count_meaningful_lines(source) < _MIN_MEANINGFUL_PLAN_LINES:
        raise ValueError(
            f"Java payload must include at least {_MIN_MEANINGFUL_PLAN_LINES} meaningful lines of code."
        )


def _count_meaningful_lines(source: str) -> int:
    count = 0
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if all(ch in "{}();," for ch in stripped):
            continue
        count += 1
        if count >= _MIN_MEANINGFUL_PLAN_LINES:
            break
    return count


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
        extracted = _extract_plan_from_dict(decoded)
        if extracted:
            candidate = extracted
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


def _normalize_planner_notes(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list):
        normalized = [str(item).strip() for item in value if str(item).strip()]
        if not normalized:
            return None
        return "\n\n".join(normalized)
    text = str(value).strip()
    return text or None


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


def _extract_java_string(payload: Dict[str, Any]) -> Optional[str]:
    for key in ("java", "java_code", "code", "source"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            decoded = _decode_plan_string(value)
            if isinstance(decoded, str):
                stripped = decoded.strip()
                if stripped:
                    return stripped
    return None


def _extract_plan_from_dict(payload: Dict[str, Any]) -> Optional[str]:
    direct_java = _extract_java_string(payload)
    if direct_java:
        return direct_java

    tool_payload = payload.get(_PLANNER_TOOL_NAME)
    if isinstance(tool_payload, dict):
        embedded = _extract_java_string(tool_payload)
        if embedded:
            return embedded
    elif isinstance(tool_payload, str) and tool_payload.strip():
        decoded = _decode_plan_string(tool_payload)
        if isinstance(decoded, dict):
            embedded = _extract_java_string(decoded)
            if embedded:
                return embedded
        elif isinstance(decoded, str) and decoded.strip():
            return decoded.strip()

    defines_block = payload.get("defines")
    if isinstance(defines_block, dict):
        defined_java = _extract_java_string(defines_block)
        if defined_java:
            return defined_java

    for key in ("choices", "message", "messages", "delta", "output"):
        value = payload.get(key)
        text = _extract_plan_text(value)
        if text:
            return text

    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            parsed = _parse_plan_content_string(content)
            if parsed:
                return parsed

    return None


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
        decoded = _try_unicode_escape_decode(stripped)
        if decoded is not None:
            return decoded
        return _best_effort_unescape_literals(stripped)
    return stripped


def _try_unicode_escape_decode(text: str) -> Optional[str]:
    try:
        return bytes(text, "utf-8").decode("unicode_escape")
    except Exception:  # pragma: no cover - best effort only
        return None


def _best_effort_unescape_literals(text: str) -> str:
    mapping = {
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "b": "\b",
        "f": "\f",
        "\\": "\\",
        '"': '"',
        "'": "'",
    }

    def _replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if not token:
            return match.group(0)
        if token.startswith("u") and len(token) == 5:
            try:
                return chr(int(token[1:], 16))
            except ValueError:
                return match.group(0)
        if token.startswith("x") and len(token) == 3:
            try:
                return chr(int(token[1:], 16))
            except ValueError:
                return match.group(0)
        return mapping.get(token, match.group(0))

    return _ESCAPE_SEQUENCE_PATTERN.sub(_replace, text)


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
    context: Optional[str] = None
    goals: Sequence[str] = field(default_factory=list)
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
    """Result returned by :class:`JavaPlanner`."""

    plan_id: str
    plan_source: str
    raw_response: Dict[str, Any]
    prompt_messages: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    prompt_hash: Optional[str] = None

class JavaPlanner:
    """High-level helper that asks the LLM to emit a Java plan."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        specification: Optional[str] = None,
        specification_path: Optional[Path] = None,
    ):
        self._llm_client = llm_client
        self._specification = self._load_specification(specification, specification_path)
        self._planner_tool_schema = self._build_planner_tool_schema()
        self._planner_tool_choice = {
            "type": "function",
            "function": {"name": _PLANNER_TOOL_NAME},
        }
        
    def generate_plan(self, request: JavaPlanRequest) -> JavaPlanResult:
        messages = self._build_messages(request)
        plan_source: str
        raw_response: Dict[str, Any]
        notes: Optional[str] = None
        plan_id = str(request.metadata.get("plan_id") or uuid.uuid4())
        request_start = time.perf_counter()
        status_prefix = "[JavaPlanner]"
        print(
            f"{status_prefix} Generating Java plan via plain-text mode (tools={request.tool_names or ['none']}).",
            flush=True,
        )
        logger.info(
            "%s Generating Java plan via plain-text mode (tools=%s)",
            status_prefix,
            request.tool_names or ["none"],
        )
        plan_logger.info(
            "planner_request plan_id=%s tools=%s context=%s generation_mode=plain",
            plan_id,
            request.tool_names or ["none"],
            bool(request.context),
        )
        prompt_hash = _compute_prompt_hash(messages)
        _log_llm_request(plan_id, messages, request)
        plan_source, raw_response, notes = self._generate_plain_plan(
            request,
            messages,
            plan_id=plan_id,
        )
        elapsed = time.perf_counter() - request_start
        print(
            f"{status_prefix} Plain-text plan ready in {elapsed:.1f}s (notes={bool(notes)})",
            flush=True,
        )
        logger.info(
            "%s Plain-text plan ready in %.1fs (notes=%s)",
            status_prefix,
            elapsed,
            bool(notes),
        )
        plan_logger.info(
            "planner_plain_success plan_id=%s elapsed=%.1fs notes=%s",
            plan_id,
            elapsed,
            bool(notes),
        )

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

    def compute_prompt_hash(self, request: JavaPlanRequest) -> str:
        messages = self._build_messages(request)
        return _compute_prompt_hash(messages)

    def _generate_plain_plan(
        self,
        request: JavaPlanRequest,
        messages: List[Dict[str, Any]],
        *,
        plan_id: str,
        failure_reason: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any], Optional[str]]:
        """Generate a plan using repeated plain-text attempts."""

        current_messages = self._build_plain_fallback_messages(request, messages, failure_reason)
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
                    request,
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
                    notes = _normalize_planner_notes(raw_notes)
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
        request: JavaPlanRequest,
        base_messages: List[Dict[str, Any]],
        failure_reason: Optional[str],
    ) -> List[Dict[str, Any]]:
        prompt = self._format_plain_followup_prompt(
            request,
            failure_reason,
            is_retry=False,
        )
        return list(base_messages) + [{"role": "user", "content": prompt}]

    def _append_plain_retry_prompt(
        self,
        request: JavaPlanRequest,
        current_messages: List[Dict[str, Any]],
        *,
        failure_reason: Optional[str],
        attempt_label: int,
        plan_id: str,
    ) -> List[Dict[str, Any]]:
        prompt = self._format_plain_followup_prompt(request, failure_reason, is_retry=True)
        plan_logger.info(
            "planner_plain_retry_prompt plan_id=%s attempt=%s reason=%s",
            plan_id,
            attempt_label,
            failure_reason,
        )
        return list(current_messages) + [{"role": "user", "content": prompt}]

    def _format_plain_followup_prompt(
        self,
        request: JavaPlanRequest,
        failure_reason: Optional[str],
        *,
        is_retry: bool,
    ) -> str:
        truncated = ""
        if failure_reason:
            truncated = textwrap.shorten(failure_reason.strip(), width=280, placeholder="…")
        helper_focus = (request.metadata or {}).get("helper_focus") or {}
        helper_name = helper_focus.get("function") if isinstance(helper_focus, dict) else None
        helper_comment = helper_focus.get("comment") if isinstance(helper_focus, dict) else None
        helper_message = helper_focus.get("message") if isinstance(helper_focus, dict) else None

        if helper_name:
            header = (
                "Your prior response was empty or malformed; re-emit Planner.java with a concrete implementation for"
                if is_retry
                else "Re-emit Planner.java with a concrete implementation for"
            )
            header = f"{header} '{helper_name}'."
        else:
            header = (
                "Your prior response was empty or malformed; output the Java class verbatim below."
                if is_retry
                else "Output the Java class verbatim below."
            )

        sections: List[str] = [header]
        if truncated:
            sections.append(f"Previous error: {truncated}")

        raw_task = textwrap.dedent(request.task or "").strip()
        if raw_task:
            task_summary = textwrap.shorten(" ".join(raw_task.split()), width=320, placeholder="…")
        else:
            task_summary = "Fulfill the previously described task."

        sections.append(f"Task reminder: {task_summary}")

        if helper_name:
            sections.append(
                f"Implement '{helper_name}' with PlanningToolStubs helpers, breaking the helper into concrete tool-driven steps."
            )
            if helper_comment:
                sections.append(helper_comment.strip())
            if helper_message and helper_message.strip() != helper_comment:
                sections.append(helper_message.strip())

        return "\n".join(section for section in sections if section).strip()

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
            " Produce a single, complete Java class that fully solves the user's task by decomposing the problem into simpler parts."
            " It should consist of a single main method calls at most seven other functions."
            " The only helper classes available are those provided in the PlanningToolsStub code."
            " These functions can be called directly to perform specific subtasks. Do not invent new APIs."
            " Provide concrete method bodies—do not return ellipses (...), placeholders, or empty classes."
            " If the problem cannot be solved directly with the available tools, stub out the functions called with comments, indicating what they should do."
        )
        schema_guidance = textwrap.dedent(
            f"""
            When you finalize the plan, call the `{_PLANNER_TOOL_NAME}` function with a JSON object matching:
            {{
              "java": "public class Planner {{ ... }}",
              "notes": "Optional commentary about risks or TODOs (omit or null when unused)"
            }}
            Do not introduce additional keys, arrays, or wrapper text around this object.
            """
        ).strip()
        message = (
            f"{header}\n\n{schema_guidance}\n\n"
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
