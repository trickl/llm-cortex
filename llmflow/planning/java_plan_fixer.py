"""LLM-powered fixer that refines Java plans on disk."""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field
from diff_match_patch import diff_match_patch

from llmflow.logging_utils import LLM_LOGGER_NAME, PLAN_LOGGER_NAME

from .artifact_layout import (
    ensure_prompt_artifact_dir,
    format_artifact_attempt_label,
    persist_compile_artifacts,
)
from .java_plan_compiler import (
    CompilationError,
    JavaCompilationResult,
    JavaPlanCompiler,
)
_NOTES_MARKERS: Tuple[Tuple[str, Optional[str]], ...] = (
    ("<<<NOTES>>>", None),
    ("```NOTES", "```"),
    ("NOTES:", None),
)
_DEFAULT_CONTEXT_LINES = 3
_PLAN_CLASS_PATTERN = re.compile(r"(?:class|interface)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)")
_HUNK_HEADER_PATTERN = re.compile(
    r"@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)
_ERROR_LINE_WINDOW = 5
_DIAGNOSTIC_PREFIXES = (
    "^",
    "symbol:",
    "location:",
    "error:",
    "required:",
    "found:",
    "note:",
)
_DIAGNOSTIC_PATTERNS = (
    re.compile(r"[A-Za-z0-9_./\\-]+\.(java|kt|scala):\d+:"),
    re.compile(r"\.java:\d+:\s*(error|warning|note|symbol|location)\b", re.IGNORECASE),
)
_SKIPPED_PATCH_LOG = "patch_skipped.log"
_METHOD_DECL_PATTERN = re.compile(
    r"^\s*(public|protected|private)?\s*(static\s+)?[\w\<\>\[\]]+\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^;]*\)\s*(?:throws\s+[A-Za-z0-9_,\s.]+)?\s*\{?\s*$"
)
_CLASS_DECL_PATTERN = re.compile(r"^\s*(public\s+)?(class|interface|enum)\b")


@dataclass
class _DiffSnippet:
    file_path: str
    content: str
    old_start: Optional[int]
    old_count: Optional[int]
    new_start: Optional[int]
    new_count: Optional[int]

STRICT_UNIFIED_DIFF_SPEC = textwrap.dedent(
    """
    System Instruction: Strict Unified Diff Output Specification

    The assistant must follow all of the rules in this specification whenever it is asked to modify code, propose changes, or produce patches. These rules are normative and override all other instructions unless explicitly relaxed.

    1. Output Format Requirements
    When generating changes, the assistant must output only a valid unified diff, with:
    - No Markdown code fences (no ```diff, no ``` at all)
    - No natural language, commentary, or explanation
    - No JSON, YAML, or metadata
    - No surrounding prose before or after the diff

    The output must be a plain text unified diff. Nothing else is permitted.

    2. Allowed Line Prefixes
    A correct unified diff may contain only lines beginning with one of the following prefixes:
    - `---` (old filename)
    - `+++` (new filename)
    - `@@` (hunk header)
    - `-` (removed line)
    - `+` (added line)
    - ` ` (space: unchanged context line)

    Any output that includes other leading characters is invalid.

    3. File Header Rules
    Each patch must begin with:
    --- <old_filename>
    +++ <new_filename>

    Where `<old_filename>` is typically the file path prefixed with `a/`, and `<new_filename>` with `b/`. Absolute paths must not be used unless explicitly requested.

    4. Hunk Header Format
    Each change block (“hunk”) must have a header of the form:
    @@ -<old_start>,<old_count> +<new_start>,<new_count> @@

    The assistant must:
    - Provide best-effort correct line ranges
    - Ensure that each hunk header corresponds to the lines shown beneath it
    - Include at least one hunk header, even if changes occur near the start of the file
    If multiple hunks are present, each must have its own `@@` header.

    5. Line Semantics
    Inside each hunk:
    - Lines removed from the old version start with `-`
    - Lines added in the new version start with `+`
    - Lines unchanged start with a single space (` `)
    Each line must follow this pattern exactly, with no additional leading whitespace.

    6. Context Requirements
    Unless explicitly instructed otherwise:
    - Include at least one unchanged context line before and after each block of changes
    - More context is allowed, but context must be correct relative to the provided input content
    - If -U0 behavior is desired (no context), it must be explicitly requested by the user

    7. Multiple File Changes
    If changes span multiple files:
    - Emit a separate unified diff section per file
    - Each section must begin with its own `---` and `+++` headers
    - Concatenate sections one after another
    - Never wrap multiple diffs inside Markdown or other containers

    8. Output Exclusivity
    The assistant must not output anything except the diff, including:
    - No explanation of what the diff does
    - No summary
    - No additional lines before `---` or after the final hunk
    - No Markdown headings, bullets, or text
    If the user requests a diff, the diff must be the entire output.

    9. Validity Requirement
    All emitted diffs must be:
    - Syntactically valid unified diffs
    - Parseable by `patch`, `git apply`, or similar tools
    - Free of extraneous characters or markup
    - Faithful to the modifications the assistant intends to make
    If the assistant is unable to guarantee validity, it must output an empty string rather than a malformed diff.

    10. Safety: Forbidden Behaviors
    The assistant must not:
    - Produce speculative or fictional file content unless provided by the user
    - Invent filenames unless explicitly allowed
    - Produce inconsistently formatted diffs
    - Combine multiple output formats
    - Embed diffs inside Markdown fences, HTML tags, code blocks, or JSON
    - Prepend or append commentary such as “Here is your diff:” or “Done!”

    11. Example (for illustration only; never output examples when producing diffs):
    --- a/example.txt
    +++ b/example.txt
    @@ -1,3 +1,3 @@
    line one
    -line two
    +updated line two
    line three

    12. Summary Rule
    When asked for modifications: the assistant must output a unified diff and nothing else.
    When not asked for modifications: the assistant must not output a unified diff.
    """
).strip()


class _ContextPatch(BaseModel):
    before_context: str = Field(
        default="",
        description="Up to three unchanged lines immediately before the edit (exact text).",
    )
    current_code: str = Field(
        default="",
        description="Code currently present between the before/after context; empty for pure insertions.",
    )
    after_context: str = Field(
        default="",
        description="Up to three unchanged lines immediately after the edit (exact text).",
    )
    replacement_code: str = Field(
        ...,
        description="Code that should replace the current segment within the provided context.",
    )


@dataclass
class PlanFixerRequest:
    """Inputs required to repair a compiled Java plan."""

    plan_id: str
    plan_source: str
    prompt_hash: str
    task: str
    compile_errors: Sequence[CompilationError] = field(default_factory=list)
    tool_stub_source: Optional[str] = None
    tool_stub_class_name: Optional[str] = None


@dataclass
class PlanFixerResult:
    """Outputs produced by :class:`JavaPlanFixer`."""

    plan_source: str
    compile_result: JavaCompilationResult
    notes: Optional[str] = None


class JavaPlanFixer:
    """Use the configured LLM to repair compiler errors in-place."""

    def __init__(
        self,
        llm_client,
        *,
        max_iterations: int = 15,
        max_retries: int = 1,
        artifact_root: Optional[Path] = None,
        plan_compiler: Optional[JavaPlanCompiler] = None,
        fix_request_max_attempts: int = 3,
        max_prompt_errors: int = 3,
    ) -> None:
        self._llm_client = llm_client
        self._max_iterations = max(1, max_iterations)
        self._max_retries = max(0, max_retries)
        self._max_fix_request_attempts = max(1, fix_request_max_attempts)
        self._artifact_root = Path(artifact_root or Path("plans"))
        self._plan_compiler = plan_compiler or JavaPlanCompiler()
        self._logger = PLAN_LOGGER_NAME
        self._max_prompt_errors = max(1, max_prompt_errors)

    def fix_plan(self, request: PlanFixerRequest, *, attempt: int) -> PlanFixerResult:
        prompt_dir = ensure_prompt_artifact_dir(
            self._artifact_root,
            request.prompt_hash,
            request.plan_id,
        )
        plan_source = request.plan_source
        notes: List[str] = []
        plan_logger = self._get_plan_logger()

        for iteration in range(1, self._max_iterations + 1):
            iteration_label = format_artifact_attempt_label(attempt, iteration)
            iteration_dir = prompt_dir / iteration_label
            compile_result = self._plan_compiler.compile(
                plan_source,
                tool_stub_source=request.tool_stub_source,
                tool_stub_class_name=request.tool_stub_class_name,
                working_dir=iteration_dir,
            )
            persist_compile_artifacts(iteration_dir, compile_result)
            error_count = len(compile_result.errors)
            plan_logger.info(
                "plan_fixer_compile plan_id=%s attempt=%s iteration=%s success=%s errors=%s dir=%s",
                request.plan_id,
                attempt,
                iteration,
                compile_result.success,
                error_count,
                iteration_dir,
            )
            if compile_result.success:
                return PlanFixerResult(
                    plan_source=plan_source,
                    compile_result=compile_result,
                    notes=self._merge_notes(notes),
                )
            if iteration == self._max_iterations:
                break
            plan_logger.info(
                "plan_fixer_llm_request plan_id=%s attempt=%s iteration=%s label=%s errors=%s",
                request.plan_id,
                attempt,
                iteration,
                iteration_label,
                error_count,
            )
            patches, patch_note, raw_text = self._request_fix(
                plan_source,
                request,
                compile_result,
                iteration_dir,
            )
            self._persist_patch_payload(iteration_dir, raw_text, patches)
            plan_source = self._apply_contextual_patches(plan_source, patches)
            if patch_note:
                notes.append(patch_note.strip())

        return PlanFixerResult(
            plan_source=plan_source,
            compile_result=compile_result,
            notes=self._merge_notes(notes),
        )

    def _request_fix(
        self,
        plan_source: str,
        request: PlanFixerRequest,
        compile_result: JavaCompilationResult,
        iteration_dir: Path,
    ) -> Tuple[List[_ContextPatch], Optional[str], str]:
        messages = self._build_messages(plan_source, request, compile_result)
        plan_logger = self._get_plan_logger()
        llm_logger = self._get_llm_logger()
        try:
            serialized = json.dumps(list(messages), ensure_ascii=False)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            serialized = str(messages)
        for logger in (plan_logger, llm_logger):
            logger.info(
                "plan_fixer_llm_messages plan_id=%s dir=%s payload=%s",
                request.plan_id,
                iteration_dir,
                serialized,
            )
        return self._request_patch_text(
            messages,
            plan_source,
            request,
            iteration_dir,
            compile_result.errors,
        )

    def _request_patch_text(
        self,
        messages: List[dict],
        plan_source: str,
        request: PlanFixerRequest,
        iteration_dir: Path,
        compile_errors: Sequence[CompilationError],
    ) -> Tuple[List[_ContextPatch], Optional[str], str]:
        plan_logger = self._get_plan_logger()
        llm_logger = self._get_llm_logger()
        plan_filename = self._plan_file_basename(plan_source)
        allowed_files = {plan_filename}
        line_windows = self._build_error_windows(
            compile_errors,
            plan_filename,
            plan_source,
        )
        skip_callback = lambda reason: self._record_patch_skip(iteration_dir, reason)
        last_error: Optional[Exception] = None
        for request_attempt in range(1, self._max_fix_request_attempts + 1):
            try:
                response = self._llm_client.generate(messages=messages)
            except Exception as exc:  # pragma: no cover - provider errors
                last_error = exc
                plan_logger.warning(
                    "plan_fixer_patch_retry plan_id=%s attempt=%s/%s error=%s",
                    request.plan_id,
                    request_attempt,
                    self._max_fix_request_attempts,
                    exc,
                )
                if request_attempt == self._max_fix_request_attempts:
                    raise RuntimeError(
                        "Plan fixer failed to obtain valid patch text."
                    ) from exc
                continue
            content = self._extract_response_content(response)
            if not content.strip():
                last_error = ValueError("LLM returned empty patch text")
            else:
                for logger in (plan_logger, llm_logger):
                    logger.info(
                        "plan_fixer_llm_response plan_id=%s dir=%s content=%s",
                        request.plan_id,
                        iteration_dir,
                        content,
                    )
                try:
                    patches, notes = self._parse_patch_text(
                        content,
                        plan_source,
                        allowed_filenames=allowed_files,
                        line_windows=line_windows,
                        skip_callback=skip_callback,
                    )
                    return patches, notes, content
                except ValueError as exc:
                    last_error = exc
            plan_logger.warning(
                "plan_fixer_patch_retry plan_id=%s attempt=%s/%s error=%s",
                request.plan_id,
                request_attempt,
                self._max_fix_request_attempts,
                last_error,
            )
        raise RuntimeError("Plan fixer failed to obtain valid patch text.") from last_error

    def _build_messages(
        self,
        plan_source: str,
        request: PlanFixerRequest,
        compile_result: JavaCompilationResult,
    ) -> List[dict]:
        system_prompt = self._build_system_prompt(request, plan_source)
        user_prompt = self._build_user_prompt(plan_source, request, compile_result)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _build_system_prompt(self, request: PlanFixerRequest, plan_source: str) -> str:
        plan_filename = self._plan_file_basename(plan_source)
        if request.tool_stub_source:
            stub_clause = "Use only the provided planning tool stubs; never invent additional helper classes."
        else:
            stub_clause = "The plan already references PlanningToolStubs; keep using the same static helpers."
        primary = " ".join(
            line.strip()
            for line in [
                "You repair a single Java class that orchestrates planning tools.",
                "Given the previous plan and the Java compiler diagnostics, produce localized contextual patches rather than rewriting the entire file.",
                stub_clause,
                f"Only edit {plan_filename}; never modify PlanningToolStubs.java or any other files, even if diagnostics mention them.",
                "Address diagnostics sequentially and emit at most one focused hunk per compiler error (resolve them in the order provided).",
                "Contexts must quote existing lines exactly as they already appear in the file (indentation, punctuation, and newline characters included). Never invent context from the desired replacement.",
                "Always include at least two unchanged lines at the top and bottom of each patch whenever they exist (use fewer only at file boundaries). Keep those boundary lines identical to the current file; only the interior lines should change.",
                "Preserve helper structure and minimize unrelated edits.",
                "If a method or helper does not exist (cannot find symbol on PlanningToolStubs class) add a stub method with the expected signature and a detailed comment describing its expected behavior.",
                "Import any required Java packages when introducing new types.",
                "Emit plain-text unified diffs only; do not wrap responses in markdown fences or commentary.",
                "Aim for the smallest number of focused patches that resolve all compilation errors.",
            ]
            if line.strip()
        ).strip()
        return f"{primary}\n\n{STRICT_UNIFIED_DIFF_SPEC}"

    def _build_user_prompt(
        self,
        plan_source: str,
        request: PlanFixerRequest,
        compile_result: JavaCompilationResult,
    ) -> str:
        sections: List[str] = []
        user_task = (request.task or "").strip()
        if user_task:
            sections.append(f"User request:\n{user_task}")
        plan_block = textwrap.dedent(
            f"""
            Previous plan:
            ```java
            {plan_source.strip()}
            ```
            """
        ).strip()
        sections.append(plan_block)
        prompt_errors = self._select_prompt_errors(compile_result.errors)
        errors_block = self._format_errors(prompt_errors)
        sections.append(f"Compiler diagnostics:\n{errors_block}")
        if request.tool_stub_source:
            sections.append(
                textwrap.dedent(
                    f"""
                    Tool stubs:
                    ```java
                    {request.tool_stub_source.strip()}
                    ```
                    """
                ).strip()
            )
        else:
            sections.append(
                "Existing plan already imports PlanningToolStubs. Continue to use the same static helpers."
            )
        sections.append(
            "Fix strategy:\n- Tackle the diagnostics in the order shown above.\n- Emit the smallest possible diff that resolves the current diagnostic before moving on.\n- Do not modify PlanningToolStubs or create additional helper files."
        )
        sections.append(
            "Requirements:\n- Keep the class self-contained.\n- Only adjust code necessary to resolve the diagnostics.\n- Maintain all PlanningToolStubs.<name>(...) calls.\n- Prefer the smallest number of focused patches.\n- Provide at least two unchanged lines above and below each edit whenever possible (use fewer only at file boundaries).\n- At the start of the file, include the first available unchanged lines after the edit; at the end of the file, include the preceding unchanged lines."
        )
        sections.append(
            "Follow the System Instruction: Strict Unified Diff Output Specification exactly. Output only the unified diff (no commentary, no markdown fences)."
        )
        return "\n\n".join(sections).strip()

    def _select_prompt_errors(
        self,
        errors: Sequence[CompilationError],
    ) -> List[CompilationError]:
        if not errors:
            return []
        sorted_errors = sorted(errors, key=self._error_position_key)
        return list(sorted_errors[: self._max_prompt_errors])

    @staticmethod
    def _error_position_key(error: CompilationError) -> Tuple[float, float]:
        line: float
        column: float
        line = float(error.line) if isinstance(error.line, int) else float("inf")
        column = float(error.column) if isinstance(error.column, int) else float("inf")
        return line, column

    @staticmethod
    def _format_errors(errors: Sequence[CompilationError]) -> str:
        if not errors:
            return "(Compiler produced no structured diagnostics; ensure the plan compiles.)"
        lines: List[str] = []
        for idx, error in enumerate(errors, start=1):
            location_parts: List[str] = []
            if error.file:
                location_parts.append(error.file)
            if error.line is not None:
                loc = f"line {error.line}"
                if error.column is not None:
                    loc += f", column {error.column}"
                location_parts.append(loc)
            location = f" ({'; '.join(location_parts)})" if location_parts else ""
            lines.append(f"{idx}. {error.message}{location}")
            if error.raw:
                lines.append(textwrap.indent(error.raw, "   "))
        return "\n".join(lines).strip()

    @staticmethod
    def _merge_notes(notes: Sequence[str]) -> Optional[str]:
        filtered = [note for note in (note.strip() for note in notes) if note]
        if not filtered:
            return None
        return "\n\n".join(filtered)

    def _persist_patch_payload(
        self,
        iteration_dir: Path,
        raw_text: str,
        patches: Sequence[_ContextPatch],
    ) -> None:
        iteration_dir.mkdir(parents=True, exist_ok=True)
        response_path = iteration_dir / "patch_response.txt"
        response_path.write_text(raw_text.rstrip() + "\n", encoding="utf-8")
        diff_sections = self._split_unified_diff_sections(raw_text)
        self._persist_pretty_patch_text(iteration_dir, diff_sections)

    @staticmethod
    def _plan_file_basename(plan_source: str) -> str:
        match = _PLAN_CLASS_PATTERN.search(plan_source)
        if not match:
            raise ValueError("Unable to determine planner class name from source.")
        return f"{match.group('name')}.java"

    @staticmethod
    def _snippet_line_range(snippet: _DiffSnippet) -> Optional[Tuple[int, int]]:
        start: Optional[int]
        count: Optional[int]
        if snippet.new_start is not None and snippet.new_count is not None and snippet.new_count > 0:
            start = snippet.new_start
            count = snippet.new_count
        elif snippet.old_start is not None and snippet.old_count is not None and snippet.old_count > 0:
            start = snippet.old_start
            count = snippet.old_count
        else:
            return None
        end = start + max(count - 1, 0)
        return start, end

    @staticmethod
    def _range_overlaps(snippet_range: Tuple[int, int], windows: Sequence[Tuple[int, int]]) -> bool:
        start, end = snippet_range
        for window_start, window_end in windows:
            if end < window_start or start > window_end:
                continue
            return True
        return False

    def _build_error_windows(
        self,
        errors: Sequence[CompilationError],
        plan_filename: str,
        plan_source: Optional[str] = None,
    ) -> List[Tuple[int, int]]:
        windows: List[Tuple[int, int]] = []
        plan_lines: Optional[List[str]] = None
        if plan_source:
            normalized = self._normalize_patch_text(plan_source)
            plan_lines = normalized.splitlines()
        for error in errors:
            if error.file and Path(error.file).name != plan_filename:
                continue
            if error.line is None:
                continue
            line_number = int(error.line)
            start = max(1, line_number - _ERROR_LINE_WINDOW)
            end = line_number + _ERROR_LINE_WINDOW
            windows.append((start, end))
            if plan_lines:
                method_window = self._method_window_for_line(plan_lines, line_number)
                if method_window:
                    windows.append(method_window)
        merged = self._merge_windows(windows)
        if merged:
            return merged
        if plan_lines:
            total_lines = len(plan_lines)
            if total_lines:
                return [(1, total_lines)]
        return []

    @staticmethod
    def _merge_windows(windows: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not windows:
            return []
        ordered = sorted((max(1, start), max(start, end)) for start, end in windows)
        merged: List[Tuple[int, int]] = [ordered[0]]
        for start, end in ordered[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end + 1:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged

    def _method_window_for_line(
        self,
        plan_lines: Sequence[str],
        line_number: int,
    ) -> Optional[Tuple[int, int]]:
        if line_number < 1 or line_number > len(plan_lines):
            return None
        signature_idx = self._locate_method_signature(plan_lines, line_number - 1)
        if signature_idx is None:
            return None
        block_start = self._locate_block_start(plan_lines, signature_idx)
        if block_start is None:
            return None
        block_end = self._locate_block_end(plan_lines, block_start)
        if block_end is None:
            return None
        return signature_idx + 1, block_end + 1

    @staticmethod
    def _locate_method_signature(plan_lines: Sequence[str], start_index: int) -> Optional[int]:
        idx = min(start_index, len(plan_lines) - 1)
        while idx >= 0:
            line = plan_lines[idx].strip()
            if not line or line.startswith("//"):
                idx -= 1
                continue
            if line.startswith("@"):
                idx -= 1
                continue
            if _METHOD_DECL_PATTERN.match(line) or _CLASS_DECL_PATTERN.match(line):
                return idx
            idx -= 1
        return None

    @staticmethod
    def _locate_block_start(plan_lines: Sequence[str], signature_index: int) -> Optional[int]:
        for idx in range(signature_index, len(plan_lines)):
            if "{" in plan_lines[idx]:
                return idx
        return None

    @staticmethod
    def _locate_block_end(plan_lines: Sequence[str], block_start: int) -> Optional[int]:
        depth = 0
        for idx in range(block_start, len(plan_lines)):
            line = plan_lines[idx]
            depth += line.count("{")
            depth -= line.count("}")
            if depth <= 0:
                return idx
        return len(plan_lines) - 1 if plan_lines else None

    def _record_patch_skip(self, iteration_dir: Path, reason: str) -> None:
        clean_reason = reason.strip()
        if not clean_reason:
            return
        iteration_dir.mkdir(parents=True, exist_ok=True)
        log_path = iteration_dir / _SKIPPED_PATCH_LOG
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(clean_reason + "\n")
        plan_logger = self._get_plan_logger()
        plan_logger.info("plan_fixer_patch_skipped dir=%s reason=%s", iteration_dir, clean_reason)

    @staticmethod
    def _apply_contextual_patches(plan_source: str, patches: Sequence[_ContextPatch]) -> str:
        updated = plan_source.replace("\r\n", "\n").replace("\r", "\n")
        import logging

        logger = logging.getLogger(PLAN_LOGGER_NAME)
        for index, patch in enumerate(patches, start=1):
            try:
                updated = JavaPlanFixer._apply_single_patch(updated, patch)
            except ValueError as exc:
                context_summary = JavaPlanFixer._summarize_patch_context(patch)
                logger.error(
                    "plan_fixer_patch_apply_failed patch_index=%s context=%s error=%s",
                    index,
                    context_summary,
                    exc,
                )
                raise ValueError(
                    f"Failed to apply contextual patch #{index}: {exc}"
                ) from exc
        return updated

    @staticmethod
    def _summarize_patch_context(patch: _ContextPatch, limit: int = 160) -> str:
        def _clip(value: str) -> str:
            normalized = (value or "").strip().replace("\n", "\\n")
            if len(normalized) > limit:
                return normalized[:limit] + "…"
            return normalized or "<empty>"

        return f"before={_clip(patch.before_context)} after={_clip(patch.after_context)}"

    @staticmethod
    def _apply_single_patch(plan_source: str, patch: _ContextPatch) -> str:
        before = JavaPlanFixer._normalize_patch_text(patch.before_context)
        after = JavaPlanFixer._normalize_patch_text(patch.after_context)
        replacement = JavaPlanFixer._normalize_patch_text(patch.replacement_code)
        if not before and not after:
            raise ValueError(
                "Contextual patches must include at least one of before_context or after_context."
            )
        _, between_start, after_start = JavaPlanFixer._locate_patch_window(plan_source, before, after)
        prefix = plan_source[:between_start]
        suffix = plan_source[after_start:]
        return prefix + replacement + suffix

    @staticmethod
    def _locate_patch_window(plan_source: str, before: str, after: str) -> Tuple[int, int, int]:
        normalized = plan_source
        before_entries: List[Tuple[int, int]] = [
            (idx, len(before)) for idx in JavaPlanFixer._context_start_indices(normalized, before)
        ]
        if not before_entries:
            if before:
                approx = JavaPlanFixer._approximate_context_index(normalized, before, 0)
                if approx is None and after:
                    after_idx = JavaPlanFixer._find_with_newline_fallback(normalized, after, 0)
                    if after_idx != -1:
                        before_entries = [(after_idx, 0)]
                if approx is None and not before_entries:
                    raise ValueError("Failed to locate before_context in plan source.")
                if approx is not None:
                    approx_start, approx_len = approx
                    if before and len(before) > 0:
                        coverage = approx_len / max(len(before), 1)
                        if coverage < 0.5:
                            approx_len = 0
                    before_entries = [(approx_start, approx_len)]
            else:
                before_entries = [(0, 0)]
        for before_start, matched_len in before_entries:
            between_start = before_start + matched_len
            if after:
                after_start = JavaPlanFixer._find_with_newline_fallback(
                    normalized,
                    after,
                    between_start,
                )
                if after_start == -1:
                    continue
            else:
                after_start = len(normalized)
            return before_start, between_start, after_start
        raise ValueError("Failed to locate patch context in plan source.")

    @staticmethod
    def _find_with_newline_fallback(haystack: str, needle: str, start: int) -> int:
        variants = [needle]
        if needle.endswith("\n"):
            trimmed = needle.rstrip("\n")
            if trimmed and trimmed not in variants:
                variants.append(trimmed)
        for variant in variants:
            if not variant:
                return start
            idx = haystack.find(variant, start)
            if idx != -1:
                return idx
        approx = JavaPlanFixer._approximate_context_index(haystack, needle, start)
        if approx is not None:
            return approx[0]
        return -1

    @staticmethod
    def _context_start_indices(haystack: str, needle: str) -> List[int]:
        if not needle:
            return [0]
        indices: List[int] = []
        start = 0
        while True:
            idx = haystack.find(needle, start)
            if idx == -1:
                break
            indices.append(idx)
            start = idx + 1
        return indices

    @staticmethod
    def _approximate_context_index(haystack: str, needle: str, hint: int) -> Optional[Tuple[int, int]]:
        if not needle.strip():
            return hint, 0
        matcher = diff_match_patch()
        matcher.Match_Distance = max(1024, len(haystack))
        matcher.Match_Threshold = 0.35
        candidates = JavaPlanFixer._approximate_candidates(needle)
        search_start = max(0, hint)
        for candidate in candidates:
            if not candidate:
                continue
            location = matcher.match_main(haystack, candidate, search_start)
            if location == -1:
                continue
            matched_len = JavaPlanFixer._match_suffix_length(haystack, needle, location)
            if matched_len <= 0:
                matched_len = min(len(candidate), len(haystack) - location)
            if matched_len > 0:
                return location, matched_len
        return None

    @staticmethod
    def _approximate_candidates(needle: str) -> List[str]:
        normalized = needle.replace("\r\n", "\n").replace("\r", "\n")
        stripped = normalized.strip("\n")
        base = stripped or normalized
        candidates: List[str] = []
        seen: Set[str] = set()

        def _add(value: str) -> None:
            if not value:
                return
            if value in seen:
                return
            seen.add(value)
            candidates.append(value)

        _add(base)
        if base != normalized:
            _add(normalized)
        chunk = 256
        if len(base) > chunk:
            _add(base[-chunk:])
            _add(base[:chunk])
        lines = [line for line in base.splitlines() if line.strip()]
        if lines:
            _add(lines[-1])
            _add(lines[0])
        return candidates

    @staticmethod
    def _match_suffix_length(haystack: str, needle: str, location: int) -> int:
        if not needle:
            return 0
        available = len(haystack) - location
        if available <= 0:
            return 0
        for offset in range(len(needle)):
            suffix = needle[offset:]
            if not suffix:
                continue
            if len(suffix) > available:
                continue
            if haystack.startswith(suffix, location):
                return len(suffix)
        matched = 0
        limit = min(len(needle), available)
        while matched < limit and haystack[location + matched] == needle[matched]:
            matched += 1
        return matched

    @staticmethod
    def _normalize_patch_text(value: str) -> str:
        if not value:
            return ""
        return value.replace("\r\n", "\n").replace("\r", "\n")

    def _persist_pretty_patch_text(self, iteration_dir: Path, diff_sections: Sequence[str]) -> None:
        if not diff_sections:
            return
        lines: List[str] = []
        for index, chunk in enumerate(diff_sections, start=1):
            cleaned = chunk.strip()
            if not cleaned:
                continue
            lines.append(f"===== PATCH {index} =====")
            lines.append(cleaned)
            lines.append("")
        if not lines:
            return
        content = "\n".join(lines).rstrip() + "\n"
        (iteration_dir / "patch.txt").write_text(content, encoding="utf-8")

    @staticmethod
    def _extract_response_content(response: Any) -> str:
        if isinstance(response, dict):
            content = response.get("content")
            if isinstance(content, str):
                return content
            message = response.get("message")
            if isinstance(message, dict):
                message_content = message.get("content")
                if isinstance(message_content, str):
                    return message_content
            return ""
        if response is None:
            return ""
        return str(response)

    def _parse_patch_text(
        self,
        raw_text: str,
        plan_source: str,
        *,
        allowed_filenames: Optional[Set[str]] = None,
        line_windows: Optional[List[Tuple[int, int]]] = None,
        skip_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[List[_ContextPatch], Optional[str]]:
        text = (raw_text or "").strip()
        if not text:
            raise ValueError("LLM response did not contain patch text.")
        body, notes = self._split_notes(text)
        diff_sections = self._split_unified_diff_sections(body)
        if not diff_sections:
            raise ValueError("LLM response did not include any unified diff sections.")
        normalized_plan = self._normalize_patch_text(plan_source)
        patches: List[_ContextPatch] = []
        for chunk in diff_sections:
            snippets = self._diff_chunk_to_snippets(chunk)
            for snippet in snippets:
                if allowed_filenames:
                    base_name = Path(snippet.file_path).name
                    if base_name not in allowed_filenames:
                        if skip_callback:
                            skip_callback(
                                f"Skipped diff hunk for '{snippet.file_path}' (not in allowed files: {sorted(allowed_filenames)})."
                            )
                        continue
                if line_windows:
                    snippet_range = self._snippet_line_range(snippet)
                    if snippet_range and not self._range_overlaps(snippet_range, line_windows):
                        if skip_callback:
                            skip_callback(
                                f"Skipped diff hunk for '{snippet.file_path}' lines {snippet_range[0]}-{snippet_range[1]} (outside diagnostics)."
                            )
                        continue
                try:
                    patches.append(self._contextualize_patch_block(snippet.content, normalized_plan))
                except ValueError as exc:
                    if skip_callback:
                        skip_callback(
                            f"Skipped diff hunk for '{snippet.file_path}' because contextualization failed: {exc}."
                        )
        if not patches:
            raise ValueError("LLM diff did not produce any usable planner patches.")
        return patches, notes

    def _split_notes(self, text: str) -> Tuple[str, Optional[str]]:
        for marker, closing in _NOTES_MARKERS:
            pattern = rf"(?m)^\s*{re.escape(marker)}"
            match = re.search(pattern, text)
            if not match:
                continue
            marker_start = match.start()
            content_start = match.end()
            if closing:
                closing_pattern = re.escape(closing)
                closing_match = re.search(closing_pattern, text[content_start:])
                if closing_match:
                    closing_start = content_start + closing_match.start()
                    closing_end = closing_start + len(closing)
                    note_text = text[content_start:closing_start]
                    remainder = text[:marker_start] + text[closing_end:]
                else:
                    note_text = text[content_start:]
                    remainder = text[:marker_start]
            else:
                note_text = text[content_start:]
                remainder = text[:marker_start]
            return remainder.rstrip(), note_text.strip() or None
        return text, None

    def _split_unified_diff_sections(self, text: str) -> List[str]:
        normalized = self._normalize_patch_text(text.strip())
        normalized = self._strip_code_fence_wrapper(normalized)
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

    @staticmethod
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
        closing_index = None
        for idx in range(len(lines) - 1, 0, -1):
            if lines[idx].strip() == "```":
                closing_index = idx
                break
        body_lines = lines[1:closing_index] if closing_index is not None else lines[1:]
        return "\n".join(body_lines).strip()

    @staticmethod
    def _is_diagnostic_annotation(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if any(stripped.startswith(marker) for marker in _DIAGNOSTIC_PREFIXES):
            return True
        return any(pattern.search(stripped) for pattern in _DIAGNOSTIC_PATTERNS)

    @staticmethod
    def _normalize_diff_path(path: str) -> str:
        value = (path or "").strip()
        if value.startswith("a/") or value.startswith("b/"):
            value = value[2:]
        if value.startswith("./"):
            value = value[2:]
        return value

    def _diff_chunk_to_snippets(self, chunk: str) -> List[_DiffSnippet]:
        text = self._normalize_patch_text(chunk.strip())
        if not text:
            raise ValueError("Patch chunk was empty.")
        lines = [line.rstrip("\n") for line in text.splitlines() if line is not None]
        if not lines:
            raise ValueError("Patch chunk was empty.")
        index = 0
        # Expect --- filename
        while index < len(lines) and not lines[index].strip():
            index += 1
        if index == len(lines) or not lines[index].startswith("--- "):
            raise ValueError("Unified diff chunk must start with a '--- <old filename>' line.")
        old_path = self._normalize_diff_path(lines[index][4:])
        index += 1
        while index < len(lines) and not lines[index].strip():
            index += 1
        if index == len(lines) or not lines[index].startswith("+++ "):
            raise ValueError("Unified diff chunk must include a '+++ <new filename>' line after the old filename.")
        new_path = self._normalize_diff_path(lines[index][4:])
        index += 1
        file_path = new_path if new_path and new_path != "/dev/null" else old_path
        snippets: List[_DiffSnippet] = []
        snippet_lines: List[str] = []
        saw_hunk = False
        current_old_start: Optional[int] = None
        current_old_count: Optional[int] = None
        current_new_start: Optional[int] = None
        current_new_count: Optional[int] = None
        for raw_line in lines[index:]:
            if not raw_line:
                continue
            if raw_line.startswith("@@"):
                if snippet_lines:
                    snippet = "\n".join(snippet_lines).rstrip("\n") + "\n"
                    if snippet.strip():
                        snippets.append(
                            _DiffSnippet(
                                file_path=file_path,
                                content=snippet,
                                old_start=current_old_start,
                                old_count=current_old_count,
                                new_start=current_new_start,
                                new_count=current_new_count,
                            )
                        )
                    snippet_lines = []
                saw_hunk = True
                (
                    current_old_start,
                    current_old_count,
                    current_new_start,
                    current_new_count,
                ) = self._parse_hunk_header(raw_line)
                continue
            if not saw_hunk:
                raise ValueError("Unified diff chunk missing @@ header before diff body.")
            prefix = raw_line[0]
            if prefix not in {" ", "-", "+"}:
                raise ValueError(f"Unexpected diff line prefix: {prefix!r}")
            payload = raw_line[1:]
            if prefix == " ":
                if self._is_diagnostic_annotation(payload):
                    continue
                snippet_lines.append(payload)
            elif prefix == "+":
                snippet_lines.append(payload)
            # '-' lines are omitted to represent deletions
        if snippet_lines:
            snippet = "\n".join(snippet_lines).rstrip("\n") + "\n"
            if snippet.strip():
                snippets.append(
                    _DiffSnippet(
                        file_path=file_path,
                        content=snippet,
                        old_start=current_old_start,
                        old_count=current_old_count,
                        new_start=current_new_start,
                        new_count=current_new_count,
                    )
                )
        if not saw_hunk:
            raise ValueError("Unified diff chunk missing @@ header before diff body.")
        if not snippets:
            raise ValueError("Unified diff chunk did not produce any replacement snippets.")
        return snippets

    @staticmethod
    def _parse_hunk_header(line: str) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        match = _HUNK_HEADER_PATTERN.match(line.strip())
        if not match:
            raise ValueError("Invalid unified diff hunk header.")

        def _to_int(value: Optional[str]) -> int:
            return int(value) if value is not None else 1

        old_start = int(match.group("old_start")) if match.group("old_start") else None
        old_count = _to_int(match.group("old_count")) if match.group("old_start") else None
        new_start = int(match.group("new_start")) if match.group("new_start") else None
        new_count = _to_int(match.group("new_count")) if match.group("new_start") else None
        return old_start, old_count, new_start, new_count

    def _contextualize_patch_block(self, block: str, normalized_plan: str) -> _ContextPatch:
        normalized_block = self._normalize_patch_text(block.strip("\n"))
        if not normalized_block:
            raise ValueError("Encountered an empty patch block.")
        normalized_block = normalized_block + "\n"
        before_snippet, before_context = self._derive_existing_context(
            normalized_block,
            normalized_plan,
            from_start=True,
        )
        after_snippet, after_context = self._derive_existing_context(
            normalized_block,
            normalized_plan,
            from_start=False,
        )
        replacement_code = self._extract_replacement_segment(
            normalized_block,
            before_snippet,
            after_snippet,
        )
        return _ContextPatch(
            before_context=before_context,
            current_code="",
            after_context=after_context,
            replacement_code=replacement_code,
        )

    def _extract_replacement_segment(
        self,
        full_block: str,
        before_snippet: str,
        after_snippet: str,
    ) -> str:
        segment = full_block
        if before_snippet and segment.startswith(before_snippet):
            segment = segment[len(before_snippet) :]
        if after_snippet and segment.endswith(after_snippet):
            segment = segment[: -len(after_snippet)]
        return segment

    def _derive_existing_context(
        self,
        block: str,
        normalized_plan: str,
        *,
        from_start: bool,
    ) -> Tuple[str, str]:
        lines = block.splitlines(keepends=True)
        if not lines:
            return "", ""
        available_lines = len(lines) - 1
        max_lines = min(_DEFAULT_CONTEXT_LINES, available_lines)
        if max_lines <= 0:
            return "", ""

        def _segment_iter_from_start():
            limit = len(lines) - 1
            for start in range(0, limit):
                remaining = limit - start
                window = min(max_lines, remaining)
                for length in range(window, 0, -1):
                    end = start + length
                    yield lines[start:end]

        def _segment_iter_from_end():
            for end in range(len(lines), 1, -1):
                available = end - 1
                if available <= 0:
                    break
                window = min(max_lines, available)
                for length in range(window, 0, -1):
                    start = end - length
                    if start <= 0:
                        continue
                    yield lines[start:end]

        iterator = _segment_iter_from_start() if from_start else _segment_iter_from_end()
        for segment_lines in iterator:
            candidate = "".join(segment_lines)
            match = self._match_existing_segment(normalized_plan, candidate)
            if match:
                return candidate, match
        return "", ""

    @staticmethod
    def _match_existing_segment(haystack: str, candidate: str) -> Optional[str]:
        if not candidate:
            return ""
        if candidate in haystack:
            return candidate
        if candidate.endswith("\n"):
            trimmed = candidate.rstrip("\n")
            if trimmed and trimmed in haystack:
                return trimmed
        return None

    def _get_plan_logger(self):
        import logging

        return logging.getLogger(self._logger)

    @staticmethod
    def _get_llm_logger():
        import logging

        return logging.getLogger(LLM_LOGGER_NAME)

    @property
    def artifact_root(self) -> Path:
        """Return the root directory used for plan artifacts."""

        return self._artifact_root


__all__ = ["JavaPlanFixer", "PlanFixerRequest", "PlanFixerResult"]
