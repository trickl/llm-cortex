"""Centralized logging helpers for Agent Cortex runs."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

LLM_LOGGER_NAME = "agentcortex.llm"
PLAN_LOGGER_NAME = "agentcortex.plan"
TOOLS_LOGGER_NAME = "agentcortex.tools"
MANIFEST_LOGGER_NAME = "agentcortex.manifest"

_LOG_FILE_SPEC = (
    (LLM_LOGGER_NAME, "llm.log", "llm"),
    (PLAN_LOGGER_NAME, "plans.log", "plans"),
    (TOOLS_LOGGER_NAME, "tools.log", "tools"),
    (MANIFEST_LOGGER_NAME, "manifest.log", "manifests"),
)


@dataclass
class RunLogContext:
    """Holds per-run logging metadata and output paths."""

    run_id: str
    logs_root: Path
    run_dir: Path
    llm_log: Path
    plan_log: Path
    tool_log: Path
    manifest_log: Path


@dataclass
class RunArtifactManager:
    """Tracks additional artifacts created during a run."""

    context: RunLogContext
    mermaid_path: Optional[Path] = None
    extra_files: List[Path] = field(default_factory=list)

    def register_mermaid(self, path: Path) -> None:
        self.mermaid_path = Path(path).resolve()

    def register_artifact(self, path: Path) -> None:
        self.extra_files.append(Path(path).resolve())

    def iter_entries(self) -> List[Tuple[str, Path]]:
        entries: List[Tuple[str, Path]] = [
            ("LLM Log", self.context.llm_log),
            ("Plan Log", self.context.plan_log),
            ("Tool Log", self.context.tool_log),
        ]
        if self.mermaid_path:
            entries.append(("Mermaid Diagram", self.mermaid_path))
        for idx, path in enumerate(self.extra_files, start=1):
            entries.append((f"Artifact {idx}", path))
        entries.append(("Run Manifest", self.context.manifest_log))
        return entries

    def write_manifest(self, success: bool) -> Path:
        entries = self.iter_entries()
        lines = [
            f"Run ID: {self.context.run_id}",
            f"Status: {'success' if success else 'failed'}",
            "",
            "Artifacts:",
        ]
        manifest_payload: Dict[str, str] = {}
        for label, path in entries:
            path_str = str(path)
            lines.append(f"- {label}: {path_str}")
            manifest_payload[label] = path_str
        manifest_text = "\n".join(lines)
        self.context.manifest_log.write_text(manifest_text, encoding="utf-8")

        manifest_logger = logging.getLogger(MANIFEST_LOGGER_NAME)
        manifest_logger.info(
            json.dumps(
                {
                    "run_id": self.context.run_id,
                    "success": success,
                    "artifacts": manifest_payload,
                },
                ensure_ascii=False,
            )
        )
        return self.context.manifest_log


def setup_run_logging(
    *,
    run_id: Optional[str] = None,
    logs_root: Optional[Path] = None,
    run_directory: Optional[Path] = None,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
) -> RunLogContext:
    """Configure per-run file loggers for LLM, plans, tools, and manifests."""

    resolved_root = Path(logs_root or "logs").resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)

    if run_directory is not None:
        run_dir = Path(run_directory).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        if run_id is None:
            raise ValueError("run_id must be provided when reusing an existing run_directory")
        active_run_id = run_id
    else:
        active_run_id = run_id or uuid4().hex
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir_name = f"{timestamp}-{active_run_id[:6]}"
        run_dir = resolved_root / run_dir_name
        run_dir.mkdir(parents=True, exist_ok=True)

    log_paths = {
        key: run_dir / filename
        for _, filename, key in _LOG_FILE_SPEC
    }

    for logger_name, _, key in _LOG_FILE_SPEC:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers = [
            handler
            for handler in logger.handlers
            if not isinstance(handler, RotatingFileHandler)
        ]
        file_handler = RotatingFileHandler(
            log_paths[key], maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(file_handler)

    return RunLogContext(
        run_id=active_run_id,
        logs_root=resolved_root,
        run_dir=run_dir,
        llm_log=log_paths["llm"],
        plan_log=log_paths["plans"],
        tool_log=log_paths["tools"],
        manifest_log=log_paths["manifests"],
    )


def summarize_messages(messages: Iterable[Dict[str, object]], limit: int = 120) -> str:
    """Return a concise single-line summary of the final user message."""

    text: Optional[str] = None
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            candidate = str(msg.get("content", "")).strip()
            if candidate:
                text = candidate
    if not text:
        return "(no user content)"
    text = text.replace("\n", " ")
    return text[:limit] + ("…" if len(text) > limit else "")


def summarize_response(response: Dict[str, object], limit: int = 120) -> str:
    """Single-line summary of the LLM response."""

    content = str(response.get("content", "")).strip()
    if not content:
        return "(empty response)"
    content = content.replace("\n", " ")
    return content[:limit] + ("…" if len(content) > limit else "")


def log_plan_failure_summary(
    *,
    attempt_number: int,
    plan_id: Optional[str],
    reason: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a concise plan failure summary to the plan logger."""

    logger = logging.getLogger(PLAN_LOGGER_NAME)
    payload = {
        "attempt": attempt_number,
        "plan_id": plan_id,
        "reason": reason,
        "metadata": metadata or {},
    }
    logger.warning(
        "plan_failure_summary attempt=%s plan_id=%s reason=%s metadata=%s",
        attempt_number,
        plan_id,
        reason,
        metadata or {},
    )


def format_execution_failure_reason(execution_result: Dict[str, Any]) -> str:
    """Return a human-readable reason for an execution failure."""

    if not isinstance(execution_result, dict):
        return "Execution failed for an unknown reason."
    metadata = execution_result.get("metadata") or {}
    stage = metadata.get("stage")
    errors = execution_result.get("errors") or []
    if stage == "compile":
        first = errors[0] if errors else {}
        return f"Compilation failed: {first.get('message') or 'See errors for details.'}"
    if errors:
        first = errors[0]
        error_type = first.get("type") or "execution_error"
        message = first.get("message") or "No error message provided."
        return f"{error_type}: {message}"
    return metadata.get("reason") or "Execution failed without structured errors."
