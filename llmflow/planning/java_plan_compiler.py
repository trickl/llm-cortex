"""Invoke ``javac`` to validate synthesized Java plans before execution."""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence
from contextlib import contextmanager


class JavaCompilationError(RuntimeError):
    """Raised when the Java compiler cannot be invoked."""


@dataclass
class CompilationError:
    """Structured representation of a ``javac`` diagnostic."""

    message: str
    file: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None
    raw: Optional[str] = None


@dataclass
class JavaCompilationResult:
    """Outcome returned by :class:`JavaPlanCompiler`."""

    success: bool
    command: Sequence[str]
    stdout: str = ""
    stderr: str = ""
    errors: List[CompilationError] = field(default_factory=list)


class JavaPlanCompiler:
    """Simple wrapper around ``javac`` for validating generated plans."""

    _CLASS_PATTERN = re.compile(r"(?:class|interface)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)")
    _PACKAGE_PATTERN = re.compile(r"^\s*package\s+([A-Za-z_][\w\.]*?)\s*;", re.MULTILINE)
    _ERROR_PATTERN = re.compile(
        r"^(?P<file>[^:]+):(?P<line>\d+)(?::(?P<column>\d+))?: error: (?P<message>.+)$"
    )

    def __init__(self, *, javac_path: str = "javac") -> None:
        self._javac_path = javac_path

    def compile(
        self,
        plan_source: str,
        *,
        tool_stub_source: Optional[str] = None,
        tool_stub_class_name: Optional[str] = None,
        working_dir: Optional[Path] = None,
    ) -> JavaCompilationResult:
        """Attempt to compile ``plan_source`` along with runtime stubs."""

        self._ensure_javac_available()
        class_name = self._detect_class_name(plan_source)
        package_name = self._extract_package(plan_source)
        with self._select_directory(working_dir) as temp_dir:
            source_paths: List[Path] = []
            source_paths.append(
                self._write_java_source(temp_dir, class_name, plan_source, package_name)
            )
            source_paths.append(
                self._write_java_source(
                    temp_dir,
                    "ToolResult",
                    self._build_tool_result_stub(package_name),
                    package_name,
                )
            )
            source_paths.append(
                self._write_java_source(
                    temp_dir,
                    "ToolError",
                    self._build_tool_error_stub(package_name),
                    package_name,
                )
            )

            if tool_stub_source and tool_stub_class_name:
                stub_package = self._extract_package(tool_stub_source)
                source_paths.append(
                    self._write_java_source(
                        temp_dir,
                        tool_stub_class_name,
                        tool_stub_source,
                        stub_package,
                    )
                )

            command = [self._javac_path]
            command.extend(str(path) for path in source_paths)
            process = subprocess.run(  # noqa: PLW1510 - intentional capture
                command,
                cwd=temp_dir,
                check=False,
                capture_output=True,
                text=True,
            )
            result = JavaCompilationResult(
                success=process.returncode == 0,
                command=tuple(command),
                stdout=process.stdout,
                stderr=process.stderr,
            )
            if not result.success:
                result.errors = self._parse_errors(process.stderr)
            return result

    # ------------------------------------------------------------------
    # Source helpers

    def _detect_class_name(self, source: str) -> str:
        match = self._CLASS_PATTERN.search(source)
        if not match:
            raise JavaCompilationError("Java source must declare at least one class or interface.")
        return match.group("name")

    def _extract_package(self, source: str) -> Optional[str]:
        match = self._PACKAGE_PATTERN.search(source)
        if match:
            return match.group(1)
        return None

    def _write_java_source(
        self,
        base_dir: Path,
        class_name: str,
        source: str,
        package_name: Optional[str] = None,
    ) -> Path:
        relative_dir = Path()
        if package_name:
            relative_dir = Path(*package_name.split("."))
        target_dir = base_dir / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{class_name}.java"
        content = source.rstrip() + "\n"
        file_path.write_text(content, encoding="utf-8")
        return file_path.relative_to(base_dir)

    # ------------------------------------------------------------------
    # Stub generation

    @staticmethod
    def _build_tool_result_stub(package_name: Optional[str]) -> str:
        lines: List[str] = []
        if package_name:
            lines.append(f"package {package_name};")
            lines.append("")
        lines.append("public final class ToolResult {")
        lines.append("    private ToolResult() {}")
        lines.append("")
        lines.append("    public static ToolResult empty() {")
        lines.append("        return new ToolResult();")
        lines.append("    }")
        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def _build_tool_error_stub(package_name: Optional[str]) -> str:
        lines: List[str] = []
        if package_name:
            lines.append(f"package {package_name};")
            lines.append("")
        lines.append("public class ToolError extends RuntimeException {")
        lines.append("    public ToolError(String message) {")
        lines.append("        super(message);")
        lines.append("    }")
        lines.append("")
        lines.append("    public ToolError(String message, Throwable cause) {")
        lines.append("        super(message, cause);")
        lines.append("    }")
        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Error parsing & environment helpers

    def _parse_errors(self, stderr: str) -> List[CompilationError]:
        errors: List[CompilationError] = []
        lines = stderr.splitlines()
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            match = self._ERROR_PATTERN.match(line)
            if match:
                raw_lines = [line]
                idx += 1
                while idx < len(lines) and (lines[idx].startswith(" ") or lines[idx].startswith("\t")):
                    raw_lines.append(lines[idx])
                    idx += 1
                errors.append(
                    CompilationError(
                        message=match.group("message").strip(),
                        file=match.group("file"),
                        line=int(match.group("line")),
                        column=int(match.group("column")) if match.group("column") else None,
                        raw="\n".join(raw_lines).rstrip(),
                    )
                )
                continue
            idx += 1
        if not errors and stderr.strip():
            errors.append(CompilationError(message=stderr.strip(), raw=stderr.strip()))
        return errors

    def _ensure_javac_available(self) -> None:
        if shutil.which(self._javac_path):
            return
        raise JavaCompilationError(
            f"Java compiler '{self._javac_path}' is not available on PATH. Install a JDK to enable plan compilation."
        )

    @contextmanager
    def _select_directory(self, working_dir: Optional[Path]):
        if working_dir is None:
            with tempfile.TemporaryDirectory(prefix="llmflow_plan_compile_") as tmpdir:
                yield Path(tmpdir)
        else:
            path = Path(working_dir)
            path.mkdir(parents=True, exist_ok=True)
            yield path


__all__ = [
    "CompilationError",
    "JavaCompilationError",
    "JavaCompilationResult",
    "JavaPlanCompiler",
]
