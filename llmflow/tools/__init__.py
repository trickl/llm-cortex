"""Tool registration package with lazy module loading.

Importing :mod:`llmflow.tools` no longer triggers imports for every tool module.
Call :func:`load_all_tools` (or :func:`load_tool_module`) when you actually need
the decorators in each module to run and register their schemas. This keeps
optional dependencies truly optional for workflows that only touch a subset of
tools (for example, unit tests targeting a specific helper).
"""

from __future__ import annotations

import ast
import importlib
import warnings
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

_TOOL_MODULES: Iterable[str] = (
	"tool_file_system",
	"tool_search_duckduckgo",
	"tool_control_tools",
	"tool_web_parser",
	"tool_datetime",
	"tool_system_monitoring",
	"tool_shell",
	"tool_git",
	"tool_file_manager",
	"tool_file_editing",
	"tool_qlty",
	"tool_cloud",
	"tool_messenger",
	"tool_calendar",
	"tool_email",
	"tool_embenddings",
	"tool_text_to_speech",
	"tool_speech",
	"tool_image_processing",
	"tool_data_analysis",
	"tool_sql_database",
	"tool_mathematical",
	"tool_code_execution",
	"tool_file_operations",
	"tool_decorator",
	"tool_youtube_downloader",
	"tool_video_audio_transcriber",
	"tool_video_processor",
	"tool_frame_extractor",
	"tool_video_preview_generator",
	"tool_scientific_papers_search_arxiv",
	"tool_download_pdf_from_url",
	"tool_text_search",
	"tool_subgoal",
)

_LOADED_MODULES: Set[str] = set()
_FAILED_MODULES: Dict[str, str] = {}
_MODULE_TAGS: Dict[str, Set[str]] = {}
_MODULE_TOOLS: Dict[str, Set[str]] = {}
_TOOL_TO_MODULE: Dict[str, str] = {}
_TOOLS_DIR = Path(__file__).resolve().parent


def load_tool_module(module_name: str, *, warn: bool = False) -> bool:
	"""Import a single tool module if it has not been loaded yet.

	Args:
		module_name: Name from :data:`_TOOL_MODULES` to import.
		warn: Whether to emit a warning when the module (or its optional
			dependency) fails to import. Defaults to ``False`` to keep noise down in
			unit tests; callers can flip it on for interactive diagnostics.

	Returns:
		True if the module imported successfully or was already loaded, False
		otherwise.
	"""

	if module_name in _LOADED_MODULES:
		return True
	if module_name in _FAILED_MODULES:
		if warn:
			warnings.warn(
				f"Tool module '{module_name}' is unavailable: {_FAILED_MODULES[module_name]}",
				RuntimeWarning,
				stacklevel=2,
			)
		return False

	try:
		importlib.import_module(f"{__name__}.{module_name}")
	except ModuleNotFoundError as exc:  # pragma: no cover - best effort
		_FAILED_MODULES[module_name] = str(exc)
		if warn:
			warnings.warn(
				f"Could not import optional tool module '{module_name}': {exc}",
				RuntimeWarning,
				stacklevel=2,
			)
		return False
	except Exception as exc:  # pragma: no cover - best effort
		_FAILED_MODULES[module_name] = str(exc)
		if warn:
			warnings.warn(
				f"Failed to import tool module '{module_name}': {exc}",
				RuntimeWarning,
				stacklevel=2,
			)
		return False

	_LOADED_MODULES.add(module_name)
	return True


def load_all_tools(*, warn: bool = False) -> None:
	"""Import every known tool module exactly once.

	This is useful when you want the registry to contain the complete set of
	tools (e.g., before exposing schemas to an LLM). Warnings are suppressed by
	default but can be enabled via ``warn=True`` when debugging missing optional
	dependencies.
	"""

	for module_name in _TOOL_MODULES:
		load_tool_module(module_name, warn=warn)


def get_failed_tool_modules() -> Dict[str, str]:
	"""Return a mapping of tool module names to their failure reasons."""

	return dict(_FAILED_MODULES)


def _is_register_tool_reference(node: ast.AST) -> bool:
	if isinstance(node, ast.Name):
		return node.id == "register_tool"
	if isinstance(node, ast.Attribute):
		return node.attr == "register_tool"
	return False


def _extract_literal_strings(node: ast.AST) -> Set[str]:
	if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
		values: Set[str] = set()
		for element in node.elts:
			if isinstance(element, ast.Constant) and isinstance(element.value, str):
				values.add(element.value)
		return values
	return set()


def _extract_register_tool_tags(decorator: ast.AST) -> Optional[Set[str]]:
	if isinstance(decorator, ast.Call):
		if not _is_register_tool_reference(decorator.func):
			return None
		tags: Set[str] = set()
		for keyword in decorator.keywords:
			if keyword.arg == "tags":
				tags = _extract_literal_strings(keyword.value)
				break
		return tags
	if _is_register_tool_reference(decorator):
		return set()
	return None


def _ensure_module_metadata(module_name: str) -> None:
	if module_name in _MODULE_TAGS:
		return
	module_path = _TOOLS_DIR / f"{module_name}.py"
	module_tags: Set[str] = set()
	tool_names: Set[str] = set()
	try:
		source = module_path.read_text(encoding="utf-8")
		tree = ast.parse(source, filename=str(module_path))
	except (OSError, SyntaxError):  # pragma: no cover - best effort
		_MODULE_TAGS[module_name] = module_tags
		_MODULE_TOOLS[module_name] = tool_names
		return

	for node in ast.walk(tree):
		if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
			for decorator in node.decorator_list:
				tags = _extract_register_tool_tags(decorator)
				if tags is not None:
					tool_names.add(node.name)
					module_tags.update(tags)
					break

	_MODULE_TAGS[module_name] = module_tags
	_MODULE_TOOLS[module_name] = tool_names
	for tool in tool_names:
		_TOOL_TO_MODULE.setdefault(tool, module_name)


def get_modules_for_tags(tags: Iterable[str], match_all: bool = True) -> Set[str]:
	requested = {tag for tag in tags if tag}
	if not requested:
		return set(_TOOL_MODULES)
	matching: Set[str] = set()
	for module_name in _TOOL_MODULES:
		_ensure_module_metadata(module_name)
		module_tags = _MODULE_TAGS.get(module_name, set())
		if not module_tags:
			continue
		if match_all:
			if requested.issubset(module_tags):
				matching.add(module_name)
		else:
			if module_tags.intersection(requested):
				matching.add(module_name)
	return matching


def get_module_for_tool_name(tool_name: str) -> Optional[str]:
	if not tool_name:
		return None
	if tool_name in _TOOL_TO_MODULE:
		return _TOOL_TO_MODULE[tool_name]
	for module_name in _TOOL_MODULES:
		_ensure_module_metadata(module_name)
		if tool_name in _MODULE_TOOLS.get(module_name, set()):
			return module_name
	return None


__all__ = [
	"load_all_tools",
	"load_tool_module",
	"get_failed_tool_modules",
	"get_modules_for_tags",
	"get_module_for_tool_name",
]