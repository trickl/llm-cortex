"""Tests for Java source normalization helpers."""
from __future__ import annotations

from llmflow.planning.java_planner import _normalize_java_source


def test_normalize_handles_markdown_headings_and_unclosed_fence() -> None:
    raw = (
        "# Java Planning Specification\n"
        "# Java Planning Specification\n\n"
        "```java\n"
        "public class Plan {\n"
        "    public void main() {}\n"
        "}\n"
        "``\n"
    )

    normalized = _normalize_java_source(raw)

    assert normalized.startswith("public class Plan")
    assert normalized.endswith("}")


def test_normalize_prefers_java_code_block_when_multiple_exist() -> None:
    raw = (
        "```python\nprint('noop')\n```\n"
        "```java\npublic class Planner {\n    public void main() {}\n}\n```\n"
    )

    normalized = _normalize_java_source(raw)

    assert "class Planner" in normalized
    assert "print" not in normalized


def test_normalize_trims_markdown_prefix_without_code_fence() -> None:
    raw = (
        "## Summary\n\n"
        "Steps to follow:\n"
        "1. Do the thing.\n"
        "2. Ship it.\n\n"
        "public class SoloPlan {\n"
        "    public void main() {\n"
        "        System.out.println(\"hi\");\n"
        "    }\n"
        "}\n"
    )

    normalized = _normalize_java_source(raw)

    assert normalized.startswith("public class SoloPlan")
    assert "Steps to follow" not in normalized


def test_normalize_decodes_escaped_newlines() -> None:
    raw = (
        "public class Planner {\\n"
        "    public static void main(String[] args) throws Exception {\\n"
        "        System.out.println(\"hi\");\\n"
        "    }\\n"
        "}"
    )

    normalized = _normalize_java_source(raw)

    assert normalized.startswith("public class Planner")
    assert normalized.splitlines()[0] == "public class Planner {"
    assert len(normalized.splitlines()) > 1
