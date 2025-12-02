#!/usr/bin/env python
"""Standalone probe for Granite via Instructor.

This helper makes it easy to inspect the raw assistant text (and any tool
calls) that Granite returns when invoked through Instructor with different
modes. It can be run against a local Ollama instance to quickly determine
whether the model emits usable plain text outside of TOOLS mode.
"""

from __future__ import annotations

import argparse
import os
import textwrap
from typing import Any, Dict, List, Optional

import instructor
from instructor import Mode
from instructor.core.exceptions import InstructorRetryException
from pydantic import BaseModel, Field

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are helping validate whether Granite can emit Java planning code without
    relying on tool calls. When responding, provide the Java source inside a
    fenced code block and ensure it compiles without external dependencies.
    """
).strip()

DEFAULT_TASK = textwrap.dedent(
    """
    Draft a short Java plan that audits all Markdown files in the repository and
    prints any TODO sections that look stale. Prefer clear helper methods over
    long inline logic.
    """
).strip()

MODE_ALIASES: Dict[str, Mode] = {
    "json": Mode.JSON,
    "md_json": Mode.MD_JSON,
    "tools": Mode.TOOLS,
}


class JavaPlanProbe(BaseModel):
    """Simple schema that captures whatever Java plan text the model emits."""

    plan: str = Field(
        ...,
        description="Full Java source produced by the model",
        min_length=1,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="qwen2.5-coder:7b",
        help="Ollama model to probe (default: qwen2.5-coder:7b)",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_TASK,
        help="User task to submit. Set to \"-\" to read from stdin.",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_ALIASES.keys()),
        default="json",
        help="Instructor mode to use (default: json)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434/v1",
        help="Base URL for the Ollama OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="How many Instructor retries to allow (default: 0).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if Granite adds unexpected fields to the schema.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable INSTRUCTOR_DEBUG logging for verbose traces.",
    )
    parser.add_argument(
        "--show-raw-json",
        action="store_true",
        help="Dump the provider response JSON for deeper inspection.",
    )
    return parser.parse_args()


def normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def read_prompt(prompt_arg: str) -> str:
    if prompt_arg != "-":
        return prompt_arg.strip()
    return textwrap.dedent(os.sys.stdin.read()).strip()


def build_messages(task: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]


def summarize_completion(completion: Any) -> None:
    if completion is None:
        print("\nNo completion payload captured.")
        return

    try:
        choice = completion.choices[0]
        message = choice.message
    except (AttributeError, IndexError, KeyError):
        print("\nUnable to unpack completion; raw object follows:")
        print(completion)
        return

    print("\n=== Assistant Message ===")
    content = getattr(message, "content", None)
    if not content:
        print("<empty content>")
    else:
        # content can be a list (per OpenAI spec) or a plain string.
        if isinstance(content, list):
            for idx, part in enumerate(content, 1):
                print(f"[{idx}] {part}")
        else:
            print(content)

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        print("\n=== Tool Calls ===")
        for idx, call in enumerate(tool_calls, 1):
            name = getattr(call.function, "name", "<unknown>")
            arguments = getattr(call.function, "arguments", "<no args>")
            print(f"{idx}. {name}: {arguments}")
    else:
        print("\n(no tool calls emitted)")


def dump_raw_response(model: BaseModel) -> None:
    raw = getattr(model, "_raw_response", None)
    if raw is None:
        print("\nNo _raw_response attribute found.")
        return
    print("\n=== _raw_response ===")
    try:
        print(raw.model_dump_json(indent=2))
    except AttributeError:
        try:
            import json

            print(json.dumps(raw, indent=2, ensure_ascii=False))
        except Exception:  # pylint: disable=broad-except
            print(raw)


def build_client(provider: str, mode: Mode, base_url: str):
    return instructor.from_provider(
        provider,
        mode=mode,
        base_url=normalize_base_url(base_url),
    )


def main() -> int:
    args = parse_args()
    if args.debug:
        os.environ.setdefault("INSTRUCTOR_DEBUG", "1")

    mode = MODE_ALIASES[args.mode]
    provider_id = f"ollama/{args.model}"
    prompt = read_prompt(args.prompt)
    client = build_client(provider_id, mode, args.base_url)
    messages = build_messages(prompt)

    try:
        probe, completion = client.create_with_completion(
            messages=messages,
            response_model=JavaPlanProbe,
            max_retries=args.max_retries,
            strict=args.strict,
        )
        print("Structured parse succeeded. Plan excerpt:\n")
        preview = probe.plan.strip()
        print(preview if len(preview) < 2000 else preview[:2000] + "\n…")
        summarize_completion(completion)
        if args.show_raw_json:
            dump_raw_response(probe)
        return 0
    except InstructorRetryException as exc:
        print("\nInstructor exhausted retries:")
        print(f"Attempts: {exc.n_attempts}")
        print(f"Mode: {args.mode}")
        summarize_completion(exc.last_completion)
        if args.show_raw_json and exc.last_completion is not None:
            try:
                print("\n=== last_completion JSON ===")
                print(exc.last_completion.model_dump_json(indent=2))
            except AttributeError:
                pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
