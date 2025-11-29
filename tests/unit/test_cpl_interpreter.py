from typing import Callable, Dict, List

import pytest

from llmflow.planning.runtime.interpreter import (
    ExecutionTracer,
    PlanInterpreter,
    PlanRuntimeError,
)
from llmflow.planning.runtime.parser import parse_java_plan
from llmflow.planning.runtime.validator import ValidationError
from llmflow.runtime.errors import ToolError
from llmflow.runtime.syscall_registry import SyscallRegistry


def _build_interpreter(source: str, syscalls: Dict[str, Callable[..., object]], tracer=None) -> PlanInterpreter:
    plan = parse_java_plan(source)
    registry = SyscallRegistry.from_mapping(syscalls)
    return PlanInterpreter(plan, registry=registry, tracer=tracer)


def test_interpreter_executes_plan():
    messages: List[str] = []

    def log_syscall(msg: str):
        messages.append(msg)

    tracer = ExecutionTracer(enabled=True)
    interpreter = _build_interpreter(
        """
        public class Plan {
            public void main() {
                logItems("one", "two");
                return;
            }

            public void logItems(String first, String second) {
                syscall.log(first);
                syscall.log(second);
                return;
            }
        }
        """,
        {"log": log_syscall},
        tracer=tracer,
    )

    interpreter.run()

    assert messages == ["one", "two"]
    assert any(event["type"] == "syscall_start" for event in tracer.as_list())


def test_missing_syscall_reports_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        _build_interpreter(
            """
            public class Plan {
                public void main() {
                    syscall.unknown();
                    return;
                }
            }
            """,
            {},
        )

    assert "Syscall 'unknown' not registered" in str(exc_info.value)


def test_tool_error_caught_by_try_catch():
    events: List[str] = []

    def flaky_syscall():
        raise ToolError("boom")

    def log_syscall(msg: str):
        events.append(msg)

    interpreter = _build_interpreter(
        """
        public class Plan {
            public void main() {
                try {
                    syscall.flaky();
                } catch (ToolError err) {
                    syscall.log("caught");
                }
                return;
            }
        }
        """,
        {"flaky": flaky_syscall, "log": log_syscall},
    )

    interpreter.run()

    assert events == ["caught"]


def test_runtime_error_when_deferred_returns_non_list():
    plan = parse_java_plan(
        """
        public class Plan {
            public void main() {
                List<String> values = loadValues();
                for (String value : values) {
                    syscall.log(value);
                }
                return;
            }

            @Deferred
            public List<String> loadValues();
        }
        """,
    )

    registry = SyscallRegistry.from_mapping({"log": lambda msg: None})

    def planner(_prompt):
        return "{ return \"oops\"; }"

    interpreter = PlanInterpreter(
        plan,
        registry=registry,
        deferred_planner=planner,
        spec_text="SPEC",
    )

    with pytest.raises(PlanRuntimeError, match="for-loop iterable must be a list"):
        interpreter.run()
