from __future__ import annotations

from typing import List

import pytest

from llmflow.planning.deferred_planner import DeferredFunctionPrompt
from llmflow.planning.runtime.ast import DeferredExecutionOptions
from llmflow.planning.runtime.interpreter import PlanInterpreter, PlanRuntimeError
from llmflow.planning.runtime.parser import parse_java_plan
from llmflow.runtime.syscall_registry import SyscallRegistry


def _build_interpreter(source: str, planner=None, options=None, log_sink: List[str] | None = None):
    plan = parse_java_plan(source)
    messages: List[str] = log_sink if log_sink is not None else []

    def log_syscall(msg: str):
        messages.append(msg)

    registry = SyscallRegistry.from_mapping({"log": log_syscall})
    interpreter = PlanInterpreter(
        plan,
        registry=registry,
        deferred_planner=planner,
        deferred_options=options or DeferredExecutionOptions(),
        spec_text="SPEC",
    )
    return interpreter, messages


def test_parse_deferred_function_declaration():
    plan = parse_java_plan(
        """
        public class Plan {
            @Deferred
            public void perform();

            public void main() {
                return;
            }
        }
        """,
    )

    perform = plan.functions["perform"]
    assert perform.is_deferred() is True
    assert perform.body is None


def test_parse_deferred_with_stub_body():
    plan = parse_java_plan(
        """
        public class Plan {
            @Deferred
            public void perform() {
                syscall.log("placeholder");
                return;
            }

            public void main() {
                return;
            }
        }
        """,
    )

    perform = plan.functions["perform"]
    assert perform.is_deferred() is True
    assert perform.body is not None
    assert len(perform.body) == 2


def test_deferred_execution_invokes_planner_once():
    planner_calls: List[DeferredFunctionPrompt] = []

    def planner(prompt: DeferredFunctionPrompt) -> str:
        planner_calls.append(prompt)
        return "{ syscall.log(\"from deferred\"); return; }"

    interpreter, messages = _build_interpreter(
        """
        public class Plan {
            public void main() {
                perform();
                return;
            }

            @Deferred
            public void perform();
        }
        """,
        planner=planner,
    )

    interpreter.run()

    assert messages == ["from deferred"]
    assert len(planner_calls) == 1
    assert planner_calls[0].context.function_name == "perform"


def test_deferred_execution_reuses_cached_body():
    planner_calls: List[DeferredFunctionPrompt] = []

    def planner(prompt: DeferredFunctionPrompt) -> str:
        planner_calls.append(prompt)
        return "{ syscall.log(\"cached run\"); return; }"

    interpreter, messages = _build_interpreter(
        """
        public class Plan {
            public void main() {
                perform();
                perform();
                return;
            }

            @Deferred
            public void perform();
        }
        """,
        planner=planner,
    )

    interpreter.run()

    assert messages == ["cached run", "cached run"]
    assert len(planner_calls) == 1


def test_deferred_execution_without_cache_regenerates():
    planner_calls: List[DeferredFunctionPrompt] = []

    def planner(prompt: DeferredFunctionPrompt) -> str:
        planner_calls.append(prompt)
        return "{ syscall.log(\"fresh run\"); return; }"

    options = DeferredExecutionOptions(reuse_cached_bodies=False)
    interpreter, _ = _build_interpreter(
        """
        public class Plan {
            public void main() {
                perform();
                perform();
                return;
            }

            @Deferred
            public void perform();
        }
        """,
        planner=planner,
        options=options,
    )

    interpreter.run()

    assert len(planner_calls) == 2


def test_invalid_synthesis_raises_runtime_error():
    def planner(_prompt: DeferredFunctionPrompt) -> str:
        return "not a block"

    interpreter, _ = _build_interpreter(
        """
        public class Plan {
            public void main() {
                perform();
                return;
            }

            @Deferred
            public void perform();
        }
        """,
        planner=planner,
    )

    with pytest.raises(PlanRuntimeError):
        interpreter.run()


def test_nested_deferred_functions():
    planner_calls: List[str] = []

    def planner(prompt: DeferredFunctionPrompt) -> str:
        planner_calls.append(prompt.context.function_name)
        if prompt.context.function_name == "outer":
            return "{ syscall.log(\"outer\"); inner(\"nested\"); return; }"
        return "{ syscall.log(msg); return; }"

    interpreter, messages = _build_interpreter(
        """
        public class Plan {
            public void main() {
                outer();
                return;
            }

            @Deferred
            public void outer();

            @Deferred
            public void inner(String msg);
        }
        """,
        planner=planner,
    )

    interpreter.run()

    assert messages == ["outer", "nested"]
    assert planner_calls == ["outer", "inner"]


def test_deferred_function_without_planner_errors():
    interpreter, _ = _build_interpreter(
        """
        public class Plan {
            public void main() {
                perform();
                return;
            }

            @Deferred
            public void perform();
        }
        """,
    )

    with pytest.raises(PlanRuntimeError, match="Deferred function 'perform'"):
        interpreter.run()