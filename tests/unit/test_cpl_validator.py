import pytest

from llmflow.planning.runtime.parser import parse_java_plan
from llmflow.planning.runtime.validator import PlanValidator, ValidationError


def _validate(source: str, syscalls: set[str]):
    plan = parse_java_plan(source)
    validator = PlanValidator(available_syscalls=syscalls)
    validator.validate(plan)


def test_validator_requires_main():
    source = """
    public class Plan {
        public void helper() {
            return;
        }
    }
    """
    with pytest.raises(ValidationError) as exc_info:
        _validate(source, set())

    assert "main" in str(exc_info.value)


def test_validator_detects_missing_syscall():
    source = """
    public class Plan {
        public void main() {
            syscall.log("hi");
            return;
        }
    }
    """
    with pytest.raises(ValidationError) as exc_info:
        _validate(source, set())

    assert "Syscall 'log' not registered" in str(exc_info.value)


def test_validator_flags_undefined_variable():
    source = """
    public class Plan {
        public void main() {
            syscall.log(msg);
            return;
        }
    }
    """
    with pytest.raises(ValidationError) as exc_info:
        _validate(source, {"log"})

    assert "Variable 'msg' not defined" in str(exc_info.value)


def test_validator_enforces_statement_limit():
    statements = "\n".join(["            syscall.log(\"x\");" for _ in range(8)])
    source = f"""
    public class Plan {{
        public void main() {{
{statements}
            return;
        }}
    }}
    """

    with pytest.raises(ValidationError) as exc_info:
        _validate(source, {"log"})

    assert "exceeds" in str(exc_info.value)


def test_validator_requires_non_void_return_value():
    source = """
    public class Plan {
        public void main() {
            return;
        }

        public String compute() {
            syscall.log("missing return");
        }
    }
    """
    with pytest.raises(ValidationError) as exc_info:
        _validate(source, {"log"})

    assert "must return a value" in str(exc_info.value)
