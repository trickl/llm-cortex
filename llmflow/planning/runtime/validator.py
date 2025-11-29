"""Static validation for Java-based plans before execution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

from .ast import (
    Assign,
    BinaryOp,
    CallExpr,
    Expr,
    ExprStmt,
    ForStmt,
    FunctionDef,
    IfStmt,
    ListLiteral,
    Literal,
    MapLiteral,
    Plan,
    ReturnStmt,
    Stmt,
    SyscallExpr,
    TryCatchStmt,
    VarDecl,
    VarRef,
)

UNKNOWN_TYPE = "Unknown"
LIST_PREFIX = "List<"
MAP_PREFIX = "Map<String,"


@dataclass
class ValidationIssue:
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    function: Optional[str] = None

    def format(self) -> str:
        parts: List[str] = []
        if self.function:
            parts.append(f"function '{self.function}'")
        if self.line is not None:
            parts.append(f"line {self.line}")
        if self.column is not None:
            parts.append(f"column {self.column}")
        suffix = f" ({', '.join(parts)})" if parts else ""
        return f"{self.message}{suffix}"


class ValidationError(Exception):
    """Raised when plan validation fails."""

    def __init__(self, issues: List[ValidationIssue]):
        self.issues = issues
        message = issues[0].format() if issues else "Plan validation failed"
        if len(issues) > 1:
            message = f"{message} (+{len(issues) - 1} more issues)"
        super().__init__(message)


class _ScopeStack:
    def __init__(self):
        self.stack: List[Dict[str, str]] = []

    def push(self):
        self.stack.append({})

    def pop(self):
        self.stack.pop()

    def define(self, name: str, type_name: str) -> bool:
        current = self.stack[-1]
        if name in current:
            return False
        current[name] = type_name
        return True

    def assign(self, name: str, type_name: Optional[str] = None) -> Optional[str]:
        for scope in reversed(self.stack):
            if name in scope:
                if type_name is not None:
                    scope[name] = type_name
                return scope[name]
        return None

    def lookup(self, name: str) -> Optional[str]:
        return self.assign(name)


class PlanValidator:
    """Performs static validation on parsed plans."""

    MAX_FUNCTION_STATEMENTS = 7

    def __init__(self, available_syscalls: Optional[Iterable[str]] = None):
        self.available_syscalls: Set[str] = set(available_syscalls or [])
        self.issues: List[ValidationIssue] = []
        self.function_map: Dict[str, FunctionDef] = {}
        self.current_function: Optional[FunctionDef] = None
        self.scope = _ScopeStack()
        self.function_returns_value: Dict[str, bool] = {}

    def validate(self, plan: Plan):
        self.issues.clear()
        self.function_map = plan.functions
        self._validate_plan_structure(plan)
        for fn in plan.ordered_functions:
            self._validate_function(fn)
        if self.issues:
            raise ValidationError(self.issues)

    def _validate_plan_structure(self, plan: Plan):
        if not plan.functions:
            self._add_issue("Plan must define at least one function", plan)
        if "main" not in plan.functions:
            self._add_issue("Plan must define a main() function", plan)
        else:
            main_fn = plan.functions["main"]
            if main_fn.return_type != "Void":
                self._add_issue("main() must return Void", main_fn)
            if main_fn.params:
                self._add_issue("main() must not take parameters", main_fn)
        seen: Set[str] = set()
        for fn in plan.ordered_functions:
            if fn.name in seen:
                self._add_issue(f"Duplicate function name '{fn.name}'", fn)
            seen.add(fn.name)

    def _validate_function(self, fn: FunctionDef):
        self.current_function = fn
        self.function_returns_value[fn.name] = False
        if fn.body is None:
            if not fn.is_deferred():
                self._add_issue(
                    f"Function '{fn.name}' must define a body or be annotated with @Deferred",
                    fn,
                )
            return
        body_stmts = fn.body or []
        if len(body_stmts) > self.MAX_FUNCTION_STATEMENTS:
            self._add_issue(
                f"Function '{fn.name}' exceeds {self.MAX_FUNCTION_STATEMENTS} statements",
                fn,
            )
        self.scope.push()
        for param in fn.params:
            if not self.scope.define(param.name, param.type):
                self._add_issue(f"Parameter '{param.name}' defined multiple times", param)
        for stmt in body_stmts:
            self._validate_stmt(stmt)
        self.scope.pop()
        if fn.return_type != "Void" and not self.function_returns_value[fn.name]:
            self._add_issue(f"Function '{fn.name}' must return a value", fn)
        self.current_function = None

    def _validate_stmt(self, stmt: Stmt):
        if isinstance(stmt, VarDecl):
            expr_type = self._infer_expr_type(stmt.expr)
            if not self.scope.define(stmt.name, stmt.type):
                self._add_issue(f"Variable '{stmt.name}' already defined in this scope", stmt)
            else:
                if not self._types_compatible(stmt.type, expr_type):
                    self._add_issue(
                        f"Cannot assign expression of type {expr_type} to {stmt.type}",
                        stmt,
                    )
        elif isinstance(stmt, Assign):
            existing_type = self.scope.lookup(stmt.name)
            if existing_type is None:
                self._add_issue(f"Variable '{stmt.name}' not defined", stmt)
            expr_type = self._infer_expr_type(stmt.expr)
            if existing_type is not None and not self._types_compatible(existing_type, expr_type):
                self._add_issue(
                    f"Cannot assign expression of type {expr_type} to {existing_type}",
                    stmt,
                )
        elif isinstance(stmt, IfStmt):
            cond_type = self._infer_expr_type(stmt.cond)
            if cond_type != "Bool" and cond_type != UNKNOWN_TYPE:
                self._add_issue("if condition must be Bool", stmt)
            self.scope.push()
            for inner in self._ensure_list(stmt.then_body):
                self._validate_stmt(inner)
            self.scope.pop()
            self.scope.push()
            for inner in self._ensure_list(stmt.else_body):
                self._validate_stmt(inner)
            self.scope.pop()
        elif isinstance(stmt, ForStmt):
            iterable_type = self._infer_expr_type(stmt.iterable_expr)
            inner_type = self._list_inner(iterable_type)
            if inner_type is None:
                self._add_issue("for loop iterable must be a List", stmt)
                inner_type = UNKNOWN_TYPE
            self.scope.push()
            self.scope.define(stmt.var_name, inner_type)
            for inner in self._ensure_list(stmt.body):
                self._validate_stmt(inner)
            self.scope.pop()
        elif isinstance(stmt, TryCatchStmt):
            self.scope.push()
            for inner in self._ensure_list(stmt.try_body):
                self._validate_stmt(inner)
            self.scope.pop()
            self.scope.push()
            if not self.scope.define(stmt.error_var, "ToolError"):
                self._add_issue(f"Catch variable '{stmt.error_var}' already defined", stmt)
            for inner in self._ensure_list(stmt.catch_body):
                self._validate_stmt(inner)
            self.scope.pop()
        elif isinstance(stmt, ReturnStmt):
            self._handle_return(stmt)
        elif isinstance(stmt, ExprStmt):
            self._infer_expr_type(stmt.expr)
        else:
            self._add_issue("Unknown statement type", stmt)

    def _handle_return(self, stmt: ReturnStmt):
        fn = self.current_function
        if fn is None:
            return
        if fn.return_type == "Void":
            if stmt.expr is not None:
                self._add_issue("Void function return cannot include a value", stmt)
            return
        if stmt.expr is None:
            self._add_issue("Non-Void function must return a value", stmt)
            return
        expr_type = self._infer_expr_type(stmt.expr)
        self.function_returns_value[fn.name] = True
        if not self._types_compatible(fn.return_type, expr_type):
            self._add_issue(
                f"Return type {expr_type} does not match {fn.return_type}",
                stmt,
            )

    def _infer_expr_type(self, expr: Expr) -> str:
        if isinstance(expr, Literal):
            if isinstance(expr.value, bool):
                return "Bool"
            if isinstance(expr.value, int):
                return "Int"
            if isinstance(expr.value, str):
                return "String"
            if isinstance(expr.value, list):
                return "List"
            return UNKNOWN_TYPE
        if isinstance(expr, VarRef):
            var_type = self.scope.lookup(expr.name)
            if var_type is None:
                self._add_issue(f"Variable '{expr.name}' not defined", expr)
                return UNKNOWN_TYPE
            return var_type
        if isinstance(expr, ListLiteral):
            element_types = [self._infer_expr_type(e) for e in expr.elements]
            inner = self._common_type(element_types)
            return f"List<{inner}>"
        if isinstance(expr, MapLiteral):
            value_types = [self._infer_expr_type(v) for v in expr.items.values()]
            inner = self._common_type(value_types)
            return f"Map<String,{inner}>"
        if isinstance(expr, CallExpr):
            target = self.function_map.get(expr.name)
            if target is None:
                self._add_issue(f"Function '{expr.name}' is not defined", expr)
                return UNKNOWN_TYPE
            if len(expr.args) != len(target.params):
                self._add_issue(
                    f"Function '{expr.name}' expects {len(target.params)} args, got {len(expr.args)}",
                    expr,
                )
            for arg_expr, param in zip(expr.args, target.params):
                arg_type = self._infer_expr_type(arg_expr)
                if not self._types_compatible(param.type, arg_type):
                    self._add_issue(
                        f"Argument for parameter '{param.name}' must be {param.type}, got {arg_type}",
                        expr,
                    )
            return target.return_type
        if isinstance(expr, SyscallExpr):
            if expr.name not in self.available_syscalls:
                self._add_issue(f"Syscall '{expr.name}' not registered", expr)
            for arg in expr.args:
                self._infer_expr_type(arg)
            return "ToolResult"
        if isinstance(expr, BinaryOp):
            if expr.op in {"&&", "||"}:
                left = self._infer_expr_type(expr.left)
                right = self._infer_expr_type(expr.right)
                if left not in {"Bool", UNKNOWN_TYPE}:
                    self._add_issue("Logical operator requires Bool operands", expr.left)
                if right not in {"Bool", UNKNOWN_TYPE}:
                    self._add_issue("Logical operator requires Bool operands", expr.right)
                return "Bool"
            if expr.op in {"==", "!="}:
                self._infer_expr_type(expr.left)
                self._infer_expr_type(expr.right)
                return "Bool"
            if expr.op == "+":
                left = self._infer_expr_type(expr.left)
                right = self._infer_expr_type(expr.right)
                if left == right and left in {"Int", "String"}:
                    return left
                if UNKNOWN_TYPE in {left, right}:
                    return UNKNOWN_TYPE
                self._add_issue("Operator '+' requires matching Int or String operands", expr)
                return UNKNOWN_TYPE
            self._add_issue(f"Unknown binary operator {expr.op}", expr)
            return UNKNOWN_TYPE
        if isinstance(expr, str):
            var_type = self.scope.lookup(expr)
            if var_type is None:
                self._add_issue(f"Identifier '{expr}' not defined", expr)
                return UNKNOWN_TYPE
            return var_type
        self._add_issue("Unknown expression type", expr)
        return UNKNOWN_TYPE

    def _ensure_list(self, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _add_issue(self, message: str, node):
        line = getattr(node, "line", None)
        column = getattr(node, "column", None)
        function = self.current_function.name if self.current_function else None
        self.issues.append(
            ValidationIssue(message=message, line=line, column=column, function=function)
        )

    @staticmethod
    def _common_type(types: List[str]) -> str:
        filtered = [t for t in types if t != UNKNOWN_TYPE]
        if not filtered:
            return UNKNOWN_TYPE
        first = filtered[0]
        for t in filtered[1:]:
            if t != first:
                return UNKNOWN_TYPE
        return first

    @staticmethod
    def _types_compatible(expected: str, actual: str) -> bool:
        if expected == actual:
            return True
        if actual == UNKNOWN_TYPE or expected == UNKNOWN_TYPE:
            return True
        if expected.startswith(LIST_PREFIX) and actual.startswith(LIST_PREFIX):
            return PlanValidator._types_compatible(
                PlanValidator._extract_inner(expected),
                PlanValidator._extract_inner(actual),
            )
        if expected.startswith(MAP_PREFIX) and actual.startswith(MAP_PREFIX):
            return PlanValidator._types_compatible(
                PlanValidator._extract_inner(expected),
                PlanValidator._extract_inner(actual),
            )
        return False

    @staticmethod
    def _extract_inner(type_name: str) -> str:
        start = type_name.find("<") + 1
        end = type_name.rfind(">")
        if start == 0 or end == -1:
            return UNKNOWN_TYPE
        return type_name[start:end]

    @staticmethod
    def _list_inner(type_name: str) -> Optional[str]:
        if not type_name.startswith(LIST_PREFIX) or not type_name.endswith(">"):
            return None
        return type_name[len(LIST_PREFIX) : -1]


__all__ = ["PlanValidator", "ValidationError", "ValidationIssue"]
