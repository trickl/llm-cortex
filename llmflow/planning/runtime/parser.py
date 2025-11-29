"""Java source parser that produces the runtime AST."""
from __future__ import annotations

import javalang
from typing import Any, Dict, Iterable, List, Optional

from .ast import (
    Annotation,
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
    Param,
    Plan,
    ReturnStmt,
    Stmt,
    SyscallExpr,
    TryCatchStmt,
    VarDecl,
    VarRef,
)


class PlanParseError(Exception):
    """Raised when Java plan parsing fails."""


def _block_statements(block) -> List[Any]:
    if block is None:
        return []
    statements = getattr(block, "statements", None)
    if statements is not None:
        return statements or []
    if isinstance(block, list):
        return block
    return []


def parse_java_plan(source: str) -> Plan:
    try:
        tree = javalang.parse.parse(source)
    except (javalang.parser.JavaSyntaxError, TypeError) as exc:
        raise PlanParseError(str(exc)) from exc

    class_decl = _extract_plan_class(tree.types)
    functions: List[FunctionDef] = []
    for method in class_decl.methods:
        functions.append(_convert_method(method))
    fn_map = {fn.name: fn for fn in functions}
    return Plan(functions=fn_map, ordered_functions=functions, line=class_decl.position.line if class_decl.position else None)


def parse_java_plan_fragment(fn: FunctionDef, body_source: str) -> List[Stmt]:
    params = ", ".join(f"{param.type} {param.name}" for param in fn.params)
    return_type = fn.return_type or "void"
    class_source = (
        "public class DeferredPlan {\n"
        f"    public {return_type} {fn.name}({params}) {body_source}\n"
        "}\n"
    )
    plan = parse_java_plan(class_source)
    generated = plan.functions.get(fn.name)
    if generated is None or generated.body is None:
        raise PlanParseError(f"Deferred body for '{fn.name}' did not parse")
    return generated.body


def _extract_plan_class(types: Iterable[Any]):
    classes = [type_node for type_node in types if isinstance(type_node, javalang.tree.ClassDeclaration)]
    if not classes:
        raise PlanParseError("Java plan must declare a class")
    for cls in classes:
        if cls.name == "Plan":
            return cls
    return classes[0]


def _convert_method(method: javalang.tree.MethodDeclaration) -> FunctionDef:
    annotations = [_convert_annotation(ann) for ann in method.annotations or []]
    params = [_convert_param(param) for param in method.parameters]
    return_type = _type_to_name(method.return_type)
    body: Optional[List[Stmt]] = None
    if method.body is not None:
        body = _convert_block(method.body)
    line, column = _node_position(method)
    return FunctionDef(
        name=method.name,
        params=params,
        return_type=return_type,
        body=body,
        annotations=annotations,
        line=line,
        column=column,
    )


def _convert_annotation(annotation: javalang.tree.Annotation) -> Annotation:
    args: List[Any] = []
    element = getattr(annotation, "element", None)
    if element is not None:
        if isinstance(element, list):
            args = [_literal_value(item.value) for item in element]
        else:
            args = [_literal_value(element.value)]
    line, column = _node_position(annotation)
    return Annotation(name=annotation.name, args=args, line=line, column=column)


def _convert_param(param: javalang.tree.FormalParameter) -> Param:
    line, column = _node_position(param)
    return Param(name=param.name, type=_type_to_name(param.type), line=line, column=column)


def _convert_block(statements: List[Any]) -> List[Stmt]:
    result: List[Stmt] = []
    for entry in statements:
        node = getattr(entry, "statement", entry)
        if node is None:
            continue
        converted = _convert_statement(node)
        if not converted:
            continue
        if isinstance(converted, list):
            result.extend(converted)
        else:
            result.append(converted)
    return result


def _convert_statement(node: Any) -> Optional[Stmt | List[Stmt]]:
    if isinstance(node, javalang.tree.BlockStatement):
        inner = getattr(node, "statement", None)
        return _convert_statement(inner)

    if isinstance(node, javalang.tree.LocalVariableDeclaration):
        decls: List[Stmt] = []
        for declarator in node.declarators:
            if declarator.initializer is None:
                raise PlanParseError("Variable declarations must include an initializer")
            expr = _convert_expression(declarator.initializer)
            line, column = _node_position(node)
            decls.append(
                VarDecl(
                    name=declarator.name,
                    type=_type_to_name(node.type),
                    expr=expr,
                    line=line,
                    column=column,
                )
            )
        return decls

    if isinstance(node, javalang.tree.StatementExpression):
        return _convert_statement(node.expression)

    if isinstance(node, javalang.tree.Assignment):
        if not isinstance(node.expressionl, javalang.tree.MemberReference):
            raise PlanParseError("Assignments must target identifiers")
        target = node.expressionl.member
        expr = _convert_expression(node.value)
        line, column = _node_position(node)
        return Assign(name=target, expr=expr, line=line, column=column)

    if isinstance(node, javalang.tree.MethodInvocation):
        expr = _convert_method_invocation(node)
        line, column = _node_position(node)
        return ExprStmt(expr=expr, line=line, column=column)

    if isinstance(node, javalang.tree.IfStatement):
        cond = _convert_expression(node.condition)
        then_body = _convert_embedded_block(node.then_statement)
        else_body = _convert_embedded_block(node.else_statement)
        line, column = _node_position(node)
        return IfStmt(cond=cond, then_body=then_body, else_body=else_body, line=line, column=column)

    if isinstance(node, javalang.tree.ForStatement):
        control = getattr(node, "control", None)
        var_decl = getattr(control, "var", None)
        if var_decl is None:
            raise PlanParseError("Only enhanced for-each loops are supported")
        var_name = getattr(var_decl, "name", None)
        if var_name is None and getattr(var_decl, "declarators", None):
            first_decl = var_decl.declarators[0]
            var_name = getattr(first_decl, "name", None)
        if var_name is None:
            raise PlanParseError("Enhanced for loop must declare an iteration variable")
        iterable = _convert_expression(control.iterable)
        body = _convert_embedded_block(node.body)
        line, column = _node_position(node)
        return ForStmt(var_name=var_name, iterable_expr=iterable, body=body, line=line, column=column)

    if isinstance(node, javalang.tree.TryStatement):
        if not node.catches:
            raise PlanParseError("try statement must include a catch block")
        catch = node.catches[0]
        error_var = catch.parameter.name
        try_body = _convert_block(_block_statements(node.block))
        catch_body = _convert_block(_block_statements(catch.block))
        line, column = _node_position(node)
        return TryCatchStmt(
            try_body=try_body,
            error_var=error_var,
            catch_body=catch_body,
            line=line,
            column=column,
        )

    if isinstance(node, javalang.tree.ReturnStatement):
        expr = _convert_expression(node.expression) if node.expression is not None else None
        line, column = _node_position(node)
        return ReturnStmt(expr=expr, line=line, column=column)

    return None


def _convert_embedded_block(statement: Any) -> List[Stmt]:
    if statement is None:
        return []
    if isinstance(statement, javalang.tree.BlockStatement):
        return _convert_block(statement.statements or [])
    converted = _convert_statement(statement)
    if converted is None:
        return []
    if isinstance(converted, list):
        return converted
    return [converted]


def _convert_expression(node: Any) -> Expr:
    if isinstance(node, javalang.tree.Literal):
        value = _literal_value(node.value)
        line, column = _node_position(node)
        return Literal(value=value, line=line, column=column)

    if isinstance(node, javalang.tree.MemberReference):
        line, column = _node_position(node)
        return VarRef(name=node.member, line=line, column=column)

    if isinstance(node, javalang.tree.BinaryOperation):
        left = _convert_expression(node.operandl)
        right = _convert_expression(node.operandr)
        line, column = _node_position(node)
        return BinaryOp(op=node.operator, left=left, right=right, line=line, column=column)

    if isinstance(node, javalang.tree.MethodInvocation):
        return _convert_method_invocation(node)

    if isinstance(node, javalang.tree.ArrayInitializer):
        elements = [_convert_expression(expr) for expr in node.initializers]
        line, column = _node_position(node)
        return ListLiteral(elements=elements, line=line, column=column)

    if isinstance(node, javalang.tree.ClassCreator):
        if isinstance(node.type, javalang.tree.ReferenceType) and node.type.name == "HashMap":
            entries: Dict[str, Expr] = {}
            for pair in node.body or []:
                key = getattr(pair, "name", None)
                if key is None:
                    raise PlanParseError("Map literals must use named fields")
                entries[key] = _convert_expression(pair.value)
            line, column = _node_position(node)
            return MapLiteral(items=entries, line=line, column=column)

    raise PlanParseError(f"Unsupported expression: {type(node).__name__}")


def _convert_method_invocation(node: javalang.tree.MethodInvocation) -> Expr:
    qualifier = node.qualifier
    args = [_convert_expression(arg) for arg in node.arguments]
    line, column = _node_position(node)
    if qualifier == "syscall":
        return SyscallExpr(name=node.member, args=args, line=line, column=column)
    return CallExpr(name=node.member, args=args, line=line, column=column)


def _type_to_name(type_node: Optional[Any]) -> str:
    if type_node is None:
        return "Void"
    if isinstance(type_node, javalang.tree.ReferenceType):
        raw_name = type_node.name
        if isinstance(raw_name, (list, tuple)):
            base = "".join(raw_name)
        else:
            base = str(raw_name)
        args = getattr(type_node, "arguments", None) or []
        if args:
            rendered_args = []
            for arg in args:
                arg_type = getattr(arg, "type", None) or getattr(arg, "pattern_type", None)
                if arg_type is None:
                    continue
                rendered_args.append(_type_to_name(arg_type))
            if rendered_args:
                base = f"{base}<{','.join(rendered_args)}>"
        name = base
    else:
        name = str(type_node.name) if hasattr(type_node, "name") else str(type_node)
    if type_node.dimensions:
        name += "[]" * len(type_node.dimensions)
    return name


def _literal_value(token: Optional[str]) -> Any:
    if token is None:
        return None
    text = token.strip()
    if text.startswith("\"") or text.startswith("'"):
        try:
            return bytes(text[1:-1], "utf-8").decode("unicode_escape")
        except Exception:
            return text[1:-1]
    if text.isdigit():
        return int(text)
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    return text


def _node_position(node: Any) -> tuple[Optional[int], Optional[int]]:
    position = getattr(node, "position", None)
    if not position:
        return None, None
    return position.line, position.column


__all__ = [
    "PlanParseError",
    "parse_java_plan",
    "parse_java_plan_fragment",
]
