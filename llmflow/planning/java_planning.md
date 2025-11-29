# Java Planning Specification

This document replaces the legacy CPL grammar with a Java-centric contract. The
language model must emit *compilable Java source* that describes the full plan.

## 1. Source Layout

- Output exactly one `public class Plan` definition.
- Place every helper as a `public` or `private` method inside `Plan`.
- Always include a `public void main()` method as the entrypoint.
- Do not emit package statements, imports, or explanations.

## 2. Method Rules

- Method names must be `camelCase` and describe a single sub-goal.
- Limit each method body to **7 statements**; prefer additional helpers over long
  methods.
- Methods may accept and return the following types: `Void`, `String`, `Int`,
  `Bool`, `ToolResult`, `List<T>`, or `Map<String,T>`.
- Annotate any method that should be synthesized at runtime with `@Deferred`. A
  deferred method may omit its body by ending the signature with a semicolon.
- Non-deferred methods must include a body enclosed in braces.

## 3. Allowed Statements

The runtime only supports a constrained subset of Java. Keep logic within these
constructs:

1. Variable declarations with initializers, e.g.
   `ToolResult repo = syscall.cloneRepo("origin/main");`
2. Assignments to existing variables.
3. Expression statements that invoke another plan method or a syscall.
4. `if/else` blocks with boolean conditions.
5. Enhanced `for` loops (`for (Patch patch : patches) { ... }`) over lists.
6. `try { ... } catch (ToolError e) { ... }` blocks for syscall failures.
7. `return` statements.

Avoid while loops, switch statements, lambda expressions, anonymous classes, or
reflection. Rely on helper methods instead of inline block expressions.

## 4. Syscall Access

- All tool usage must go through the reserved `syscall` helper:
  `syscall.applyTextRewrite(path, before, after);`
- Do **not** invent new syscall names. Use only the whitelist provided in the
  planner prompt.
- Capture syscall returns in typed variables so other methods can reuse results.

## 5. Data Handling

- Prefer values returned from syscalls or helper methods over manual literals.
- When literals are unavoidable, restrict them to primitive strings/ints/bools
  or array initializers (`new String[] {"foo"}`) that the parser can convert.
- Maps must be produced via helper methods or syscalls; manual `HashMap`
  construction is unsupported.

## 6. Deferred Functions

- Use `@Deferred` when the runtime should synthesize the body later.
- Deferred methods may omit their body or provide a sketch that the runtime can
  overwrite.
- Non-deferred methods must include compilable bodies.

## 7. Output Requirements

- Respond exclusively with Java source that satisfies this document.
- Do not wrap the output in markdown fences or add commentary.
- Ensure the program compiles in isolation and references only supplied
  syscalls or helper methods defined in `Plan`.
