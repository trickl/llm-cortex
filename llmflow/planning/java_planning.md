# Java Planning Specification

The language model must emit *compilable Java source* that describes the full plan.

## 1. Source Layout

- Output exactly one top-level class. It may be named however you like, and it
  does not need to be `public`.
- Place every helper as a `public` or `private` method inside that class.
- Include a `public void main()` entrypoint when multiple steps need to be
  orchestrated; emitting only a `main` method is acceptable for simple plans.
- Do not emit prose explanations.

## 2. Method Rules

- Limit each method body to **7 statements**; prefer additional helpers over long
  methods.
- Methods may accept and return the following types: `Void`, `String`, `Int`,
  `Bool`, `ToolResult`, `List<T>`, or `Map<String,T>`.
- Invented helper functions must either call more concrete helpers or invoke a
  syscall. Tool calls should only appear at the leaves of the call graph.
- Since invented helpers are implicitly deferred, omit business logic beyond
  chaining helper calls or dispatching a syscall.

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

## 6. Hierarchical Planning Rules

The runtime enforces the following planning discipline:

- **Rule 1 — Invent functions to decompose the problem into steps.** Helpers
  represent smaller sub-goals beneath the caller.
- **Rule 2 — If a step can be completed using an available tool, call the tool
  directly.** Do not invent another helper when a syscall can perform the work.
- **Rule 3 — Each invented function must reduce complexity.** Every helper must
  be strictly more specific than its caller so decomposition steadily converges
  toward concrete actions.
- **Rule 4 — All invented functions are implicitly deferred; tool calls are the
  terminal nodes.** Helper bodies therefore string together other helpers until
  reaching a syscall.
- **Rule 5 — Maximum hierarchical depth is 7.** The root counts as depth 1. If a
  plan would exceed 7 nested helper levels, redesign it using broader steps.

## 7. Output Requirements

- Respond exclusively with Java source that satisfies this document. A single
  assistant-text message containing the class definition is acceptable; a
  `define_java_plan` tool call is not required.
- Do not wrap the output in markdown fences or add commentary.
- Ensure the program compiles in isolation and references only supplied
  syscalls or helper methods defined in your class.
