# GitHub Copilot Instructions for LLMFlow

Use this document when configuring Copilot Custom Instructions (or similar tooling) so that all AI-assisted code suggestions respect the project’s conventions.

## Project Snapshot
- **Repository:** `llm-cortex`
- **Purpose:** Build goal-driven AI agents (LLMFlow) that orchestrate tools via the GAME methodology (Goals, Actions, Memory, Environment).
- **Key Traits:** Tool-rich environment, emphasis on clarity, maintainability, and reliable automation.

## Canonical Coding Style
- **Primary reference:** [`coding-style.md`](../coding-style.md). This file summarizes the public-domain guidance from PEP 8 plus repo-specific expectations.
- Copilot must prefer the patterns in `coding-style.md` whenever it autocompletes, rewrites, or suggests code/documentation.
- When unsure, Copilot should echo the relevant rule (e.g., import ordering, naming, docstring expectations) before proposing an alternative.

## Prompting Guideline for Users
When adding Copilot Custom Instructions, include language such as:
> “For the `llm-cortex` project, strictly follow the repository’s `coding-style.md` plus any guidance in `.github/copilot-instructions.md`.”

## Implementation Expectations for Copilot
1. **Imports & Layout:** Keep imports grouped (stdlib → third-party → local) and insert blank lines exactly as described in `coding-style.md`.
2. **Naming:** Suggest `CapWords` for classes/exceptions, `lower_case_with_underscores` for functions/variables, and `UPPER_CASE` for constants.
3. **Type Hints:** Format annotations with one space after `:` and spaces around `=` when defaults accompany annotations.
4. **Docstrings & Comments:** Provide PEP 257-compliant docstrings for public functions/classes and keep inline comments meaningful.
5. **Control Flow & Complexity:** Offer small, focused helpers instead of deeply nested logic; ensure return paths remain consistent and explicit.
6. **Testing & Docs:** Whenever Copilot proposes new features, it should also suggest corresponding tests and README/docs updates.
7. **Quality Gates:** Default to the repo’s automation stack—`black`, `isort`, `pylint`, and `pytest`—and remind users to run `pre-commit` hooks or the equivalent commands before committing.
8. **Module Size & Structure:** Keep each source file under 300 lines; when a module grows, refactor code into helpers or new files instead of squeezing additional logic in.
9. **Fail Fast:** Do not introduce fallbacks that hide missing capabilities. Raise explicit errors when prerequisites (APIs, SDKs, tool support) aren’t available so issues surface immediately. Do not introduce fallbacks for unexpected behavior. Fail fast and visibly.

## Review Checklist
Before accepting Copilot changes:
- Confirm they cite or comply with `coding-style.md` rules.
- Ensure suggested code paths include docstrings or comments when the style guide requires them.
- Verify that any new files reference the style guide when relevant (e.g., new tooling docs, development guides).

Keeping this file updated—and referencing it in Copilot’s Custom Instructions—ensures AI assistance remains aligned with the team’s standards.