# Agent Configuration

This directory contains declarative presets that describe how an agent should be
instantiated (tools, prompts, workflow constraints, and environment bindings).

## quality_issue_agent.yaml (OpenAI default)

This preset powers the "quality issue autofixer" scenario when running the
agent with OpenAI-hosted models. Highlights:

- **Base context** – opinionated system prompt that tells the agent to review
  Qlty issues, edit the repository, and ship fixes via pull requests.
- **Tooling** – enables git, file discovery, file editing, and Qlty API helpers.
  Add or remove entries under `tools.tags.include` or explicitly list individual
  tool names under `tools.explicit` for tighter control.
- **Environment** – all workspace-specific identifiers (repository root,
  workspace/project pairs, tokens) are referenced via environment variables so
  they can be swapped per deployment without editing the preset.
- **Workflow** – iteration budget, reflection cadence, and branch naming rules
  are parameterised and can be overridden by downstream orchestration code.

To wire this preset into an agent runner, load the YAML (e.g. with
`yaml.safe_load`), hydrate any `${VAR}` placeholders from `os.environ`, then
hand the parsed structure to your `Agent` initialiser.

## quality_issue_agent_ollama.yaml

Granite/Ollama-specific variant of the same preset. The workflow, tools, and
environment bindings mirror the OpenAI file, but the `llm` block targets
`granite4:3b` along with a longer timeout suitable for local inference. When
running against Granite (or any Ollama model without reliable tool calls), set
`structured_mode: "json"` in the corresponding `llm_config.yaml` so Instructor
parses plain JSON instead of expecting OpenAI-style tool invocations. The
integration test suite references this file so logs and documentation stay in
sync with the model actually under test.
