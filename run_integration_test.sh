#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

(
  export QLTY_OWNER_KEY='trickl' \
         QLTY_PROJECT_KEY='agent-cortex' \
         GITHUB_REPO_SLUG='trickl/agent-cortex' \
         QUALITY_AGENT_TEST_REPO_URL='git@github.com:trickl/agent-cortex.git' \
         LLMFLOW_SUBGOAL_LLM_CONFIG="${REPO_ROOT}/tests/integration/ollama_llm_config.yaml" \
         PROJECT_REPO_ROOT="$(mktemp -d)"
  .venv/bin/python -m pytest -s -vv tests/integration/test_quality_issue_agent.py -m include_integration_test
)
