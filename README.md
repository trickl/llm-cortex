# AgentCortex
### *An open, modular, extensible kernel for tool-using AI agents*

> **AgentCortex** is an open-source evolution of the excellent  
> **[LLMFlow](https://github.com/kamikaze2020/llmflow)** project.  
> Built on its foundations, this fork expands the original design into a  
> **general-purpose kernel** for planning, reasoning, and connecting large language models  
> to modular tools, agents, and workflows.

---

## 🙏 Credits

This project began as a fork of **LLMFlow** by **kamikaze2020**, whose work provided the core inspiration for a lightweight, structured orchestration layer around LLMs.

We are deeply grateful for the original design, architecture, and ideas contributed by that project. AgentCortex would not exist without it.

---

# 🚀 What is AgentCortex?

**AgentCortex** is the open-source kernel for building powerful, modular, self-improving AI agents that can:

- Think using reasoning LLMs  
- Load tools dynamically  
- Plan multi-step workflows  
- Operate on real codebases  
- Integrate into CI/CD pipelines  
- Improve themselves over time  
- Remain simple, inspectable, and extensible  

It is **not** a monolithic framework.  
It is **not** a heavy dependency tree.  
And it is **not** an attempt at AGI.

Instead, it is a *clean, minimal runtime* for structured agent behaviour — with tools and agents pluggable by design.

---

# 🧠 The Vision

Modern agentic systems (e.g., Copilot Workspace, Cody, Cursor) demonstrate remarkable capabilities —  
but they are:

- closed-source  
- proprietary  
- tightly coupled  
- non-extensible  

Developer communities cannot study them, extend them, or build their own equivalents.

**AgentCortex aims to change that.**

The ambition of this project is to create:

> **A completely open, modular, transparent agentic kernel  
> that the community can extend, improve, and evolve.**

AgentCortex provides the **core reasoning loop**.  
The community provides the **tools and agents**.  
Together, we get a flexible, composable ecosystem.

---

# 🧩 Key Ideas

### **1. The Cortex = Kernel**

The Cortex is intentionally small and stable.  
It handles:

- planning  
- reasoning  
- tool discovery  
- tool loading  
- tool execution  
- agent orchestration  

It does *not* include domain-specific tools.

---

### **2. Tools = Modular Plugins**

Tools live in **separate repositories** and can be:

- Python modules  
- Docker images  
- REST endpoints  
- CLI utilities  
- Dynamically loaded code  

Each tool advertises:

- name  
- capabilities  
- entrypoint  
- version  
- metadata  

Cortex can load tools from:

- local directories  
- Git repositories  
- PyPI / Private indexes  
- Artifactory  
- URLs  
- dynamic registration (ephemeral tools)

---

### **3. Agents = Plans + Tools**

Agents are declarative behaviours such as:

- “Check code quality”  
- “Fix code smells”  
- “Refactor functions”  
- “Improve documentation”  
- “Design new features”  

An agent produces a *plan*.  
The Cortex executes that plan via available tools.

---

### **4. Self-Improvement Loop**

Because tools and agents are modular:

- Agents can improve the Cortex repo itself  
- Agents can improve the tools repo  
- Agents can propose or generate new tools  
- Agents can generate new agents  
- The ecosystem becomes reflexive and self-maintaining  

This is **bounded**, **safe**, and entirely reviewable — but very powerful.

---

# 🏗️ Project Status

AgentCortex is in early development, but actively evolving.

### **Phase 1 (In progress):**
- Functional end-to-end prototype  
- Embedded tools inside Cortex for bootstrapping  
- Jenkins-triggered workflow  
- Example use case: fix “code smells” using QLTY REST API  

### **Phase 2 (Coming next):**
- Extract tools into separate repositories  
- Define plugin contract  
- Implement dynamic loaders (Python, Docker, REST, CLI)  
- Harden the kernel  

### **Phase 3:**
- Documentation, examples, diagrams  
- Developer guides (“How to write a tool/agent”)  
- Branding, naming confirmation  

### **Phase 4:**
- Additional use cases  
- Community-built tools  
- Meta-agents  
- Self-improving workflows  

---

# 📦 Quick Start (early prototype)

```bash
git clone https://github.com/<yourname>/agentcortex
cd agentcortex
pip install -r requirements.txt

# Example run (placeholder)
python main.py --event post_commit --repo https://github.com/your/repo
```

## 🛠 Java Plan Executor

You can execute the new Java-based plans programmatically without touching the
lower-level parser/interpreter APIs. The `PlanExecutor` handles parsing,
validation, execution, and tracing in one shot:

```python
from llmflow.planning.executor import PlanExecutor
from llmflow.runtime.syscalls import build_default_syscall_registry

registry = build_default_syscall_registry()
executor = PlanExecutor(registry)

plan = """
public class Plan {
	public void main() {
		syscall.log("Hello, Cortex!");
		return;
	}
}
"""

result = executor.execute_from_string(plan, capture_trace=True)

if result["success"]:
	print("Plan succeeded", result["trace"])
else:
	print("Plan failed", result["errors"])
```

`result` is always a dict that contains a `success` flag, the interpreter
`return_value`, structured `errors`, optional `trace` events, and any metadata
you pass in.

## 🔌 Syscall Modules

The runtime ships with a batteries-included syscall registry so Java plans can
invoke real git, filesystem, and Qlty tools without extra plumbing. Use
`llmflow.runtime.syscalls.register_default_syscalls` (or the convenience
constructor `llmflow.runtime.syscalls.build_default_syscall_registry`) to
populate a `SyscallRegistry` before handing it to the interpreter or
`PlanExecutor`:

```python
from llmflow.planning.executor import PlanExecutor
from llmflow.runtime.syscalls import build_default_syscall_registry

registry = build_default_syscall_registry()
executor = PlanExecutor(registry)

# Optional: override the logger or swap in custom tool modules.
# register_default_syscalls(registry, logger=my_logger)
```

The default modules expose the following syscall names to plans:

- Utility: `log`
- Git: `cloneRepo`, `createBranch`, `suggestBranchName`, `switchBranch`,
  `stagePaths`, `commitChanges`, `getUncommittedChanges`, `pushBranch`,
  `createPullRequest`
- Files: `listFilesInTree`, `readTextFile`, `overwriteTextFile`, `applyTextRewrite`
- Qlty: `qltyListIssues`, `qltyGetFirstIssue`

Each syscall raises `ToolError` (catchable via `try`/`catch` blocks) when the
underlying tool reports a failure, so plans can rely on consistent error
handling.

## 🧠 Java Plan Synthesizer

Priority 5 introduces a dedicated planner hook so agents can request structured
Java programs before execution. The `JavaPlanner` class automatically loads the
planning specification, injects the allowed syscall list, and returns the raw
Java source together with request metadata.

```python
from llmflow.llm_client import LLMClient
from llmflow.planning import JavaPlanRequest, JavaPlanner

llm = LLMClient()
planner = JavaPlanner(llm)

plan = planner.generate_plan(
	JavaPlanRequest(
		task="Triage the failing lint issue",
		goals=["Fetch the issue details", "Reproduce and patch the failure"],
		allowed_syscalls=["log", "qltyListIssues", "readTextFile"],
	)
)

print(plan.plan_source)
```

Later steps feed this plan into the Java runtime so the agent can execute the
synthesized workflow end-to-end.

### 🔁 Java Plan Retry & Telemetry

Use `PlanOrchestrator` when you want the full generate → execute → repair loop
with structured reporting. The orchestrator automatically:

- retries failed plans (validation or runtime) with bounded repair hints,
- captures per-attempt traces/tool usage, and
- returns a concise human-readable summary you can drop into agent memory.

```python
from llmflow.planning import JavaPlanRequest, JavaPlanner, PlanOrchestrator
from llmflow.planning.plan_runner import PlanRunner

planner = JavaPlanner(llm_client)
orchestrator = PlanOrchestrator(
	planner,
	runner_factory=lambda: PlanRunner(),
	max_retries=2,
)

result = orchestrator.execute_with_retries(
	JavaPlanRequest(
		task="Repair the failing lint issue",
		goals=["Reproduce", "Patch", "Verify"],
		allowed_syscalls=["log", "qltyListIssues"],
	),
	capture_trace=True,
)

print(result["summary"])          # e.g., "✅ Java plan run – 2 attempt(s) …"
print(result["telemetry"])        # structured data for logs or analytics
```

When integrating with the agent loop, write `result["summary"]` back to the
conversation and stash `result["telemetry"]` for diagnostics so controllers and
users can understand exactly what each plan attempted.

## 🧭 Java Plan Agent Workflow

The top-level `Agent` now relies exclusively on the Java planner/orchestrator
stack. Legacy iterative loops have been removed in favour of a deterministic
pipeline:

1. Build a `JavaPlanRequest` from the incoming task, goal memory, and the
   filtered syscall registry (tool access is controlled via tags).
2. Ask `JavaPlanner` to generate the program and schedule it through
   `PlanOrchestrator`, which handles retries and repair prompts.
3. Stream summaries, telemetry, and tool traces back into the agent memory so
   subsequent turns can reason about successes or blockers.

Agents expose a `plan_max_retries` parameter (CLI/config key
`plan_max_retries`) that caps orchestrator attempts per user turn. Setting it to
0 disables retries, while higher values enable the planner to repair invalid or
failing programs automatically. Because the legacy `agent_execution` and
`agent_prompting` modules now raise `RuntimeError` on import, ensure any custom
agents instantiate `llmflow.core.agent.Agent` directly and configure tool tags
via `available_tool_tags`/`match_all_tags`.

## 🤝 Contributing

We welcome contributions of all kinds:

- New tools  
- New agents  
- Documentation improvements  
- Bug fixes  
- Architecture feedback  
- Loader implementations  
- New use-case demos  
- Community discussions  

A full `CONTRIBUTING.md` will be added once the kernel is stable.

---

## 📜 License

**MIT License**

This project is fully open under the MIT license, encouraging broad community contribution and both commercial and non-commercial use.

---

## 🌍 Why This Matters

Developers everywhere are experimenting with agentic workflows —  
but they lack a clean, open, modular foundation to build upon.

**AgentCortex** aims to be that foundation:

- Transparent  
- Hackable  
- Extensible  
- Community-first  
- Pragmatic  
- Practical  

By providing an open kernel for tool-using agents, AgentCortex makes it possible for developers, teams, and communities to build, examine, extend, and improve agentic systems together.

If the project resonates, the community will shape it into something far greater — an ecosystem of shared agents, tools, and ideas built on an open and evolving foundation.
