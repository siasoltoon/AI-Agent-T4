# AI-Agent-T4

Production-oriented, terminal-first autonomous coding agent for ephemeral GPU runtimes such as Google Colab/Tesla T4 and Linux VPS deployments.

`AI-Agent-T4` is an independent evolution of the execution ideas developed in `AI-Agent-Platform`; the original repository is not modified by this project.

## Core capabilities

- **Real Ollama tool calling** — the model selects tools through `/api/chat`; the runtime executes them and returns observations to the model.
- **Autonomous loop** — inspect → reason → act → observe → verify → recover, bounded by step, command and recovery budgets.
- **Deterministic completion gate** — the model must request the `finish` tool; runtime verification is performed before completion is accepted.
- **Coding tools** — shell, targeted file read/write, pytest, git status and git diff.
- **Persistent execution state** — atomic JSON checkpoints under `.agent_state/` make runs auditable and resumable at the runtime level.
- **Failure recovery** — failed tool calls feed explicit failure evidence back into the next model turn.
- **Workspace path controls** — structured file tools reject paths that resolve outside the configured workspace.
- **Terminal-first UX** — no dashboard or frontend dependency.
- **GPU-aware deployment** — intended for local Ollama inference with CUDA/T4, while remaining testable without a GPU.

## Architecture

```text
                 User Task
                    │
                    ▼
             terminal_ui / CLI
                    │
                    ▼
              AgentExecutor
          ┌─────────┼─────────┐
          │         │         │
       Context    Policy    State
          │         │         │
          └─────────┼─────────┘
                    ▼
              OllamaRuntime
                    │
              Qwen3-Coder 30B
                    │
              native tool calls
                    ▼
              WorkspaceTools
        ┌──────┬──────┬──────┬──────┐
        ▼      ▼      ▼      ▼      ▼
      shell  files  pytest   git   finish
        │      │      │      │      │
        └──────┴──────┴──────┴──────┘
                    │
                    ▼
                Observation
                    │
                    └──────► next model turn

             finish ──► deterministic verify ──► COMPLETED
                         │
                         └──────────────► recovery if failed
```

## Model

Default model: `qwen3-coder:30b`.

Default Ollama endpoint: `http://127.0.0.1:11434`.

The runtime does not assume that model text is executable. Tool calls are explicit structured actions. A model response without a tool call cannot declare success.

## Quick start

```bash
bash deployment/bootstrap.sh
source .venv/bin/activate
agent status
agent health
agent run "Inspect this repository, run the tests, fix any failures, and verify the final state."
agent history
```

## CLI

```text
agent run <task>       Autonomous coding task
agent status            Runtime/model configuration
agent health            Model + GPU health
agent gpu               NVIDIA GPU snapshot
agent model             Ollama/model availability
agent history           Recent execution records
```

## Configuration

Copy `.env.example` to `.env` and tune the runtime for the deployment. Important controls include:

- `MODEL_NAME`
- `OLLAMA_HOST`
- `WORKSPACE_ROOT`
- `STATE_DIR`
- `MAX_AGENT_STEPS`
- `MAX_RECOVERY_ATTEMPTS`
- `MAX_COMMAND_SECONDS`
- `MODEL_TIMEOUT_SECONDS`
- `MODEL_TEMPERATURE`
- `MAX_CONTEXT_CHARS`
- `MAX_TOOL_OUTPUT_CHARS`

For a T4 with Qwen3-Coder 30B, start conservatively and increase context only after measuring VRAM/RAM usage.

## Verification

```bash
python -m pytest -q
python -m compileall -q agent_core config linux_runtime model_runtime terminal_ui
```

## Colab lifecycle

Colab runtimes are disposable. Keep the repository and configuration as the source of truth; Ollama/model caches and Web Terminal processes may disappear after a runtime reset. Run the bootstrap process again when required.

## Security boundary

The agent is intended to operate on a workspace supplied by the operator. Structured file tools enforce workspace-relative paths. Shell execution is intentionally powerful for software engineering and therefore **is not a hostile-code sandbox**; deploy it inside a dedicated VM/container or other OS-level isolation when tasks are untrusted.

## Engineering principles

1. Evidence beats model claims.
2. Tools beat simulated commands.
3. Verification is mandatory before completion.
4. Failures become observations, not silent retries.
5. Budgets prevent infinite loops.
6. State is persisted atomically.
7. The model proposes actions; deterministic runtime code executes them.
