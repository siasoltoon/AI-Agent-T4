# AI-Agent-T4

Production-oriented, terminal-first autonomous coding agent for ephemeral GPU runtimes such as Google Colab/Tesla T4 and Linux VPS deployments.

`AI-Agent-T4` is an independent evolution of the execution ideas developed in `AI-Agent-Platform`; the original repository is not modified by this project.

## Core capabilities

- **Real Ollama tool calling** — the model selects tools through `/api/chat`; the runtime executes them and returns observations to the model.
- **Autonomous loop** — inspect → reason → act → observe → verify → recover, bounded by step, command and recovery budgets.
- **Deterministic completion gate** — the model must request the `finish` tool; runtime verification is performed before completion is accepted.
- **Coding tools** — shell, targeted file read/write, pytest, git status and git diff.
- **Persistent execution state** — atomic JSON checkpoints under `.agent_state/` make runs auditable and resumable at the runtime level.
- **Remote disaster recovery** — every configured checkpoint can commit execution state and workspace changes to the dedicated `agent-checkpoints` branch; a fresh runtime can fetch and restore the latest or a specific execution when the workspace is clean.
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

       after each tool ──► atomic state ──► Git checkpoint
                                               │
                                               └──► GitHub

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

## Resume after interruption or runtime loss

Execution state is written atomically after tool observations. With `CHECKPOINT_EVERY_TOOL=true`, the workspace and execution state are also committed and pushed to `CHECKPOINT_BRANCH` after each tool when GitHub is reachable. A SIGINT/SIGTERM handler performs a final interruption snapshot and checkpoint before returning control.

If the runtime disappears before a checkpoint can be pushed, only the latest successfully pushed checkpoint can be recovered; a hard kill such as `SIGKILL` cannot be intercepted. This is why frequent checkpoints are enabled by default.

On the same runtime, resume directly:

```bash
agent resume exec_YYYYMMDDHHMMSS_ab12cd34
```

After a fresh clone/runtime, the same command can recover the requested execution from the remote checkpoint branch when the workspace is clean. To recover the newest execution automatically:

```bash
agent resume latest
```

The recovery path refuses to overwrite a dirty workspace. This is deliberate: disaster recovery must never silently destroy unrelated local work.

If GitHub is temporarily unreachable, local `.agent_state` remains usable. Once connectivity returns, a later checkpoint can push the durable state. Remote recovery itself requires the checkpoint branch to have been successfully pushed at least once.

## CLI

```text
agent run <task>       Autonomous coding task
agent resume <id>      Resume a specific execution; restores remote checkpoint if needed
agent resume latest    Resume the newest recoverable execution
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
- `AUTO_GIT_CHECKPOINT`
- `GIT_REMOTE`
- `CHECKPOINT_BRANCH`
- `CHECKPOINT_EVERY_TOOL`

For a T4 with Qwen3-Coder 30B, start conservatively and increase context only after measuring VRAM/RAM usage.

## Verification

```bash
python -m pytest -q
python -m compileall -q agent_core config linux_runtime model_runtime terminal_ui
```

## Colab lifecycle

Colab runtimes are disposable. Keep the repository and configuration as the source of truth; Ollama/model caches and Web Terminal processes may disappear after a runtime reset. Run the bootstrap process again when required, then use `agent resume latest` to recover the newest successfully pushed execution checkpoint.

## Security boundary

The agent is intended to operate on a workspace supplied by the operator. Structured file tools enforce workspace-relative paths. Shell execution is intentionally powerful for software engineering and therefore **is not a hostile-code sandbox**; deploy it inside a dedicated VM/container or other OS-level isolation when tasks are untrusted.

## Engineering principles

1. Evidence beats model claims.
2. Tools beat simulated commands.
3. Verification is mandatory before completion.
4. Failures become observations, not silent retries.
5. Budgets prevent infinite loops.
6. State is persisted atomically.
7. Checkpoints make ephemeral runtimes recoverable.
8. The model proposes actions; deterministic runtime code executes them.
