# AI-Agent-T4

Terminal-first AI Agent runtime for ephemeral Google Colab + NVIDIA Tesla T4.

This repository is an independent T4/Colab adaptation of the hardened execution architecture from `AI-Agent-Platform`. The original repository is intentionally not modified by this project.

## Design

- **Terminal only** — no dashboard and no frontend.
- **Local model runtime** — Ollama-compatible HTTP API with a clean runtime abstraction.
- **Agentic loop** — plan → act → observe → verify, with bounded steps and bounded recovery.
- **Linux-native execution** — shell commands run from a controlled workspace working directory.
- **T4 aware** — NVIDIA GPU health is exposed directly in the terminal.
- **Ephemeral-safe** — execution records live under `.agent_state/` and can be recreated after a Colab reset.
- **Evidence-based completion** — model-generated claims are not accepted as completion without runtime verification.

## Quick start

```bash
bash deployment/bootstrap.sh
source .venv/bin/activate
agent status
agent gpu
agent model
agent run "Inspect this repository, run the tests, and report verified results."
```

## CLI

```text
agent run <task>       Run an autonomous task
agent status            Runtime and workspace status
agent health            Full local health check
agent gpu               NVIDIA GPU snapshot
agent model             Ollama/model status
agent history           Recent execution history
```

The terminal is the product UI. Execution events are rendered live: planning, step number, tool command, observation, verification, recovery, and the final execution ID.

## Environment

Expected deployment is a Colab runtime with a Tesla T4 and CUDA available. The code also runs on CPU for deterministic development/testing, but GPU-backed model inference is the intended deployment.

Default model: `qwen2.5-coder:7b`

Default Ollama endpoint: `http://127.0.0.1:11434`

## Colab lifecycle

A Colab runtime is disposable. GitHub is the source of truth. After a runtime reset, clone this repository again and run `deployment/bootstrap.sh`.

## Validation

```bash
python -m pytest -q
python -m compileall -q agent_core config linux_runtime model_runtime terminal_ui
```

## Architecture

```text
Web Terminal / ttyd
        │
        ▼
  terminal_ui (CLI)
        │
        ▼
  agent_core (plan/act/observe/verify)
        │
   ┌────┴─────┐
   ▼          ▼
Ollama     linux_runtime
   │          │
   ▼          ▼
T4 GPU     Colab workspace
```

There is deliberately no dashboard/frontend and no remote PC-worker dependency in this version.
