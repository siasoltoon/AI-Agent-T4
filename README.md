# AI-Agent-T4

Terminal-first AI Agent runtime for ephemeral Google Colab + NVIDIA Tesla T4.

This repository is an independent T4/Colab adaptation of the hardened execution architecture from `AI-Agent-Platform`. The original repository is intentionally not modified by this project.

## Design

- **Terminal only** — no dashboard and no frontend.
- **Local model runtime** — Ollama-compatible HTTP API, with a clean runtime abstraction.
- **Agentic loop** — plan → act → observe → verify, with bounded steps and bounded recovery.
- **Linux-native execution** — shell commands and workspace tools are designed for Colab/Linux.
- **T4 aware** — CUDA/GPU health is exposed directly in the terminal.
- **Ephemeral-safe** — runtime state lives under `.agent_state/` and the repository can bootstrap itself again after a Colab reset.
- **Evidence-based completion** — successful model output alone is never treated as task completion.

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
agent gpu               NVIDIA T4/CUDA snapshot
agent model             Model runtime status
agent history           Recent execution history
agent logs              Recent execution log
agent cancel <id>       Cancel a running execution
```

The terminal is the product UI. The agent streams execution events such as planning, tool calls, verification, recovery and the final execution ID.

## Environment

Expected Colab hardware is a Tesla T4 with CUDA available. The code also runs on CPU for deterministic development/testing, but GPU-backed model inference is the intended deployment.

Default model: `qwen2.5-coder:7b`

Default Ollama endpoint: `http://127.0.0.1:11434`

## Colab lifecycle

A Colab runtime is disposable. Do not store important source-of-truth data only inside the runtime. Clone/synchronize this repository again after a runtime reset, then run `deployment/bootstrap.sh`.

## Validation

```bash
python -m pytest -q
python -m compileall agent_core model_runtime linux_runtime terminal_ui
```
