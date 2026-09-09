from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from agent_core.executor import AgentExecutor
from config.settings import SETTINGS
from linux_runtime.gpu import gpu_snapshot
from model_runtime.ollama import OllamaRuntime

console = Console()


def runtime() -> AgentExecutor:
    return AgentExecutor(
        OllamaRuntime(SETTINGS.ollama_host, SETTINGS.model_name, SETTINGS.model_timeout_seconds),
        SETTINGS.workspace_root,
        SETTINGS.max_agent_steps,
        SETTINGS.max_recovery_attempts,
        emit=lambda event, data: render(event, data),
    )


def render(event: str, data: dict) -> None:
    if event == "start":
        console.print(Panel(f"[bold]Execution[/bold] {data['execution_id']}\n{data['task']}"))
    elif event == "planning":
        console.print("[bold cyan][AGENT][/bold cyan] Planning...")
    elif event == "step":
        console.print(f"[dim][AGENT] Step {data['number']}/{data['max']}[/dim]")
    elif event == "tool":
        console.print(f"[yellow][TOOL][/yellow] {data['command']}")
    elif event == "observation":
        out = (data.get("stdout") or data.get("stderr") or "").strip()
        if out:
            console.print(out[-5000:])
    elif event == "verify":
        console.print("[bold blue][VERIFY][/bold blue] Running deterministic verification...")
    elif event == "recovery":
        console.print(f"[bold magenta][RECOVERY][/bold magenta] Attempt {data['attempt']}: {data['error']}")
    elif event == "completed":
        console.print(f"[bold green]✓ Task completed[/bold green]  {data['execution_id']}")
        console.print(data["summary"])
    elif event == "failed":
        console.print(f"[bold red]✗ Task failed[/bold red]  {data['execution_id']}")
        console.print(data["error"])


def cmd_run(task: str) -> int:
    result = runtime().run(task)
    SETTINGS.state_dir.mkdir(parents=True, exist_ok=True)
    with (SETTINGS.state_dir / f"{result.execution_id}.json").open("w", encoding="utf-8") as handle:
        json.dump(result.__dict__, handle, ensure_ascii=False, indent=2, default=str)
    return 0 if result.status == "completed" else 1


def cmd_status() -> int:
    model = OllamaRuntime(SETTINGS.ollama_host, SETTINGS.model_name, SETTINGS.model_timeout_seconds)
    console.print(Panel(f"Workspace: {SETTINGS.workspace_root}\nState: {SETTINGS.state_dir}\nModel: {SETTINGS.model_name}\nOllama: {SETTINGS.ollama_host}\nModel runtime: {model.health().get('ok')}"))
    return 0


def cmd_health() -> int:
    cmd_status()
    console.print(json.dumps(gpu_snapshot(), ensure_ascii=False, indent=2))
    return 0


def cmd_gpu() -> int:
    console.print_json(json.dumps(gpu_snapshot()))
    return 0


def cmd_model() -> int:
    model = OllamaRuntime(SETTINGS.ollama_host, SETTINGS.model_name, SETTINGS.model_timeout_seconds)
    console.print_json(json.dumps(model.health(), ensure_ascii=False))
    return 0


def cmd_history() -> int:
    files = sorted(SETTINGS.state_dir.glob("exec_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]
    if not files:
        console.print("No executions recorded.")
        return 0
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        console.print(f"{data['execution_id']}  {data['status']}  steps={data['steps']}  recovery={data['recovery_attempts']}  {data['summary'][:120]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="agent", description="Terminal-first AI Agent for T4/Colab")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run"); p.add_argument("task", nargs="+")
    sub.add_parser("status"); sub.add_parser("health"); sub.add_parser("gpu"); sub.add_parser("model"); sub.add_parser("history")
    args = parser.parse_args()
    if args.command == "run": return cmd_run(" ".join(args.task))
    if args.command == "status": return cmd_status()
    if args.command == "health": return cmd_health()
    if args.command == "gpu": return cmd_gpu()
    if args.command == "model": return cmd_model()
    if args.command == "history": return cmd_history()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
