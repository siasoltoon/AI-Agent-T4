from __future__ import annotations

import argparse
import json

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
        SETTINGS.max_command_seconds,
        SETTINGS.model_temperature,
        emit=render,
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
    (SETTINGS.state_dir / f"{result.execution_id}.json").write_text(
        json.dumps(result.__dict__, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return 0 if result.status == "completed" else 1


def cmd_status() -> int:
    model = OllamaRuntime(SETTINGS.ollama_host, SETTINGS.model_name, SETTINGS.model_timeout_seconds)
    health = model.health()
    console.print(Panel(
        f"Workspace: {SETTINGS.workspace_root}\n"
        f"State: {SETTINGS.state_dir}\n"
        f"Model: {SETTINGS.model_name}\n"
        f"Ollama: {SETTINGS.ollama_host}\n"
        f"Model runtime: {health.get('ok')}\n"
        f"Model present: {health.get('model_present', False)}\n"
        f"Max steps: {SETTINGS.max_agent_steps}\n"
        f"Max recovery: {SETTINGS.max_recovery_attempts}\n"
        f"Max command seconds: {SETTINGS.max_command_seconds}"
    ))
    return 0 if health.get("ok") and health.get("model_present") else 1


def cmd_health() -> int:
    model = OllamaRuntime(SETTINGS.ollama_host, SETTINGS.model_name, SETTINGS.model_timeout_seconds)
    model_health = model.health()
    gpu = gpu_snapshot()
    console.print(Panel(json.dumps({"model": model_health, "gpu": gpu}, ensure_ascii=False, indent=2)))
    return 0 if model_health.get("ok") and model_health.get("model_present") else 1


def cmd_gpu() -> int:
    console.print_json(json.dumps(gpu_snapshot()))
    return 0 if gpu_snapshot().get("ok") else 1


def cmd_model() -> int:
    model = OllamaRuntime(SETTINGS.ollama_host, SETTINGS.model_name, SETTINGS.model_timeout_seconds)
    data = model.health()
    console.print_json(json.dumps(data, ensure_ascii=False))
    return 0 if data.get("ok") and data.get("model_present") else 1


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
