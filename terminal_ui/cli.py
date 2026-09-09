from __future__ import annotations

import argparse
import json

from rich.console import Console
from rich.panel import Panel

from agent_core.durability import DurableState
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
        state_dir=SETTINGS.state_dir,
        auto_git_checkpoint=SETTINGS.auto_git_checkpoint,
        git_remote=SETTINGS.git_remote,
        checkpoint_branch=SETTINGS.checkpoint_branch,
        checkpoint_every_tool=SETTINGS.checkpoint_every_tool,
    )


def durable() -> DurableState:
    return DurableState(
        SETTINGS.state_dir,
        SETTINGS.workspace_root,
        SETTINGS.auto_git_checkpoint,
        SETTINGS.git_remote,
        SETTINGS.checkpoint_branch,
    )


def render(event: str, data: dict) -> None:
    if event == "start": console.print(Panel(f"[bold]Execution[/bold] {data['execution_id']}\n{data['task']}"))
    elif event == "planning": console.print("[bold cyan][AGENT][/bold cyan] Planning...")
    elif event == "step": console.print(f"[dim][AGENT] Step {data['number']}/{data['max']}[/dim]")
    elif event == "tool": console.print(f"[yellow][TOOL][/yellow] {data.get('command', data.get('tool', ''))}")
    elif event == "observation":
        out = (data.get("stdout") or data.get("stderr") or "").strip()
        if out: console.print(out[-5000:])
    elif event == "verify": console.print("[bold blue][VERIFY][/bold blue] Running deterministic verification...")
    elif event == "recovery": console.print(f"[bold magenta][RECOVERY][/bold magenta] Attempt {data['attempt']}: {data['error']}")
    elif event == "checkpoint_warning": console.print(f"[bold yellow][CHECKPOINT][/bold yellow] {data.get('error', 'GitHub checkpoint unavailable')}")
    elif event == "interrupted": console.print(f"[bold yellow]⚠ Execution interrupted and checkpointed[/bold yellow]  {data['execution_id']}")
    elif event == "completed":
        console.print(f"[bold green]✓ Task completed[/bold green]  {data['execution_id']}")
        console.print(data["summary"])
    elif event == "failed":
        console.print(f"[bold red]✗ Task failed[/bold red]  {data['execution_id']}")
        console.print(data["error"])


def cmd_run(task: str) -> int:
    result = runtime().run(task)
    return 0 if result.status == "completed" else 1


def cmd_resume(execution_id: str) -> int:
    state = durable()
    if execution_id == "latest":
        running = state.list_running()
        if running:
            execution_id = str(running[0]["execution_id"])
        else:
            restored = state.restore_from_remote()
            if not restored.get("ok"):
                console.print(f"[bold red]✗ Could not restore latest checkpoint:[/bold red] {restored.get('error')}")
                return 1
            execution_id = str(restored["execution_id"])
            console.print(f"[bold cyan][RECOVERY][/bold cyan] Restored {execution_id} from {restored['branch']} @ {restored['commit'][:12]}")
    else:
        try:
            state.load(execution_id)
        except FileNotFoundError:
            restored = state.restore_from_remote(execution_id)
            if not restored.get("ok"):
                console.print(f"[bold red]✗ Execution not available:[/bold red] {restored.get('error')}")
                return 1
            console.print(f"[bold cyan][RECOVERY][/bold cyan] Restored {execution_id} from {restored['branch']} @ {restored['commit'][:12]}")
    result = runtime().resume(execution_id)
    return 0 if result.status == "completed" else 1


def cmd_status() -> int:
    model = OllamaRuntime(SETTINGS.ollama_host, SETTINGS.model_name, SETTINGS.model_timeout_seconds)
    health = model.health()
    running = durable().list_running()
    console.print(Panel(f"Workspace: {SETTINGS.workspace_root}\nState: {SETTINGS.state_dir}\nModel: {SETTINGS.model_name}\nOllama: {SETTINGS.ollama_host}\nModel runtime: {health.get('ok')}\nModel present: {health.get('model_present', False)}\nMax steps: {SETTINGS.max_agent_steps}\nMax recovery: {SETTINGS.max_recovery_attempts}\nMax command seconds: {SETTINGS.max_command_seconds}\nCheckpoint branch: {SETTINGS.checkpoint_branch}\nRunning executions: {len(running)}"))
    return 0 if health.get("ok") and health.get("model_present") else 1


def cmd_health() -> int:
    model = OllamaRuntime(SETTINGS.ollama_host, SETTINGS.model_name, SETTINGS.model_timeout_seconds)
    model_health = model.health()
    gpu = gpu_snapshot()
    console.print(Panel(json.dumps({"model": model_health, "gpu": gpu}, ensure_ascii=False, indent=2)))
    return 0 if model_health.get("ok") and model_health.get("model_present") else 1


def cmd_gpu() -> int:
    data = gpu_snapshot()
    console.print_json(json.dumps(data))
    return 0 if data.get("ok") else 1


def cmd_model() -> int:
    data = OllamaRuntime(SETTINGS.ollama_host, SETTINGS.model_name, SETTINGS.model_timeout_seconds).health()
    console.print_json(json.dumps(data, ensure_ascii=False))
    return 0 if data.get("ok") and data.get("model_present") else 1


def cmd_history() -> int:
    files = sorted(SETTINGS.state_dir.glob("exec_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]
    if not files:
        console.print("No executions recorded.")
        return 0
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        console.print(f"{data['execution_id']}  {data['status']}  steps={data['steps']}  recovery={data['recovery_attempts']}  {str(data['summary'])[:120]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="agent", description="Production terminal-first autonomous coding agent")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run"); p.add_argument("task", nargs="+")
    r = sub.add_parser("resume"); r.add_argument("execution_id", help="Execution ID or 'latest'")
    sub.add_parser("status"); sub.add_parser("health"); sub.add_parser("gpu"); sub.add_parser("model"); sub.add_parser("history")
    args = parser.parse_args()
    if args.command == "run": return cmd_run(" ".join(args.task))
    if args.command == "resume": return cmd_resume(args.execution_id)
    if args.command == "status": return cmd_status()
    if args.command == "health": return cmd_health()
    if args.command == "gpu": return cmd_gpu()
    if args.command == "model": return cmd_model()
    if args.command == "history": return cmd_history()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
