from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent_core.tools import TOOL_SCHEMAS, WorkspaceTools, dispatch, tool_result_message
from linux_runtime.shell import LinuxShell

SYSTEM_PROMPT = """You are a production-grade autonomous software engineering agent.
Operate by inspect -> reason -> act -> observe -> verify -> recover. Never claim completion from intention.
Use deterministic tools instead of inventing command output or file contents.
Prefer targeted inspection, minimal edits, real tests, and git diff/status checks. Never modify tests unless explicitly requested.
Use the finish tool only after the strongest relevant verification has passed. If a tool fails, diagnose the evidence and continue within the budget.
All file paths must remain inside the configured workspace.
"""


@dataclass
class ExecutionResult:
    execution_id: str
    status: str
    summary: str
    steps: int
    recovery_attempts: int
    evidence: list[dict[str, Any]]


class AgentExecutor:
    def __init__(self, model: Any, workspace: Path, max_steps: int = 32, max_recovery: int = 6, max_command_seconds: int = 600, temperature: float = 0.1, emit: Callable[[str, dict[str, Any]], None] | None = None, state_dir: Path | None = None) -> None:
        self.model = model
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.shell = LinuxShell(self.workspace, max_seconds=max_command_seconds)
        self.tools = WorkspaceTools(self.workspace, max_command_seconds=max_command_seconds)
        self.max_steps = max(1, min(int(max_steps), 128))
        self.max_recovery = max(0, min(int(max_recovery), 12))
        self.temperature = max(0.0, min(float(temperature), 1.0))
        self.emit = emit or (lambda _event, _data: None)
        self.state_dir = (state_dir or (self.workspace / ".agent_state")).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _persist(self, execution_id: str, payload: dict[str, Any]) -> None:
        target = self.state_dir / f"{execution_id}.json"
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        tmp.replace(target)

    def _verify(self, task: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        status = self.tools.git_status()
        checks.append({"type": "workspace", "passed": status.get("ok") is True, "stdout": status.get("stdout", ""), "stderr": status.get("stderr", "")})
        lower = task.lower()
        if any(x in lower for x in ("test", "tests", "pytest", "test suite", "verify")) or any(h.get("tool") == "pytest" for h in history):
            result = self.tools.pytest()
            checks.append({"type": "pytest", "passed": result.get("ok") is True, "stdout": result.get("stdout", ""), "stderr": result.get("stderr", "")})
        return {"passed": all(c["passed"] for c in checks), "checks": checks}

    def _legacy_turn(self, task: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = json.dumps({"task": task, "workspace": str(self.workspace), "observations": history[-10:]}, ensure_ascii=False, default=str)
        return json.loads(self.model.generate(prompt, system=SYSTEM_PROMPT + " Return exactly one JSON action object.", temperature=self.temperature))

    def _finish(self, execution_id: str, task: str, history: list[dict[str, Any]], step: int, recovery: int, summary: str) -> ExecutionResult | None:
        verification = self._verify(task, history)
        self.emit("verify", verification)
        if not verification["passed"]:
            return None
        result = ExecutionResult(execution_id, "completed", summary, step, recovery, verification["checks"])
        self._persist(execution_id, result.__dict__)
        self.emit("completed", {"summary": summary, "execution_id": execution_id})
        return result

    def run(self, task: str) -> ExecutionResult:
        execution_id = f"exec_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        history: list[dict[str, Any]] = []
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Workspace: {self.workspace}\nTask: {task}"}]
        recovery = 0
        self.emit("start", {"execution_id": execution_id, "task": task})
        self.emit("planning", {"message": "Planning and executing autonomously..."})
        self._persist(execution_id, {"execution_id": execution_id, "status": "running", "task": task, "steps": 0, "recovery_attempts": 0, "evidence": []})

        for step in range(1, self.max_steps + 1):
            self.emit("step", {"number": step, "max": self.max_steps})
            try:
                if hasattr(self.model, "chat"):
                    response = self.model.chat(messages, tools=TOOL_SCHEMAS, temperature=self.temperature)
                    message = response.get("message") or {}
                    tool_calls = message.get("tool_calls") or []
                    messages.append(message)
                    if not tool_calls:
                        raise RuntimeError("Model stopped without a tool call; completion requires the finish tool.")
                    for call in tool_calls:
                        fn = call.get("function") or {}
                        name = str(fn.get("name") or "")
                        arguments = fn.get("arguments") or {}
                        if isinstance(arguments, str): arguments = json.loads(arguments)
                        self.emit("tool", {"tool": name, "command": arguments.get("command", name)})
                        result = dispatch(self.tools, name, arguments)
                        record = {"step": step, "tool": name, "arguments": arguments, "ok": result.get("ok", True), "result": result}
                        history.append(record)
                        self.emit("observation", {"tool": name, "stdout": result.get("stdout", ""), "stderr": result.get("stderr", "")})
                        if name == "finish" and result.get("finish_request"):
                            completed = self._finish(execution_id, task, history, step, recovery, result.get("summary", "Task completed and verified."))
                            if completed is not None: return completed
                            raise RuntimeError("Finish request rejected because deterministic verification failed.")
                        messages.append(tool_result_message(str(call.get("id") or uuid.uuid4().hex), result, name))
                        if result.get("ok") is False:
                            raise RuntimeError(f"Tool {name} failed: {result.get('stderr') or result.get('error') or 'unknown error'}")
                else:
                    decision = self._legacy_turn(task, history)
                    if str(decision.get("action", "")).lower() == "finish":
                        completed = self._finish(execution_id, task, history, step, recovery, str(decision.get("summary", "Task completed and verified.")))
                        if completed is not None: return completed
                        raise RuntimeError("Completion rejected: deterministic verification failed.")
                    if str(decision.get("action", "")).lower() != "tool" or str(decision.get("tool", "")).lower() != "shell":
                        raise ValueError("Legacy runtime accepts only shell actions.")
                    args = decision.get("arguments") or {}
                    result = self.tools.shell(str(args.get("command", "")), int(args.get("timeout", 120)))
                    history.append({"step": step, "tool": "shell", "arguments": args, "ok": result.get("ok"), "result": result})
                    self.emit("tool", {"tool": "shell", "command": args.get("command", "")})
                    self.emit("observation", {"tool": "shell", "stdout": result.get("stdout", ""), "stderr": result.get("stderr", "")})
                    if not result.get("ok"): raise RuntimeError(result.get("stderr", "Command failed"))
            except Exception as exc:
                if recovery >= self.max_recovery:
                    result = ExecutionResult(execution_id, "failed", str(exc), step, recovery, history)
                    self._persist(execution_id, result.__dict__)
                    self.emit("failed", {"error": str(exc), "execution_id": execution_id})
                    return result
                recovery += 1
                self.emit("recovery", {"attempt": recovery, "error": str(exc)})
                history.append({"type": "failure", "error": str(exc), "recovery_attempt": recovery})
                if hasattr(self.model, "chat"):
                    messages.append({"role": "user", "content": f"Runtime failure: {exc}. Inspect the current state and continue; do not claim completion yet."})

        message = f"Maximum agent steps ({self.max_steps}) reached without verified completion."
        result = ExecutionResult(execution_id, "failed", message, self.max_steps, recovery, history)
        self._persist(execution_id, result.__dict__)
        self.emit("failed", {"error": message, "execution_id": execution_id})
        return result
