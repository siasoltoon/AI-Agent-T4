from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent_core.tools import TOOL_SCHEMAS, WorkspaceTools, dispatch, tool_result_message
from linux_runtime.shell import LinuxShell
from model_runtime.ollama import OllamaRuntime

SYSTEM_PROMPT = """You are a production-grade autonomous software engineering agent.
Operate by inspect -> reason -> act -> observe -> verify -> recover. Never claim completion from intention.
You have deterministic tools. Use them instead of inventing command output or file contents.
Prefer read_file for targeted inspection, shell for discovery/build commands, write_file for precise edits, git_diff/status for change awareness, and pytest for tests.
Keep changes minimal and directly related to the task. Never modify tests unless the user explicitly requests it.
Before finishing, run the strongest relevant verification available and only finish when the evidence supports the claim.
If a tool fails, inspect the error, adapt, and retry within the remaining budget.
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
    def __init__(self, model: OllamaRuntime, workspace: Path, max_steps: int = 32, max_recovery: int = 6, max_command_seconds: int = 600, temperature: float = 0.1, emit: Callable[[str, dict[str, Any]], None] | None = None, state_dir: Path | None = None) -> None:
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
        wants_tests = any(x in lower for x in ("test", "tests", "pytest", "test suite", "verify"))
        if wants_tests or any(str(h.get("tool")) == "pytest" for h in history):
            result = self.tools.pytest()
            checks.append({"type": "pytest", "passed": result.get("ok") is True, "stdout": result.get("stdout", ""), "stderr": result.get("stderr", "")})
        return {"passed": all(c["passed"] for c in checks), "checks": checks}

    def _legacy_turn(self, task: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = json.dumps({"task": task, "workspace": str(self.workspace), "observations": history[-10:]}, ensure_ascii=False, default=str)
        text = self.model.generate(prompt, system=SYSTEM_PROMPT + " Return exactly one JSON action object.", temperature=self.temperature)
        return json.loads(text)

    def _chat_turn(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return self.model.chat(messages, tools=TOOL_SCHEMAS, temperature=self.temperature)

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
                    response = self._chat_turn(messages)
                    message = response.get("message") or {}
                    tool_calls = message.get("tool_calls") or []
                    content = str(message.get("content") or "").strip()
                    messages.append(message)
                    if not tool_calls:
                        if content:
                            verification = self._verify(task, history)
                            self.emit("verify", verification)
                            if verification["passed"]:
                                result = ExecutionResult(execution_id, "completed", content, step, recovery, verification["checks"])
                                self._persist(execution_id, result.__dict__)
                                self.emit("completed", {"summary": content, "execution_id": execution_id})
                                return result
                        raise RuntimeError("Model produced no tool call and no verifiable completion.")
                    for call in tool_calls:
                        fn = call.get("function") or {}
                        name = str(fn.get("name") or "")
                        arguments = fn.get("arguments") or {}
                        if isinstance(arguments, str):
                            arguments = json.loads(arguments)
                        self.emit("tool", {"tool": name, "command": arguments.get("command", name)})
                        result = dispatch(self.tools, name, arguments)
                        record = {"step": step, "tool": name, "arguments": arguments, "ok": result.get("ok", True), "result": result}
                        history.append(record)
                        self.emit("observation", {"tool": name, "stdout": result.get("stdout", ""), "stderr": result.get("stderr", "")})
                        messages.append(tool_result_message(str(call.get("id") or uuid.uuid4().hex), result))
                        if result.get("ok") is False:
                            raise RuntimeError(f"Tool {name} failed: {result.get('stderr') or result.get('error') or 'unknown error'}")
                else:
                    decision = self._legacy_turn(task, history)
                    action = str(decision.get("action", "")).lower()
                    if action == "finish":
                        verification = self._verify(task, history)
                        self.emit("verify", verification)
                        if verification["passed"]:
                            result = ExecutionResult(execution_id, "completed", str(decision.get("summary", "Task completed and verified.")), step, recovery, verification["checks"])
                            self._persist(execution_id, result.__dict__)
                            return result
                        raise RuntimeError("Completion rejected: deterministic verification failed.")
                    if action != "tool" or str(decision.get("tool", "")).lower() != "shell":
                        raise ValueError("Legacy runtime accepts only shell actions.")
                    args = decision.get("arguments") or {}
                    command = str(args.get("command", "")).strip()
                    result = self.tools.shell(command, int(args.get("timeout", 120)))
                    record = {"step": step, "tool": "shell", "arguments": args, "ok": result.get("ok"), "result": result}
                    history.append(record)
                    self.emit("tool", {"tool": "shell", "command": command})
                    self.emit("observation", {"tool": "shell", "stdout": result.get("stdout", ""), "stderr": result.get("stderr", "")})
                    if not result.get("ok"):
                        raise RuntimeError(result.get("stderr", "Command failed"))
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
                    messages.append({"role": "user", "content": f"Tool/runtime failure: {exc}. Diagnose it and continue from the current state."})

        message = f"Maximum agent steps ({self.max_steps}) reached without verified completion."
        result = ExecutionResult(execution_id, "failed", message, self.max_steps, recovery, history)
        self._persist(execution_id, result.__dict__)
        self.emit("failed", {"error": message, "execution_id": execution_id})
        return result
