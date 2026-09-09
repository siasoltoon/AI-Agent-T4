from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from linux_runtime.shell import LinuxShell
from model_runtime.ollama import ModelRuntimeError, OllamaRuntime


SYSTEM_PROMPT = '''You are an autonomous software agent running in a Linux terminal workspace.
You must act, observe, and verify. Never claim a task is complete merely because you proposed a solution.
At every turn return exactly one JSON object and no markdown:
{"action":"tool","tool":"shell","arguments":{"command":"...","timeout":120}}
or
{"action":"finish","summary":"...","verification":"..."}
Use shell for inspection, editing, tests, git, Python and other commands. Keep commands focused.
After mutations, verify them with a real command. If a command fails, diagnose and retry when useful.
Never use paths outside the configured workspace unless the user explicitly asks for system inspection.
'''


@dataclass
class ExecutionResult:
    execution_id: str
    status: str
    summary: str
    steps: int
    recovery_attempts: int
    evidence: list[dict[str, Any]]


class AgentExecutor:
    def __init__(self, model: OllamaRuntime, workspace: Path, max_steps: int = 32, max_recovery: int = 6, emit: Callable[[str, dict[str, Any]], None] | None = None) -> None:
        self.model = model
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.shell = LinuxShell(self.workspace)
        self.max_steps = max(1, min(max_steps, 64))
        self.max_recovery = max(0, min(max_recovery, 6))
        self.emit = emit or (lambda _event, _data: None)

    @staticmethod
    def _json(text: str) -> dict[str, Any]:
        text = str(text).strip()
        candidates = [text]
        candidates += re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.I | re.S)
        for candidate in candidates:
            try:
                value = json.loads(candidate)
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                pass
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                value = json.loads(match.group(0))
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                pass
        raise ValueError("Model did not return a valid JSON action.")

    def _context(self, task: str, history: list[dict[str, Any]]) -> str:
        recent = history[-8:]
        return json.dumps({"task": task, "workspace": str(self.workspace), "observations": recent}, ensure_ascii=False, default=str)[-24000:]

    def _verify(self, task: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        mutation_seen = any(h.get("tool") == "shell" and any(x in str(h.get("command", "")) for x in (" > ", "mkdir", "cp ", "mv ", "rm ", "git commit", "sed -i", "python")) for h in history)
        if mutation_seen:
            check = self.shell.run("git status --short 2>/dev/null || true", timeout=30)
            checks.append({"type": "workspace_state", "passed": check.get("returncode") == 0, "stdout": check.get("stdout", "")})
        verify_words = ("test", "verify", "check", "ensure", "run")
        if any(word in task.lower() for word in verify_words):
            check = self.shell.run("python -m pytest -q", timeout=120)
            checks.append({"type": "pytest", "passed": check.get("ok") is True, "stdout": check.get("stdout", ""), "stderr": check.get("stderr", "")})
        passed = all(c["passed"] for c in checks) if checks else True
        return {"passed": passed, "checks": checks}

    def run(self, task: str) -> ExecutionResult:
        execution_id = f"exec_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        history: list[dict[str, Any]] = []
        recovery = 0
        self.emit("start", {"execution_id": execution_id, "task": task})
        self.emit("planning", {"message": "Planning and executing autonomously..."})

        for step in range(1, self.max_steps + 1):
            self.emit("step", {"number": step, "max": self.max_steps})
            try:
                prompt = self._context(task, history)
                response = self.model.generate(prompt, system=SYSTEM_PROMPT)
                decision = self._json(response)
                action = str(decision.get("action", "")).lower()
                if action == "finish":
                    verification = self._verify(task, history)
                    self.emit("verify", verification)
                    if verification["passed"]:
                        summary = str(decision.get("summary", "Task completed and verified."))
                        self.emit("completed", {"summary": summary, "execution_id": execution_id})
                        return ExecutionResult(execution_id, "completed", summary, step, recovery, verification["checks"])
                    raise RuntimeError("Model requested completion but deterministic verification failed.")

                if action != "tool" or str(decision.get("tool", "")).lower() != "shell":
                    raise ValueError("Only the shell tool is accepted by this terminal-first runtime.")
                args = decision.get("arguments") or {}
                command = str(args.get("command", "")).strip()
                timeout = int(args.get("timeout", 120))
                self.emit("tool", {"tool": "shell", "command": command})
                result = self.shell.run(command, timeout=timeout)
                record = {"step": step, "tool": "shell", "command": command, "ok": result.get("ok"), "stdout": result.get("stdout", ""), "stderr": result.get("stderr", ""), "returncode": result.get("returncode")}
                history.append(record)
                self.emit("observation", record)
                if not result.get("ok"):
                    raise RuntimeError(f"Command failed with exit code {result.get('returncode')}: {result.get('stderr', '')[-2000:]}")
            except (ModelRuntimeError, Exception) as exc:
                if recovery >= self.max_recovery:
                    self.emit("failed", {"error": str(exc), "execution_id": execution_id})
                    return ExecutionResult(execution_id, "failed", str(exc), step, recovery, history)
                recovery += 1
                self.emit("recovery", {"attempt": recovery, "error": str(exc)})
                history.append({"type": "failure", "error": str(exc), "recovery_attempt": recovery})

        message = f"Maximum agent steps ({self.max_steps}) reached without verified completion."
        self.emit("failed", {"error": message, "execution_id": execution_id})
        return ExecutionResult(execution_id, "failed", message, self.max_steps, recovery, history)
