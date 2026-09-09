from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent_core.durability import DurableState, InterruptGuard
from agent_core.tools import TOOL_SCHEMAS, WorkspaceTools, dispatch, tool_result_message

SYSTEM_PROMPT = """You are a production-grade autonomous software engineering agent.
Operate by inspect -> reason -> act -> observe -> verify -> recover. Never claim completion from intention.
Use deterministic tools instead of inventing command output or file contents.
Prefer targeted inspection, minimal edits, real tests, and git diff/status checks. Never modify tests unless explicitly requested.
Use the finish tool only after the strongest relevant verification has passed. If a tool fails, diagnose the evidence and continue within the budget.
All file paths must remain inside the configured workspace.
If the runtime reports a prior interrupted execution, continue from the supplied state instead of restarting completed work.
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
    def __init__(self, model: Any, workspace: Path, max_steps: int = 32, max_recovery: int = 6, max_command_seconds: int = 600, temperature: float = 0.1, emit: Callable[[str, dict[str, Any]], None] | None = None, state_dir: Path | None = None, auto_git_checkpoint: bool = True, git_remote: str = "origin", checkpoint_branch: str = "agent-checkpoints", checkpoint_every_tool: bool = True) -> None:
        self.model = model
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.tools = WorkspaceTools(self.workspace, max_command_seconds=max_command_seconds)
        self.max_steps = max(1, min(int(max_steps), 128))
        self.max_recovery = max(0, min(int(max_recovery), 12))
        self.temperature = max(0.0, min(float(temperature), 1.0))
        self.emit = emit or (lambda _event, _data: None)
        self.state = DurableState(state_dir or (self.workspace / ".agent_state"), self.workspace, auto_git_checkpoint, git_remote, checkpoint_branch)
        self.checkpoint_every_tool = checkpoint_every_tool
        self._interrupted = False

    def _snapshot(self, execution_id: str, status: str, task: str, messages: list[dict[str, Any]], history: list[dict[str, Any]], step: int, recovery: int, summary: str = "") -> None:
        self.state.save(execution_id, {"execution_id": execution_id, "status": status, "task": task, "messages": messages, "history": history[-100:], "steps": step, "recovery_attempts": recovery, "summary": summary, "workspace": str(self.workspace), "updated_at": datetime.now(timezone.utc).isoformat()})

    def _verify(self, task: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        status = self.tools.git_status()
        checks.append({"type": "workspace", "passed": status.get("ok") is True, "stdout": status.get("stdout", ""), "stderr": status.get("stderr", "")})
        actionable = [h for h in history if h.get("tool") not in {None, "finish"}]
        checks.append({"type": "execution_evidence", "passed": bool(actionable), "detail": "At least one deterministic tool must run before finish."})
        lower = task.lower()
        if any(x in lower for x in ("test", "tests", "pytest", "test suite")) or any(h.get("tool") == "pytest" for h in history):
            result = self.tools.pytest()
            checks.append({"type": "pytest", "passed": result.get("ok") is True, "stdout": result.get("stdout", ""), "stderr": result.get("stderr", "")})
        return {"passed": all(c["passed"] for c in checks), "checks": checks}

    def _finish(self, execution_id: str, task: str, history: list[dict[str, Any]], messages: list[dict[str, Any]], step: int, recovery: int, summary: str) -> ExecutionResult | None:
        verification = self._verify(task, history)
        self.emit("verify", verification)
        if not verification["passed"]:
            return None
        result = ExecutionResult(execution_id, "completed", summary, step, recovery, verification["checks"])
        self.state.save(execution_id, {**result.__dict__, "task": task, "messages": messages, "history": history, "workspace": str(self.workspace), "updated_at": datetime.now(timezone.utc).isoformat()})
        checkpoint = self.state.checkpoint_git(execution_id, "completed")
        if self.state.auto_git and not checkpoint.get("ok"):
            self.state.save(execution_id, {**result.__dict__, "task": task, "messages": messages, "history": history, "workspace": str(self.workspace), "updated_at": datetime.now(timezone.utc).isoformat(), "checkpoint_error": checkpoint.get("error", "checkpoint failed")})
            self.emit("checkpoint_warning", checkpoint)
        self.emit("completed", {"summary": summary, "execution_id": execution_id})
        return result

    def _interrupt(self, execution_id: str, task: str, messages: list[dict[str, Any]], history: list[dict[str, Any]], step: int, recovery: int, signum: int) -> None:
        self._interrupted = True
        self._snapshot(execution_id, "interrupted", task, messages, history, step, recovery, "Execution interrupted; resume with `agent resume %s`." % execution_id)
        self.tools.interrupt_active_process()
        checkpoint = self.state.checkpoint_git(execution_id, "interrupted")
        if self.state.auto_git and not checkpoint.get("ok"):
            self.emit("checkpoint_warning", checkpoint)
        self.emit("interrupted", {"execution_id": execution_id, "signal": signum})

    def _run_loop(self, execution_id: str, task: str, messages: list[dict[str, Any]], history: list[dict[str, Any]], start_step: int, recovery: int) -> ExecutionResult:
        current_step = start_step

        def on_interrupt(signum: int) -> None:
            self._interrupt(execution_id, task, messages, history, current_step, recovery, signum)

        with InterruptGuard(on_interrupt):
            for step in range(start_step + 1, self.max_steps + 1):
                current_step = step
                self.emit("step", {"number": step, "max": self.max_steps})
                try:
                    if self._interrupted:
                        break
                    response = self.model.chat(messages, tools=TOOL_SCHEMAS, temperature=self.temperature)
                    if not isinstance(response, dict):
                        raise RuntimeError("Model returned an invalid response object.")
                    message = response.get("message") or {}
                    if not isinstance(message, dict):
                        raise RuntimeError("Model returned an invalid message object.")
                    tool_calls = message.get("tool_calls") or []
                    messages.append(message)
                    if not tool_calls:
                        raise RuntimeError("Model stopped without a tool call; completion requires the finish tool.")
                    for call in tool_calls:
                        if not isinstance(call, dict):
                            raise RuntimeError("Model returned an invalid tool call.")
                        fn = call.get("function") or {}
                        name = str(fn.get("name") or "")
                        arguments = fn.get("arguments") or {}
                        if isinstance(arguments, str):
                            arguments = json.loads(arguments)
                        if not isinstance(arguments, dict):
                            raise RuntimeError(f"Tool arguments for {name or 'unknown'} must be an object.")
                        self.emit("tool", {"tool": name, "command": arguments.get("command", name)})
                        record = {"step": step, "tool": name, "arguments": arguments, "status": "started"}
                        history.append(record)
                        self._snapshot(execution_id, "running", task, messages, history, step, recovery)
                        result = dispatch(self.tools, name, arguments)
                        record.update({"ok": result.get("ok", True), "result": result, "status": "completed"})
                        if self._interrupted:
                            break
                        self.emit("observation", {"tool": name, "stdout": result.get("stdout", ""), "stderr": result.get("stderr", "")})
                        if name == "finish" and result.get("finish_request"):
                            completed = self._finish(execution_id, task, history, messages, step, recovery, result.get("summary", "Task completed and verified."))
                            if completed is not None:
                                return completed
                            raise RuntimeError("Finish request rejected because deterministic verification failed.")
                        messages.append(tool_result_message(str(call.get("id") or uuid.uuid4().hex), result, name))
                        self._snapshot(execution_id, "running", task, messages, history, step, recovery)
                        if self.checkpoint_every_tool:
                            checkpoint = self.state.checkpoint_git(execution_id, f"step-{step}-{name}")
                            if not checkpoint.get("ok") and not checkpoint.get("skipped"):
                                self.emit("checkpoint_warning", checkpoint)
                        if result.get("ok") is False:
                            raise RuntimeError(f"Tool {name} failed: {result.get('stderr') or result.get('error') or 'unknown error'}")
                except Exception as exc:
                    if self._interrupted:
                        break
                    if recovery >= self.max_recovery:
                        result = ExecutionResult(execution_id, "failed", str(exc), step, recovery, history)
                        self.state.save(execution_id, {**result.__dict__, "task": task, "messages": messages, "history": history, "workspace": str(self.workspace), "updated_at": datetime.now(timezone.utc).isoformat()})
                        self.state.checkpoint_git(execution_id, "failed")
                        self.emit("failed", {"error": str(exc), "execution_id": execution_id})
                        return result
                    recovery += 1
                    self.emit("recovery", {"attempt": recovery, "error": str(exc)})
                    history.append({"type": "failure", "error": str(exc), "recovery_attempt": recovery})
                    messages.append({"role": "user", "content": f"Runtime failure: {exc}. Inspect the current state and continue; do not claim completion yet."})
                    self._snapshot(execution_id, "running", task, messages, history, step, recovery)
        if self._interrupted:
            return ExecutionResult(execution_id, "interrupted", "Execution interrupted and durably checkpointed.", current_step, recovery, history)
        message = f"Maximum agent steps ({self.max_steps}) reached without verified completion."
        result = ExecutionResult(execution_id, "failed", message, self.max_steps, recovery, history)
        self.state.save(execution_id, {**result.__dict__, "task": task, "messages": messages, "history": history, "workspace": str(self.workspace), "updated_at": datetime.now(timezone.utc).isoformat()})
        self.state.checkpoint_git(execution_id, "max-steps")
        self.emit("failed", {"error": message, "execution_id": execution_id})
        return result

    def run(self, task: str) -> ExecutionResult:
        execution_id = f"exec_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Workspace: {self.workspace}\nTask: {task}"}]
        self.emit("start", {"execution_id": execution_id, "task": task})
        self.emit("planning", {"message": "Planning and executing autonomously..."})
        self._snapshot(execution_id, "running", task, messages, [], 0, 0)
        return self._run_loop(execution_id, task, messages, [], 0, 0)

    def resume(self, execution_id: str) -> ExecutionResult:
        data = self.state.load(execution_id)
        if data.get("status") == "completed":
            return ExecutionResult(execution_id, "completed", str(data.get("summary", "Already completed.")), int(data.get("steps", 0)), int(data.get("recovery_attempts", 0)), list(data.get("evidence", [])))
        self._interrupted = False
        return self._run_loop(execution_id, str(data["task"]), list(data.get("messages") or []), list(data.get("history") or []), int(data.get("steps", 0)), int(data.get("recovery_attempts", 0)))
