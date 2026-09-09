from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable


class ToolError(RuntimeError):
    pass


class WorkspaceTools:
    """Deterministic tools exposed to the model. Paths are workspace-relative."""

    def __init__(self, workspace: Path, max_command_seconds: int = 600) -> None:
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.max_command_seconds = max(1, min(int(max_command_seconds), 600))

    def _path(self, value: str) -> Path:
        raw = Path(str(value or "."))
        target = (self.workspace / raw).resolve() if not raw.is_absolute() else raw.resolve()
        if target != self.workspace and self.workspace not in target.parents:
            raise ToolError("Path escapes the configured workspace.")
        return target

    def shell(self, command: str, timeout: int = 120) -> dict[str, Any]:
        if not str(command).strip():
            raise ToolError("Command cannot be empty.")
        seconds = max(1, min(int(timeout), self.max_command_seconds))
        completed = subprocess.run(["bash", "-lc", str(command)], cwd=self.workspace, text=True, capture_output=True, timeout=seconds)
        return {"ok": completed.returncode == 0, "returncode": completed.returncode, "stdout": completed.stdout[-16000:], "stderr": completed.stderr[-16000:], "cwd": str(self.workspace)}

    def read_file(self, path: str, max_chars: int = 30000) -> dict[str, Any]:
        target = self._path(path)
        if not target.is_file():
            raise ToolError(f"File not found: {path}")
        limit = max(1000, min(int(max_chars), 100000))
        return {"path": str(target.relative_to(self.workspace)), "content": target.read_text(encoding="utf-8")[:limit]}

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        target = self._path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")
        return {"ok": True, "path": str(target.relative_to(self.workspace)), "bytes": target.stat().st_size}

    def git_status(self) -> dict[str, Any]:
        return self.shell("git status --short --branch")

    def git_diff(self) -> dict[str, Any]:
        return self.shell("git diff --no-ext-diff -- .")

    def pytest(self, args: str = "-q") -> dict[str, Any]:
        return self.shell(f"python -m pytest {args}", timeout=min(300, self.max_command_seconds))


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {"type": "function", "function": {"name": "shell", "description": "Run a focused bash command in the workspace.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a UTF-8 text file relative to the workspace.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "max_chars": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Create or replace a UTF-8 text file inside the workspace.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "git_status", "description": "Show git branch and working-tree status.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "git_diff", "description": "Show the current unstaged git diff.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "pytest", "description": "Run the project's pytest suite.", "parameters": {"type": "object", "properties": {"args": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "finish", "description": "Request completion only after the task has been verified. The runtime performs deterministic verification before accepting this request.", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}, "verification": {"type": "string"}}, "required": ["summary", "verification"]}}},
]


def dispatch(tools: WorkspaceTools, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "finish":
        return {"ok": True, "finish_request": True, "summary": str(arguments.get("summary", "Task completed.")), "verification": str(arguments.get("verification", ""))}
    fn: Callable[..., dict[str, Any]] | None = getattr(tools, name, None)
    if fn is None or name.startswith("_"):
        raise ToolError(f"Unknown tool: {name}")
    try:
        return fn(**arguments)
    except subprocess.TimeoutExpired:
        return {"ok": False, "timeout": True, "error": "Tool execution timed out."}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def tool_result_message(tool_call_id: str, result: dict[str, Any], tool_name: str = "") -> dict[str, Any]:
    message: dict[str, Any] = {"role": "tool", "content": json.dumps(result, ensure_ascii=False, default=str)}
    if tool_name:
        message["tool_name"] = tool_name
    return message
