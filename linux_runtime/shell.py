from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


class ShellExecutionError(RuntimeError):
    pass


class LinuxShell:
    def __init__(self, workspace: Path, max_seconds: int = 600) -> None:
        self.workspace = workspace.resolve()
        self.max_seconds = max_seconds

    def _safe_cwd(self, cwd: str | None) -> Path:
        target = (self.workspace / cwd).resolve() if cwd and not Path(cwd).is_absolute() else Path(cwd or self.workspace).resolve()
        if target != self.workspace and self.workspace not in target.parents:
            raise ShellExecutionError("Working directory escapes the configured workspace.")
        target.mkdir(parents=True, exist_ok=True)
        return target

    def run(self, command: str, timeout: int = 120, cwd: str | None = None) -> dict[str, Any]:
        command = str(command).strip()
        if not command:
            raise ShellExecutionError("Command cannot be empty.")
        timeout = max(1, min(int(timeout), self.max_seconds))
        workdir = self._safe_cwd(cwd)
        try:
            completed = subprocess.run(command, shell=True, cwd=workdir, text=True, capture_output=True, timeout=timeout)
            return {
                "ok": completed.returncode == 0,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-12000:],
                "stderr": completed.stderr[-12000:],
                "cwd": str(workdir),
                "command": command,
            }
        except subprocess.TimeoutExpired as exc:
            return {"ok": False, "timeout": True, "returncode": None, "stdout": str(exc.stdout or "")[-12000:], "stderr": str(exc.stderr or "")[-12000:], "cwd": str(workdir), "command": command}
