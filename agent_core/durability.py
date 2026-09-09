from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class DurableState:
    """Crash-safe execution journal plus optional Git checkpointing."""

    def __init__(self, state_dir: Path, workspace: Path, auto_git: bool = True, remote: str = "origin", branch: str = "agent-checkpoints") -> None:
        self.state_dir = state_dir.resolve()
        self.workspace = workspace.resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.auto_git = auto_git
        self.remote = remote
        self.branch = branch

    def save(self, execution_id: str, payload: dict[str, Any]) -> None:
        target = self.state_dir / f"{execution_id}.json"
        fd, name = tempfile.mkstemp(prefix=f".{execution_id}.", suffix=".tmp", dir=self.state_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, target)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def load(self, execution_id: str) -> dict[str, Any]:
        path = self.state_dir / f"{execution_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Execution state not found: {execution_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_running(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for path in self.state_dir.glob("exec_*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("status") == "running":
                    items.append(data)
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(items, key=lambda x: x.get("updated_at", ""), reverse=True)

    def checkpoint_git(self, execution_id: str, reason: str) -> dict[str, Any]:
        if not self.auto_git:
            return {"ok": False, "skipped": True, "reason": "disabled"}
        def run(*args: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(list(args), cwd=self.workspace, text=True, capture_output=True, timeout=60)
        try:
            if run("git", "rev-parse", "--is-inside-work-tree").returncode != 0:
                return {"ok": False, "error": "workspace is not a git repository"}
            branch = run("git", "branch", "--show-current").stdout.strip()
            if not branch:
                return {"ok": False, "error": "detached HEAD; refusing automatic checkpoint"}
            add = run("git", "add", "-A")
            if add.returncode != 0:
                return {"ok": False, "error": add.stderr[-4000:]}
            diff = run("git", "diff", "--cached", "--quiet")
            if diff.returncode == 0:
                return {"ok": True, "changed": False, "branch": branch}
            commit = run("git", "commit", "-m", f"agent checkpoint: {execution_id} ({reason})")
            if commit.returncode != 0:
                return {"ok": False, "error": commit.stderr[-4000:]}
            push = run("git", "push", self.remote, f"HEAD:{self.branch}")
            if push.returncode != 0:
                return {"ok": False, "committed": True, "pushed": False, "error": push.stderr[-4000:]}
            return {"ok": True, "changed": True, "pushed": True, "branch": self.branch}
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"ok": False, "error": str(exc)}


class InterruptGuard:
    """Convert SIGINT/SIGTERM into a callback so state can be persisted first."""

    def __init__(self, callback):
        self.callback = callback
        self.previous: dict[int, Any] = {}

    def __enter__(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.previous[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle)
        return self

    def _handle(self, signum, _frame):
        self.callback(signum)

    def __exit__(self, *_args):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)
