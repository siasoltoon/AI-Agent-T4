from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
from datetime import datetime, timezone
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

    def _list_by_status(self, statuses: set[str]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for path in self.state_dir.glob("exec_*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("status") in statuses:
                    data.setdefault("updated_at", datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat())
                    items.append(data)
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(items, key=lambda x: x.get("updated_at", ""), reverse=True)

    def list_running(self) -> list[dict[str, Any]]:
        return self._list_by_status({"running"})

    def list_resumable(self) -> list[dict[str, Any]]:
        """Return local executions that can safely be resumed, newest first."""
        return self._list_by_status({"running", "interrupted"})

    def _git(self, *args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], cwd=self.workspace, text=True, capture_output=True, timeout=timeout)

    def _ensure_git_identity(self) -> dict[str, Any]:
        """Ensure fresh/ephemeral runtimes can create checkpoint commits."""
        name = self._git("config", "--get", "user.name")
        email = self._git("config", "--get", "user.email")
        if name.returncode == 0 and name.stdout.strip() and email.returncode == 0 and email.stdout.strip():
            return {"ok": True, "changed": False}
        configured: list[str] = []
        if name.returncode != 0 or not name.stdout.strip():
            result = self._git("config", "user.name", "AI Agent")
            if result.returncode != 0:
                return {"ok": False, "error": result.stderr[-4000:] or "could not configure Git user.name"}
            configured.append("user.name")
        if email.returncode != 0 or not email.stdout.strip():
            result = self._git("config", "user.email", "ai-agent@localhost")
            if result.returncode != 0:
                return {"ok": False, "error": result.stderr[-4000:] or "could not configure Git user.email"}
            configured.append("user.email")
        return {"ok": True, "changed": bool(configured), "configured": configured}

    def checkpoint_git(self, execution_id: str, reason: str) -> dict[str, Any]:
        if not self.auto_git:
            return {"ok": False, "skipped": True, "reason": "disabled"}
        try:
            if self._git("rev-parse", "--is-inside-work-tree").returncode != 0:
                return {"ok": False, "error": "workspace is not a git repository"}
            identity = self._ensure_git_identity()
            if not identity.get("ok"):
                return {"ok": False, "error": identity.get("error", "Git identity setup failed")}
            branch = self._git("branch", "--show-current").stdout.strip()
            if not branch:
                return {"ok": False, "error": "detached HEAD; refusing automatic checkpoint"}
            add = self._git("add", "-A")
            if add.returncode != 0:
                return {"ok": False, "error": add.stderr[-4000:]}
            state_path = self.state_dir / f"{execution_id}.json"
            if state_path.exists() and self.workspace in state_path.parents:
                force = self._git("add", "-f", str(state_path.relative_to(self.workspace)))
                if force.returncode != 0:
                    return {"ok": False, "error": force.stderr[-4000:]}
            diff = self._git("diff", "--cached", "--quiet")
            if diff.returncode == 0:
                return {"ok": True, "changed": False, "branch": branch, "identity_configured": identity.get("changed", False)}
            commit = self._git("commit", "-m", f"agent checkpoint: {execution_id} ({reason})", timeout=120)
            if commit.returncode != 0:
                return {"ok": False, "error": commit.stderr[-4000:]}
            push = self._git("push", self.remote, f"HEAD:{self.branch}", timeout=120)
            if push.returncode != 0:
                return {"ok": False, "committed": True, "pushed": False, "error": push.stderr[-4000:]}
            return {"ok": True, "changed": True, "pushed": True, "branch": self.branch, "identity_configured": identity.get("changed", False)}
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"ok": False, "error": str(exc)}

    def restore_from_remote(self, execution_id: str | None = None) -> dict[str, Any]:
        """Restore the workspace and state from a clean remote checkpoint."""
        try:
            if self._git("rev-parse", "--is-inside-work-tree").returncode != 0:
                return {"ok": False, "error": "workspace is not a git repository"}
            if self._git("status", "--porcelain").stdout.strip():
                return {"ok": False, "error": "workspace is not clean; refusing destructive checkpoint restore"}
            fetch = self._git("fetch", self.remote, self.branch, timeout=120)
            if fetch.returncode != 0:
                return {"ok": False, "error": fetch.stderr[-4000:] or "git fetch failed"}
            remote_ref = f"{self.remote}/{self.branch}"
            if self._git("rev-parse", "--verify", remote_ref).returncode != 0:
                return {"ok": False, "error": f"checkpoint branch not found: {remote_ref}"}

            if execution_id:
                state_path = f".agent_state/{execution_id}.json"
                commit = self._git("log", "-1", "--format=%H", remote_ref, "--", state_path).stdout.strip()
                if not commit:
                    return {"ok": False, "error": f"remote execution not found: {execution_id}"}
            else:
                commit = self._git("log", "-1", "--format=%H", remote_ref, "--", ".agent_state").stdout.strip()
                if not commit:
                    return {"ok": False, "error": "no remote execution state found"}
                names = self._git("ls-tree", "-r", "--name-only", commit, ".agent_state").stdout.splitlines()
                state_names = sorted(n for n in names if n.startswith(".agent_state/exec_") and n.endswith(".json"))
                if not state_names:
                    return {"ok": False, "error": "remote checkpoint contains no execution state"}
                execution_id = Path(state_names[-1]).stem

            reset = self._git("reset", "--hard", commit, timeout=120)
            if reset.returncode != 0:
                return {"ok": False, "error": reset.stderr[-4000:] or "git reset failed"}
            return {"ok": True, "execution_id": execution_id, "commit": commit, "branch": self.branch}
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"ok": False, "error": str(exc)}

    def restore_latest_resumable(self) -> dict[str, Any]:
        """Restore the newest running/interrupted remote execution, if one exists."""
        try:
            if self._git("rev-parse", "--is-inside-work-tree").returncode != 0:
                return {"ok": False, "found": False, "error": "workspace is not a git repository"}
            if self._git("status", "--porcelain").stdout.strip():
                return {"ok": False, "found": False, "error": "workspace is not clean; refusing destructive checkpoint restore"}

            remote_branch = self._git("ls-remote", "--exit-code", "--heads", self.remote, self.branch, timeout=60)
            if remote_branch.returncode == 2:
                return {"ok": True, "found": False}
            if remote_branch.returncode != 0:
                return {"ok": False, "found": False, "error": remote_branch.stderr[-4000:] or "git ls-remote failed"}

            fetch = self._git("fetch", self.remote, self.branch, timeout=120)
            if fetch.returncode != 0:
                return {"ok": False, "found": False, "error": fetch.stderr[-4000:] or "git fetch failed"}
            remote_ref = f"{self.remote}/{self.branch}"
            if self._git("rev-parse", "--verify", remote_ref).returncode != 0:
                return {"ok": True, "found": False}

            names = self._git("ls-tree", "-r", "--name-only", remote_ref, ".agent_state").stdout.splitlines()
            candidates: list[tuple[str, str, str]] = []
            for name in names:
                if not name.startswith(".agent_state/exec_") or not name.endswith(".json"):
                    continue
                execution_id = Path(name).stem
                raw = self._git("show", f"{remote_ref}:{name}").stdout
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if data.get("status") not in {"running", "interrupted"}:
                    continue
                updated_at = str(data.get("updated_at", ""))
                commit = self._git("log", "-1", "--format=%H", remote_ref, "--", name).stdout.strip()
                if commit:
                    candidates.append((updated_at, execution_id, commit))

            if not candidates:
                return {"ok": True, "found": False}
            _, execution_id, commit = max(candidates, key=lambda item: item[0])
            reset = self._git("reset", "--hard", commit, timeout=120)
            if reset.returncode != 0:
                return {"ok": False, "found": True, "error": reset.stderr[-4000:] or "git reset failed"}
            return {"ok": True, "found": True, "execution_id": execution_id, "commit": commit, "branch": self.branch}
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"ok": False, "found": False, "error": str(exc)}
