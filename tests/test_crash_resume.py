from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from agent_core.durability import DurableState
from agent_core.executor import AgentExecutor


class FinishModel:
    def chat(self, messages, *, tools=None, temperature=0.1):
        return {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "resume-finish",
                        "function": {
                            "name": "finish",
                            "arguments": {
                                "summary": "Resumed from durable checkpoint.",
                                "verification": "deterministic verification passed",
                            },
                        },
                    }
                ],
            }
        }


def git(path: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=path, text=True, capture_output=True, check=check)


def init_repo(path: Path) -> Path:
    git(path, "init")
    git(path, "config", "user.email", "agent-test@example.invalid")
    git(path, "config", "user.name", "Agent Test")
    (path / "README.md").write_text("crash-resume test\n", encoding="utf-8")
    git(path, "add", "README.md")
    git(path, "commit", "-m", "test fixture")
    return path


def test_sigterm_checkpoint_survives_fresh_clone_and_resume(tmp_path: Path):
    workspace = init_repo(tmp_path / "workspace")
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "--bare", str(remote))
    git(workspace, "remote", "add", "origin", str(remote))
    git(workspace, "push", "origin", "HEAD:main")

    script = textwrap.dedent(
        """
        from pathlib import Path
        from agent_core.executor import AgentExecutor

        class SlowModel:
            def chat(self, messages, *, tools=None, temperature=0.1):
                return {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "slow-call",
                            "function": {"name": "shell", "arguments": {"command": "sleep 30"}},
                        }],
                    }
                }

        workspace = Path(__import__("sys").argv[1])
        remote = __import__("sys").argv[2]
        executor = AgentExecutor(
            SlowModel(),
            workspace,
            max_steps=5,
            max_recovery=0,
            max_command_seconds=60,
            state_dir=workspace / ".agent_state",
            auto_git_checkpoint=True,
            git_remote=remote,
            checkpoint_branch="agent-checkpoints",
        )
        result = executor.run("survive an interruption")
        print(result.status, result.execution_id, flush=True)
        """
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(workspace), "origin"],
        cwd=workspace,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    state_file = None
    deadline = time.time() + 15
    while time.time() < deadline:
        candidates = list((workspace / ".agent_state").glob("exec_*.json"))
        if candidates:
            state_file = candidates[0]
            data = state_file.read_text(encoding="utf-8")
            if '"steps": 0' in data:
                break
        time.sleep(0.1)
    assert state_file is not None, "execution state was not created"

    child.send_signal(signal.SIGTERM)
    stdout, stderr = child.communicate(timeout=20)
    assert child.returncode == 0, f"child failed: stdout={stdout!r} stderr={stderr!r}"

    interrupted = __import__("json").loads(state_file.read_text(encoding="utf-8"))
    execution_id = interrupted["execution_id"]
    assert interrupted["status"] == "interrupted"
    assert interrupted["steps"] == 1

    remote_ref = git(workspace, "ls-remote", "origin", "refs/heads/agent-checkpoints").stdout.strip()
    assert remote_ref, "interrupted checkpoint was not pushed to the remote"

    fresh = tmp_path / "fresh"
    git(tmp_path, "clone", str(remote), str(fresh))
    git(fresh, "checkout", "-B", "main", "origin/main")

    restored = DurableState(
        fresh / ".agent_state",
        fresh,
        auto_git=True,
        remote="origin",
        branch="agent-checkpoints",
    ).restore_from_remote(execution_id)
    assert restored["ok"] is True
    assert restored["execution_id"] == execution_id
    restored_state = DurableState(fresh / ".agent_state", fresh, auto_git=False).load(execution_id)
    assert restored_state["status"] == "interrupted"
    assert restored_state["steps"] == 1

    result = AgentExecutor(
        FinishModel(),
        fresh,
        max_steps=3,
        max_recovery=0,
        state_dir=fresh / ".agent_state",
        auto_git_checkpoint=False,
    ).resume(execution_id)
    assert result.status == "completed"
    assert result.execution_id == execution_id


# Keep the fixture self-contained on platforms where the bare remote clone helper is unavailable.
assert shutil.which("git") is not None
