from pathlib import Path

from agent_core.durability import DurableState
from agent_core.executor import AgentExecutor
from agent_core.tools import ToolError, WorkspaceTools


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    def chat(self, messages, *, tools=None, temperature=0.1):
        return next(self.responses)


def tool_call(name: str, arguments: dict, call_id: str = "call-1"):
    return {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": call_id, "function": {"name": name, "arguments": arguments}}],
        }
    }


def init_git(path: Path) -> None:
    import subprocess

    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)


def test_finish_requires_deterministic_verification(tmp_path: Path):
    init_git(tmp_path)
    model = FakeModel([
        tool_call("finish", {"summary": "done", "verification": "trust me"}),
        tool_call("write_file", {"path": "verified.txt", "content": "ok"}, "call-2"),
        tool_call("finish", {"summary": "verified", "verification": "file exists"}, "call-3"),
    ])
    result = AgentExecutor(model, tmp_path, max_steps=3, max_recovery=1, auto_git_checkpoint=False).run("Create verified.txt")
    assert result.status == "completed"
    assert (tmp_path / "verified.txt").read_text() == "ok"


def test_workspace_paths_cannot_escape(tmp_path: Path):
    tools = WorkspaceTools(tmp_path)
    for path in ("../outside.txt", str(tmp_path.parent / "outside.txt")):
        try:
            tools.read_file(path)
        except ToolError as exc:
            assert "escapes" in str(exc)
        else:
            raise AssertionError("path escape was not rejected")


def test_resumable_returns_running_and_interrupted_newest_first(tmp_path: Path):
    state = DurableState(tmp_path / ".agent_state", tmp_path, auto_git=False)
    state.save("exec_old", {"execution_id": "exec_old", "status": "running", "updated_at": "2026-01-01T00:00:00+00:00"})
    state.save("exec_new", {"execution_id": "exec_new", "status": "interrupted", "updated_at": "2026-01-02T00:00:00+00:00"})
    state.save("exec_done", {"execution_id": "exec_done", "status": "completed", "updated_at": "2026-01-03T00:00:00+00:00"})
    assert [item["execution_id"] for item in state.list_resumable()] == ["exec_new", "exec_old"]


def test_model_without_tool_call_is_recoverable(tmp_path: Path):
    init_git(tmp_path)
    model = FakeModel([
        {"message": {"role": "assistant", "content": "I am finished."}},
        tool_call("write_file", {"path": "result.txt", "content": "ok"}, "call-2"),
        tool_call("finish", {"summary": "finished", "verification": "result.txt was written"}, "call-3"),
    ])
    result = AgentExecutor(model, tmp_path, max_steps=3, max_recovery=1, auto_git_checkpoint=False).run("Create result.txt")
    assert result.status == "completed"
    assert result.recovery_attempts == 1


def test_tool_output_is_bounded(tmp_path: Path):
    tools = WorkspaceTools(tmp_path)
    result = tools.shell("python -c \"print('x' * 20000)\"")
    assert result["ok"] is True
    assert len(result["stdout"]) <= 16000
