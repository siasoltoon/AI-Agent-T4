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


def test_finish_without_execution_evidence_is_rejected(tmp_path: Path):
    init_git(tmp_path)
    model = FakeModel([tool_call("finish", {"summary": "done", "verification": "trust me"})])
    result = AgentExecutor(model, tmp_path, max_steps=1, max_recovery=0, auto_git_checkpoint=False).run("Do the task")
    assert result.status == "failed"
    assert "verification" in result.summary.lower()


def test_normal_finish_after_deterministic_tool_succeeds(tmp_path: Path):
    init_git(tmp_path)
    model = FakeModel([
        tool_call("write_file", {"path": "verified.txt", "content": "ok"}),
        tool_call("finish", {"summary": "verified", "verification": "write_file succeeded"}, "call-2"),
    ])
    result = AgentExecutor(model, tmp_path, max_steps=2, max_recovery=0, auto_git_checkpoint=False).run("Create verified.txt")
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


def test_model_without_tool_call_hits_failure_budget_cleanly(tmp_path: Path):
    init_git(tmp_path)
    model = FakeModel([{"message": {"role": "assistant", "content": "I am finished."}}])
    result = AgentExecutor(model, tmp_path, max_steps=1, max_recovery=0, auto_git_checkpoint=False).run("Create a file")
    assert result.status == "failed"
    assert "tool call" in result.summary.lower()


def test_tool_output_is_bounded(tmp_path: Path):
    tools = WorkspaceTools(tmp_path)
    result = tools.shell("python -c \"print('x' * 20000)\"")
    assert result["ok"] is True
    assert len(result["stdout"]) <= 16000


def test_model_context_is_bounded_and_preserves_system_and_task(tmp_path: Path):
    init_git(tmp_path)
    executor = AgentExecutor(FakeModel([]), tmp_path, auto_git_checkpoint=False, max_context_chars=4000)
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "TASK"},
    ]
    for index in range(20):
        messages.append({"role": "assistant", "content": f"assistant-{index}"})
        messages.append({"role": "tool", "tool_call_id": f"call-{index}", "content": "x" * 500})
    bounded = executor._model_messages(messages)
    assert bounded[0]["content"] == "SYSTEM"
    assert bounded[1]["content"] == "TASK"
    assert len(__import__("json").dumps(bounded, ensure_ascii=False, separators=(",", ":"))) <= 4000
    assert all(item.get("role") != "tool" or index > 1 for index, item in enumerate(bounded))
