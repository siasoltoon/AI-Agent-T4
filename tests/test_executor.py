from pathlib import Path

from agent_core.executor import AgentExecutor


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


def test_executor_runs_tool_and_requires_verified_finish(tmp_path: Path):
    init_git(tmp_path)
    model = FakeModel([
        tool_call("write_file", {"path": "result.txt", "content": "hello"}),
        tool_call("read_file", {"path": "result.txt"}, "call-2"),
        tool_call("finish", {"summary": "Created and inspected the file.", "verification": "read_file returned hello"}, "call-3"),
    ])
    result = AgentExecutor(model, tmp_path, max_steps=5, max_recovery=0, auto_git_checkpoint=False).run("Create result.txt and verify it")
    assert result.status == "completed"
    assert (tmp_path / "result.txt").read_text() == "hello"


def test_executor_retries_missing_tool_call_without_spending_recovery(tmp_path: Path):
    init_git(tmp_path)
    prose_response = {
        "message": {
            "role": "assistant",
            "content": "I will inspect the repository first.",
        }
    }
    model = FakeModel([
        prose_response,
        tool_call("write_file", {"path": "result.txt", "content": "hello"}),
        tool_call("read_file", {"path": "result.txt"}, "call-2"),
        tool_call("finish", {"summary": "Created and inspected the file.", "verification": "read_file returned hello"}, "call-3"),
    ])
    result = AgentExecutor(
        model,
        tmp_path,
        max_steps=5,
        max_recovery=0,
        max_model_retries=2,
        auto_git_checkpoint=False,
    ).run("Create result.txt and verify it")
    assert result.status == "completed"
    assert result.recovery_attempts == 0
    assert (tmp_path / "result.txt").read_text() == "hello"


def test_executor_recovers_from_model_exception_before_tool_execution(tmp_path: Path):
    init_git(tmp_path)

    class FlakyModel:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, *, tools=None, temperature=0.1):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("temporary model timeout")
            return tool_call("finish", {"summary": "Verified no-op task."}, "call-finish")

    result = AgentExecutor(
        FlakyModel(),
        tmp_path,
        max_steps=2,
        max_recovery=0,
        max_model_retries=2,
        auto_git_checkpoint=False,
    ).run("Verify the current workspace")
    assert result.status == "completed"
    assert result.recovery_attempts == 0


def test_executor_bounds_steps(tmp_path: Path):
    init_git(tmp_path)
    model = FakeModel([tool_call("shell", {"command": "true"}, f"call-{i}") for i in range(10)])
    result = AgentExecutor(model, tmp_path, max_steps=2, max_recovery=0, auto_git_checkpoint=False).run("keep working")
    assert result.status == "failed"
    assert result.steps == 2
