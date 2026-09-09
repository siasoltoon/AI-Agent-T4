from pathlib import Path

from agent_core.tools import WorkspaceTools, dispatch


def test_workspace_paths_are_confined(tmp_path: Path):
    tools = WorkspaceTools(tmp_path)
    assert tools.write_file("ok.txt", "hello")["ok"]
    assert tools.read_file("ok.txt")["content"] == "hello"
    outside = tmp_path.parent / "outside.txt"
    result = dispatch(tools, "read_file", {"path": "../outside.txt"})
    assert result["ok"] is False


def test_shell_runs_in_workspace(tmp_path: Path):
    tools = WorkspaceTools(tmp_path)
    result = tools.shell("pwd")
    assert result["ok"] is True
    assert str(tmp_path.resolve()) in result["stdout"]


def test_finish_is_runtime_signal(tmp_path: Path):
    tools = WorkspaceTools(tmp_path)
    result = dispatch(tools, "finish", {"summary": "done", "verification": "tests passed"})
    assert result["finish_request"] is True
