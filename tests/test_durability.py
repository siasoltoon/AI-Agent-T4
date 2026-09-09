from pathlib import Path

from agent_core.durability import DurableState


def test_state_save_is_reloadable_and_running_is_discoverable(tmp_path: Path):
    state = DurableState(tmp_path / ".agent_state", tmp_path, auto_git=False)
    state.save("exec_test_1", {"execution_id": "exec_test_1", "status": "running", "steps": 3, "updated_at": "2026-09-09T18:00:00+00:00"})

    loaded = state.load("exec_test_1")
    assert loaded["steps"] == 3
    assert [item["execution_id"] for item in state.list_running()] == ["exec_test_1"]


def test_completed_state_is_not_listed_as_running(tmp_path: Path):
    state = DurableState(tmp_path / ".agent_state", tmp_path, auto_git=False)
    state.save("exec_test_2", {"execution_id": "exec_test_2", "status": "completed", "steps": 4})

    assert state.list_running() == []


def test_interrupted_state_is_locally_resumable(tmp_path: Path):
    state = DurableState(tmp_path / ".agent_state", tmp_path, auto_git=False)
    state.save("exec_old", {"execution_id": "exec_old", "status": "interrupted", "steps": 2, "updated_at": "2026-09-09T18:00:00+00:00"})
    state.save("exec_new", {"execution_id": "exec_new", "status": "running", "steps": 1, "updated_at": "2026-09-09T18:01:00+00:00"})
    state.save("exec_done", {"execution_id": "exec_done", "status": "completed", "steps": 5, "updated_at": "2026-09-09T18:02:00+00:00"})

    resumable = state.list_resumable()
    assert [item["execution_id"] for item in resumable] == ["exec_new", "exec_old"]
