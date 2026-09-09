from pathlib import Path

from agent_core.executor import AgentExecutor


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    def generate(self, prompt, *, system="", temperature=0.1):
        return next(self.responses)


def test_executor_runs_shell_and_requires_finish(tmp_path: Path):
    model = FakeModel([
        '{"action":"tool","tool":"shell","arguments":{"command":"printf hello > result.txt"}}',
        '{"action":"tool","tool":"shell","arguments":{"command":"cat result.txt"}}',
        '{"action":"finish","summary":"Created and inspected the file.","verification":"cat returned hello"}',
    ])
    result = AgentExecutor(model, tmp_path, max_steps=5, max_recovery=0).run("Create result.txt and verify it")
    assert result.status == "completed"
    assert (tmp_path / "result.txt").read_text() == "hello"


def test_executor_bounds_steps(tmp_path: Path):
    model = FakeModel(['{"action":"tool","tool":"shell","arguments":{"command":"true"}}'] * 10)
    result = AgentExecutor(model, tmp_path, max_steps=2, max_recovery=0).run("keep working")
    assert result.status == "failed"
    assert result.steps == 2
