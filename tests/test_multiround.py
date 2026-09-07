from types import SimpleNamespace

from synthesis.counterfactual import _rollout


class SequenceGenerator:
    name = "fake"
    revision = "pinned"

    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def generate(self, prompt, seed):
        return next(self.outputs)


class Environment:
    def __init__(self):
        self.problem = SimpleNamespace(
            public_problem=SimpleNamespace(io_mode="stdin")
        )
        self.current_candidate = "bad"

    def call(self, name, arguments):
        if name == "run_candidate":
            return {
                "status": "ok", "stdout": "bad\n", "stderr": "",
                "error": None, "exit_code": 0, "truncated": False,
            }
        self.current_candidate = arguments["code"]
        if arguments["code"] == "good":
            return {
                "status": "accepted", "passed": 1, "total": 1,
                "pass_rate": 1.0, "failing_input": None,
            }
        return {
            "status": "wrong_answer", "passed": 0, "total": 1,
            "pass_rate": 0.0, "failing_input": "x",
        }


def test_multiround_masks_failed_submit_and_supervises_recovery():
    generator = SequenceGenerator([
        '{"name":"submit","arguments":{"code":"bad"}}',
        '{"name":"run_candidate","arguments":{"input":"x"}}',
        '{"name":"submit","arguments":{"code":"good"}}',
    ])
    result = _rollout(
        generator,
        [{"role": "user", "content": "state", "trainable": False}],
        11,
        Environment(),
        allow_run=True,
        limits=(6, 3, 3),
    )
    assert result.accepted
    assistants = [
        message for message in result.messages
        if message["role"] == "assistant"
    ]
    assert [message["trainable"] for message in assistants] == [
        False, True, True,
    ]
    assert assistants[0]["tool_calls"][0]["name"] == "submit"
    assert assistants[1]["tool_calls"][0]["name"] == "run_candidate"


def test_no_run_counterfactual_rejects_run_action():
    generator = SequenceGenerator([
        '{"name":"run_candidate","arguments":{"input":"x"}}',
    ])
    result = _rollout(
        generator,
        [{"role": "user", "content": "state", "trainable": False}],
        11,
        Environment(),
        allow_run=False,
    )
    assert not result.accepted
    assert result.reject_reason == "run_not_allowed_or_budget_exhausted"
