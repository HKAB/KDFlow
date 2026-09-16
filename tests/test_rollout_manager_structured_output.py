from types import SimpleNamespace

from kdflow.trainer.rollout_manager import RolloutManager


def _output(text, finish_reason="stop"):
    return {
        "text": text,
        "output_ids": [1, 2],
        "meta_info": {"finish_reason": finish_reason},
    }


def _manager(max_retries=2, retry_temperature=0.4):
    manager = RolloutManager.__new__(RolloutManager)
    manager.args = SimpleNamespace(
        rollout=SimpleNamespace(
            rollout_regex_max_retries=max_retries,
            rollout_regex_retry_temperature=retry_temperature,
        )
    )
    return manager


def test_retries_only_invalid_structured_outputs():
    manager = _manager()
    calls = []

    def generate(prompts, sampling_params, image_data=None):
        calls.append((prompts, sampling_params, image_data))
        return [_output("<o><s>b</s><v>beta</v></o>")], {}

    manager._generate = generate
    outputs, invalid, metrics = manager._retry_invalid_structured_outputs(
        ["prompt-a", "prompt-b"],
        [_output("<o></o>"), _output("<o><s>b</s><v>truncated", "length")],
        {
            "regex": r"<o>(?:<s>[^<]+</s><v>[^<]+</v>)*</o>",
            "temperature": 0.7,
        },
    )

    assert invalid == set()
    assert outputs[1]["text"] == "<o><s>b</s><v>beta</v></o>"
    assert calls[0][0] == ["prompt-b"]
    assert calls[0][1]["temperature"] == 0.4
    assert metrics["structured_output/retry_requests"] == 1.0
    assert metrics["structured_output/recovered_ratio"] == 0.5


def test_returns_persistent_invalid_indices_after_retry_budget():
    manager = _manager(max_retries=2)

    def generate(prompts, sampling_params, image_data=None):
        return [_output("<o><s>x</s><v>still truncated", "length")], {}

    manager._generate = generate
    _, invalid, metrics = manager._retry_invalid_structured_outputs(
        ["prompt"],
        [_output("bad", "length")],
        {"regex": r"<o></o>", "temperature": 0.7},
    )

    assert invalid == {0}
    assert metrics["structured_output/retry_requests"] == 2.0
    assert metrics["structured_output/recovered_ratio"] == 0.0
