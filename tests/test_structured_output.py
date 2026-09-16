from pathlib import Path

import pytest

from kdflow.utils.structured_output import (
    find_invalid_rollout_regex_outputs,
    load_rollout_regex,
    validate_rollout_regex_outputs,
)


def test_loads_rollout_regex_from_environment(tmp_path, monkeypatch):
    regex_path = tmp_path / "rollout.regex"
    regex_path.write_text("<result>.*</result>\n", encoding="utf-8")
    monkeypatch.setenv("KDFLOW_ROLLOUT_REGEX_FILE", str(regex_path))

    assert load_rollout_regex() == r"<result>.*</result>"


def test_rejects_empty_rollout_regex(tmp_path):
    regex_path = tmp_path / "empty.regex"
    regex_path.write_text(" \n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_rollout_regex(str(regex_path))


def test_accepts_outputs_matching_rollout_regex():
    validate_rollout_regex_outputs(
        ["<result></result>", "<result>foreign</result>"],
        r"<result>.*</result>",
    )


def test_rejects_unconstrained_thinking_output():
    with pytest.raises(RuntimeError, match="violates"):
        validate_rollout_regex_outputs(
            ["thought\nThinking Process\n<result></result>"],
            r"<result>.*</result>",
        )


def test_reports_all_invalid_output_indices():
    assert find_invalid_rollout_regex_outputs(
        ["<result>ok</result>", "truncated", "<result>also ok</result>", "bad"],
        r"<result>.*</result>",
    ) == [1, 3]


def test_tts_regex_accepts_long_example_and_bounds_runaway_values():
    regex_path = (
        Path(__file__).parents[1]
        / "examples"
        / "on_policy_kd"
        / "tts_rollout.regex"
    )
    regex = load_rollout_regex(str(regex_path))
    long_output = (
        "<o><s>16 GB/256 GB</s><v>mười sáu gigabyte chia hai trăm năm mươi "
        "sáu gigabyte</v><s>2.899 tệ</s><v>hai nghìn tám trăm chín mươi chín "
        "tệ</v><s>(10,7 triệu đồng)</s><v>mười triệu bảy trăm nghìn đồng</v></o>"
    )

    validate_rollout_regex_outputs([long_output], regex)
    assert find_invalid_rollout_regex_outputs(
        [f"<o><s>x</s><v>{'a' * 769}</v></o>"], regex
    ) == [0]
