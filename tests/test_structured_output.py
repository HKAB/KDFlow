import pytest

from kdflow.utils.structured_output import (
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
