import pytest

from kdflow.utils.checkpointing import (
    CHECKPOINT_SUCCESS_FILE,
    TRAINER_STATE_FILE,
    finalize_checkpoint,
    prepare_checkpoint_directory,
    prune_checkpoints,
    resolve_checkpoint_path,
    validate_resume_metadata,
)


def _complete_checkpoint(root, step):
    checkpoint = prepare_checkpoint_directory(str(root), step)
    trainer_state = root / f"step_{step:08d}" / TRAINER_STATE_FILE
    trainer_state.write_bytes(b"state")
    finalize_checkpoint(str(root), checkpoint)
    return checkpoint


def test_latest_resolves_only_completed_checkpoint(tmp_path):
    checkpoint = _complete_checkpoint(tmp_path, 12)

    assert resolve_checkpoint_path(str(tmp_path), "latest") == checkpoint
    assert resolve_checkpoint_path(str(tmp_path), "step_00000012") == checkpoint


def test_incomplete_checkpoint_is_never_resumable(tmp_path):
    checkpoint = prepare_checkpoint_directory(str(tmp_path), 3)
    (tmp_path / "latest").write_text("step_00000003\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="incomplete"):
        resolve_checkpoint_path(str(tmp_path), checkpoint)


def test_incomplete_checkpoint_can_be_retried(tmp_path):
    old_checkpoint = prepare_checkpoint_directory(str(tmp_path), 3)
    (tmp_path / "step_00000003" / "partial").write_text("bad")

    new_checkpoint = prepare_checkpoint_directory(str(tmp_path), 3)

    assert new_checkpoint == old_checkpoint
    assert not (tmp_path / "step_00000003" / "partial").exists()


def test_prunes_old_completed_checkpoints(tmp_path):
    first = _complete_checkpoint(tmp_path, 1)
    second = _complete_checkpoint(tmp_path, 2)
    third = _complete_checkpoint(tmp_path, 3)

    prune_checkpoints(str(tmp_path), keep=2)

    assert not (tmp_path / "step_00000001").exists()
    assert (tmp_path / "step_00000002" / CHECKPOINT_SUCCESS_FILE).is_file()
    assert (tmp_path / "step_00000003" / CHECKPOINT_SUCCESS_FILE).is_file()
    assert second != third
    assert first != second


def test_rejects_resume_configuration_mismatch():
    with pytest.raises(ValueError, match="student_world_size"):
        validate_resume_metadata(
            {"student_world_size": 4}, {"student_world_size": 8}
        )


def test_allows_only_explicit_resume_configuration_mismatches():
    mismatches = validate_resume_metadata(
        {"student_world_size": 4, "temperature": 0.7},
        {"student_world_size": 4, "temperature": 0.4},
        allowed_mismatch_keys={"temperature"},
    )

    assert mismatches == {"temperature": (0.7, 0.4)}
