import hashlib
import os
import shutil
from pathlib import Path
from typing import Optional

from kdflow.utils.structured_output import load_rollout_regex
from kdflow.utils.token_suppression import load_kd_suppress_token_ids


CHECKPOINT_SUCCESS_FILE = "_SUCCESS"
LATEST_CHECKPOINT_FILE = "latest"
TRAINER_STATE_FILE = "trainer_state.pt"
ROLLOUT_RESUME_MUTABLE_KEYS = frozenset(
    {
        "generate_max_len",
        "temperature",
        "top_p",
        "rollout_regex_sha256",
        "rollout_regex_max_retries",
        "rollout_regex_retry_temperature",
    }
)


def build_checkpoint_invariants(
    args,
    dataset_size: int,
    steps_per_epoch: int,
    dataset_fingerprint: Optional[str] = None,
) -> dict:
    """Configuration values that must remain stable for an exact continuation."""
    rollout_regex = load_rollout_regex()
    return {
        "student_name_or_path": args.model.student_name_or_path,
        "teacher_name_or_path": args.model.teacher_name_or_path,
        "kd_algorithm": args.kd.kd_algorithm,
        "kd_loss_fn": args.kd.kd_loss_fn,
        "kd_temperature": args.kd.kd_temperature,
        "multi_teacher_config": args.kd.multi_teacher_config,
        "train_dataset_path": args.data.train_dataset_path,
        "train_split": args.data.train_split,
        "input_key": args.data.input_key,
        "teacher_input_key": args.data.teacher_input_key,
        "input_template": args.data.input_template,
        "apply_chat_template": args.data.apply_chat_template,
        "prompt_max_len": args.data.prompt_max_len,
        "student_world_size": args.train.num_nodes * args.train.num_gpus_per_node,
        "fsdp_size": args.fsdp.fsdp_size,
        "ring_attn_size": args.model.ring_attn_size,
        "train_batch_size": args.train.train_batch_size,
        "micro_train_batch_size": args.train.micro_train_batch_size,
        "rollout_batch_size": args.rollout.rollout_batch_size,
        "n_samples_per_prompt": args.rollout.n_samples_per_prompt,
        "dataset_size": dataset_size,
        "dataset_fingerprint": dataset_fingerprint,
        "steps_per_epoch": steps_per_epoch,
        "seed": args.train.seed,
        "num_epochs": args.train.num_epochs,
        "learning_rate": args.train.learning_rate,
        "lr_scheduler": args.train.lr_scheduler,
        "lr_warmup_ratio": args.train.lr_warmup_ratio,
        "min_lr": args.train.min_lr,
        "weight_decay": args.train.weight_decay,
        "adam_betas": tuple(args.train.adam_betas),
        "bf16": args.train.bf16,
        "gradient_checkpointing": args.train.gradient_checkpointing,
        "use_dynamic_bsz": args.train.use_dynamic_bsz,
        "max_token_len_per_gpu": args.train.max_token_len_per_gpu,
        "generate_max_len": args.rollout.generate_max_len,
        "temperature": args.rollout.temperature,
        "top_p": args.rollout.top_p,
        "rollout_regex_max_retries": args.rollout.rollout_regex_max_retries,
        "rollout_regex_retry_temperature": (
            args.rollout.rollout_regex_retry_temperature
        ),
        "rollout_regex_sha256": (
            hashlib.sha256(rollout_regex.encode("utf-8")).hexdigest()
            if rollout_regex is not None
            else None
        ),
        "kd_suppress_token_ids": load_kd_suppress_token_ids(),
    }


def resolve_checkpoint_path(
    checkpoint_root: str, resume_from_checkpoint: Optional[str]
) -> Optional[str]:
    """Resolve and validate an explicit checkpoint path or the latest pointer."""
    if not resume_from_checkpoint:
        return None

    root = Path(checkpoint_root).expanduser().resolve()
    if resume_from_checkpoint == "latest":
        latest_path = root / LATEST_CHECKPOINT_FILE
        if not latest_path.is_file():
            raise FileNotFoundError(f"No latest checkpoint pointer at {latest_path}")
        checkpoint = root / latest_path.read_text(encoding="utf-8").strip()
    else:
        requested = Path(resume_from_checkpoint).expanduser()
        checkpoint = requested if requested.is_absolute() else root / requested
        checkpoint = checkpoint.resolve()

    if not (checkpoint / CHECKPOINT_SUCCESS_FILE).is_file():
        raise RuntimeError(
            f"Checkpoint is incomplete or invalid (missing {CHECKPOINT_SUCCESS_FILE}): "
            f"{checkpoint}"
        )
    if not (checkpoint / TRAINER_STATE_FILE).is_file():
        raise RuntimeError(
            f"Checkpoint is missing {TRAINER_STATE_FILE}: {checkpoint}"
        )
    return str(checkpoint)


def prepare_checkpoint_directory(checkpoint_root: str, global_step: int) -> str:
    """Create an empty step directory, replacing only an incomplete retry."""
    root = Path(checkpoint_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / f"step_{global_step:08d}"
    if checkpoint.exists():
        if (checkpoint / CHECKPOINT_SUCCESS_FILE).exists():
            raise FileExistsError(f"Completed checkpoint already exists: {checkpoint}")
        shutil.rmtree(checkpoint)
    checkpoint.mkdir()
    return str(checkpoint)


def finalize_checkpoint(checkpoint_root: str, checkpoint_path: str) -> None:
    """Mark a checkpoint complete and atomically publish it as latest."""
    root = Path(checkpoint_root).expanduser().resolve()
    checkpoint = Path(checkpoint_path).resolve()
    if checkpoint.parent != root:
        raise ValueError(f"Checkpoint {checkpoint} is not directly under {root}")

    success_tmp = checkpoint / f".{CHECKPOINT_SUCCESS_FILE}.tmp"
    success_tmp.write_text("ok\n", encoding="utf-8")
    os.replace(success_tmp, checkpoint / CHECKPOINT_SUCCESS_FILE)
    latest_tmp = root / f".{LATEST_CHECKPOINT_FILE}.tmp"
    latest_tmp.write_text(f"{checkpoint.name}\n", encoding="utf-8")
    os.replace(latest_tmp, root / LATEST_CHECKPOINT_FILE)


def prune_checkpoints(checkpoint_root: str, keep: int) -> None:
    """Retain the newest completed checkpoints and ignore incomplete writes."""
    root = Path(checkpoint_root).expanduser().resolve()
    completed = sorted(
        (
            path
            for path in root.glob("step_*")
            if path.is_dir() and (path / CHECKPOINT_SUCCESS_FILE).is_file()
        ),
        key=lambda path: path.name,
    )
    for checkpoint in completed[:-keep]:
        shutil.rmtree(checkpoint)


def validate_resume_metadata(
    saved: dict, expected: dict, allowed_mismatch_keys=()
) -> dict:
    """Reject topology/data changes that would make continuation ambiguous."""
    mismatches = {
        key: (saved.get(key), value)
        for key, value in expected.items()
        if saved.get(key) != value
    }
    allowed_mismatch_keys = set(allowed_mismatch_keys)
    fatal_mismatches = {
        key: value
        for key, value in mismatches.items()
        if key not in allowed_mismatch_keys
    }
    if fatal_mismatches:
        details = ", ".join(
            f"{key}: saved={old!r}, current={new!r}"
            for key, (old, new) in sorted(fatal_mismatches.items())
        )
        raise ValueError(f"Checkpoint configuration mismatch: {details}")
    return {
        key: value
        for key, value in mismatches.items()
        if key in allowed_mismatch_keys
    }
