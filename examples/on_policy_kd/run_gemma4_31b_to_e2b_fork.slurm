#!/bin/bash
#SBATCH --job-name=kdflow-gemma4-31b-to-e2b
#SBATCH --account=truongnp5
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000G
#SBATCH --output=logs/kdflow-gemma4-31b-e2b-%j.out
#SBATCH --error=logs/kdflow-gemma4-31b-e2b-%j.err

# This job uses the existing checkout of the HKAB/KDFlow fork. It never clones
# songmzhang/KDFlow and never installs the old per-file overrides.
#
# Fresh run:
#   sbatch examples/on_policy_kd/run_gemma4_31b_to_e2b_fork.slurm
# Resume the latest completed checkpoint in the same output directory:
#   sbatch --export=ALL,RESUME_FROM_CHECKPOINT=latest \
#     examples/on_policy_kd/run_gemma4_31b_to_e2b_fork.slurm

CONTAINER="/mnt/data/slurm/images/songmzhang+kdflow+v0.2.1-sgl0517-torch211-cu129.sqsh"
KDFLOW_SRC="/mnt/data/users/truongnp5/uv_env/kdflow/experiment/KDFlow"
REQUIRED_KDFLOW_COMMIT="fa2b66255ba52bcdcaa995c1ad40b0f122ef2a28"

STUDENT_REPO="/mnt/data/huggingface/hub/models--google--gemma-4-E2B-it"
TEACHER_REPO="/mnt/data/huggingface/hub/models--google--gemma-4-31B-it"
STUDENT="${STUDENT_REPO}/snapshots/$(cat "${STUDENT_REPO}/refs/main")"
TEACHER="${TEACHER_REPO}/snapshots/$(cat "${TEACHER_REPO}/refs/main")"

DATA="/mnt/data/users/truongnp5/uv_env/kdflow/data/tts/train.jsonl"
HF_CACHE="/mnt/data/users/truongnp5/cache/huggingface"
HF_DATASETS_CACHE="${HF_CACHE}/datasets"
OUTPUT="/mnt/data/users/truongnp5/uv_env/kdflow/gemma4-31b-to-e2b-v2"
CKPT="${OUTPUT}/checkpoints"
WANDB_DIR="${OUTPUT}/wandb"
TRITON_CACHE_DIR="/tmp/${USER}/triton-${SLURM_JOB_ID}"

RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
SAVE_STEPS="${SAVE_STEPS:-250}"
MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-2}"

mkdir -p logs "${OUTPUT}" "${WANDB_DIR}" "${HF_DATASETS_CACHE}" \
    "${TRITON_CACHE_DIR}"

if [ ! -d "${KDFLOW_SRC}/.git" ]; then
    echo "ERROR: fork checkout does not exist: ${KDFLOW_SRC}"
    echo "Clone git@github.com:HKAB/KDFlow.git there before submitting."
    exit 2
fi

KDFLOW_ORIGIN="$(git -C "${KDFLOW_SRC}" remote get-url origin)" || exit 1
case "${KDFLOW_ORIGIN}" in
    *HKAB/KDFlow*) ;;
    *)
        echo "ERROR: ${KDFLOW_SRC} does not point to the HKAB/KDFlow fork."
        echo "origin=${KDFLOW_ORIGIN}"
        exit 2
        ;;
esac

git -C "${KDFLOW_SRC}" merge-base --is-ancestor \
    "${REQUIRED_KDFLOW_COMMIT}" HEAD || {
        echo "ERROR: ${KDFLOW_SRC} does not contain ${REQUIRED_KDFLOW_COMMIT}."
        echo "Update codex/persistent-remote-teacher before submitting."
        exit 2
    }

NODE_IP="$(hostname -I | awk '{print $1}')"
RAY_PORT="${RAY_PORT:-6379}"
RAY_ADDRESS="${NODE_IP}:${RAY_PORT}"

echo "================================================================"
echo "HOST          : $(hostname)"
echo "JOB           : ${SLURM_JOB_ID}"
echo "KDFLOW SOURCE : ${KDFLOW_SRC}"
echo "KDFLOW COMMIT : $(git -C "${KDFLOW_SRC}" rev-parse HEAD)"
echo "STUDENT       : ${STUDENT}"
echo "TEACHER       : ${TEACHER}"
echo "DATA          : ${DATA}"
echo "OUTPUT        : ${OUTPUT}"
echo "RESUME        : ${RESUME_FROM_CHECKPOINT:-disabled}"
echo "================================================================"

srun --nodes=1 --ntasks=1 \
    --container-image="${CONTAINER}" \
    --container-mounts="/mnt/data:/mnt/data" \
    env PYTHONPATH="${KDFLOW_SRC}" \
        HF_HOME="${HF_CACHE}" TRANSFORMERS_CACHE="${HF_CACHE}" \
        HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
        TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
        NCCL_DEBUG=WARN NCCL_IB_DISABLE=0 \
        TRITON_CACHE_DIR="${TRITON_CACHE_DIR}" \
        KDFLOW_ROLLOUT_REGEX_FILE="${KDFLOW_SRC}/examples/on_policy_kd/tts_rollout.regex" \
        KDFLOW_KD_SUPPRESS_TOKEN_IDS="98,100,101" \
        KDFLOW_SRC="${KDFLOW_SRC}" STUDENT="${STUDENT}" TEACHER="${TEACHER}" \
        DATA="${DATA}" OUTPUT="${OUTPUT}" CKPT="${CKPT}" \
        WANDB_DIR="${WANDB_DIR}" NODE_IP="${NODE_IP}" \
        RAY_PORT="${RAY_PORT}" RAY_ADDRESS="${RAY_ADDRESS}" \
        RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT}" \
        SAVE_STEPS="${SAVE_STEPS}" MAX_CHECKPOINTS="${MAX_CHECKPOINTS}" \
    bash -s <<'INNER_BASH'

cleanup() {
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

cd "${KDFLOW_SRC}" || exit 1

IMPORTED_KDFLOW="$(python -c 'import kdflow; print(kdflow.__file__)')" || exit 1
echo "Imported KDFlow: ${IMPORTED_KDFLOW}"
case "${IMPORTED_KDFLOW}" in
    "${KDFLOW_SRC}"/*) ;;
    *)
        echo "ERROR: imported stale KDFlow from ${IMPORTED_KDFLOW}"
        exit 2
        ;;
esac

python - <<'PY' || exit 1
from kdflow.arguments.rollout_args import RolloutArguments
from kdflow.arguments.training_args import TrainingArguments

assert hasattr(TrainingArguments, "resume_from_checkpoint")
assert hasattr(RolloutArguments, "rollout_regex_max_retries")
print("Resumable checkpoint implementation is available.")
PY

ray stop --force >/dev/null 2>&1 || true
ray start --head --node-ip-address="${NODE_IP}" --port="${RAY_PORT}" \
    --num-gpus=8 --num-cpus=64 --disable-usage-stats || exit 1
ray status --address="${RAY_ADDRESS}" || exit 1

EXTRA_TRAIN_ARGS=()
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    EXTRA_TRAIN_ARGS+=(
        --resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}"
        --resume_allow_rollout_config_change True
    )
fi

python -m kdflow.cli.train_kd_on_policy \
    --student_name_or_path "${STUDENT}" \
    --teacher_name_or_path "${TEACHER}" \
    --attn_implementation sdpa --use_liger_kernel False --enable_thinking False \
    --num_nodes 1 --num_gpus_per_node 8 --backend fsdp2 \
    --fsdp_size -1 --cpu_offload False \
    --train_dataset_path "${DATA}" --input_key messages \
    --apply_chat_template True --preprocess_num_workers 32 \
    --prompt_max_len 8192 --generate_max_len 1024 --max_len 9216 \
    --train_batch_size 64 --micro_train_batch_size 1 --num_epochs 2 \
    --use_dynamic_bsz True --max_token_len_per_gpu 12288 \
    --chunked_loss_size 1024 \
    --learning_rate 1e-6 --min_lr 1e-8 --lr_warmup_ratio 0.05 \
    --max_norm 1.0 --weight_decay 0.0 --bf16 True \
    --gradient_checkpointing True \
    --kd_algorithm vanilla_kd --kd_loss_fn rkl --kd_ratio 1.0 \
    --kd_temperature 1.0 \
    --teacher_tp_size 4 --teacher_dp_size 2 --teacher_pp_size 1 \
    --teacher_mem_fraction_static 0.36 --teacher_forward_n_batches 4 \
    --rollout_num_engines 8 --rollout_tp_size 1 --rollout_batch_size 64 \
    --n_samples_per_prompt 1 --rollout_mem_fraction_static 0.25 \
    --temperature 0.7 --top_p 0.95 --print_rollout_sample False \
    --rollout_regex_max_retries 2 --rollout_regex_retry_temperature 0.4 \
    --enable_sleep True \
    --logging_steps 5 --use_wandb True --wandb_mode offline \
    --wandb_project gemma4-tts-kd --wandb_group 31b-to-e2b \
    --wandb_run_name "kdflow-${SLURM_JOB_ID}" --wandb_dir "${WANDB_DIR}" \
    --save_steps "${SAVE_STEPS}" --max_checkpoints "${MAX_CHECKPOINTS}" \
    --eval_steps -1 --ckpt_path "${CKPT}" --save_path "${OUTPUT}" \
    "${EXTRA_TRAIN_ARGS[@]}"

TRAIN_EXIT=$?
if [ "${TRAIN_EXIT}" -ne 0 ]; then
    echo "Training failed with exit code ${TRAIN_EXIT}."
    exit "${TRAIN_EXIT}"
fi
echo "Training finished successfully."

INNER_BASH
