#!/bin/bash
#SBATCH --job-name=kdflow-e2b-student
#SBATCH --account=truongnp5
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000G
#SBATCH --output=logs/kdflow-e2b-student-%j.out
#SBATCH --error=logs/kdflow-e2b-student-%j.err

# Submit with --exclude=<teacher-hostname>. Ray requires the teacher and student
# raylets to have different node IPs. The student raylet and Python driver run
# in this single srun container so they share its private /tmp socket namespace.

: "${TEACHER_RAY_ADDRESS:?Set TEACHER_RAY_ADDRESS=<teacher-ip>:6379}"
: "${TEACHER_HOSTNAME:?Set TEACHER_HOSTNAME=<teacher-hostname>}"

CONTAINER="/mnt/data/slurm/images/songmzhang+kdflow+v0.2.1-sgl0517-torch211-cu129.sqsh"
KDFLOW_SRC="/mnt/data/users/truongnp5/uv_env/kdflow/experiment/KDFlow"
EXPECTED_BASE_COMMIT="b83de0f64c5ca404c969e27cea45c6da3ac35485"
TRAIN_GPUS=4
TEACHER_GPUS=4

MAX_SAMPLES="${MAX_SAMPLES:-}"
if [ -n "${MAX_SAMPLES}" ]; then
    NUM_EPOCHS="${NUM_EPOCHS:-1}"
else
    NUM_EPOCHS="${NUM_EPOCHS:-2}"
fi

STUDENT_REPO="/mnt/data/huggingface/hub/models--google--gemma-4-E2B-it"
TEACHER_REPO="/mnt/data/huggingface/hub/models--google--gemma-4-31B-it"
STUDENT="${STUDENT_REPO}/snapshots/$(cat "${STUDENT_REPO}/refs/main")"
TEACHER="${TEACHER_REPO}/snapshots/$(cat "${TEACHER_REPO}/refs/main")"

DATA="/mnt/data/users/truongnp5/uv_env/kdflow/data/train.jsonl"
HF_CACHE="/mnt/data/users/truongnp5/cache/huggingface"
OUTPUT="/mnt/data/users/truongnp5/uv_env/kdflow/gemma4-31b-to-e2b-remote"
CKPT="${OUTPUT}/checkpoints"
WANDB_DIR="${OUTPUT}/wandb"
TRITON_CACHE_DIR="/tmp/${USER}/triton-${SLURM_JOB_ID}"
STUDENT_HOSTNAME="$(hostname)"
STUDENT_IP="$(hostname -I | awk '{print $1}')"
TEACHER_IP="${TEACHER_RAY_ADDRESS%:*}"

mkdir -p logs "${OUTPUT}" "${WANDB_DIR}" "${HF_CACHE}/datasets"
test -f "${KDFLOW_SRC}/kdflow/backend/sglang/hidden_state_alignment.py" || exit 1
git -C "${KDFLOW_SRC}" merge-base --is-ancestor "${EXPECTED_BASE_COMMIT}" HEAD \
    || exit 1
if ! grep -q persistent_remote "${KDFLOW_SRC}/kdflow/cli/train_kd_on_policy.py"; then
    echo "ERROR: KDFLOW_SRC does not contain the remote-teacher implementation."
    exit 2
fi
if [ "${STUDENT_HOSTNAME}" = "${TEACHER_HOSTNAME}" ] \
    || [ "${STUDENT_IP}" = "${TEACHER_IP}" ]; then
    echo "ERROR: teacher and student jobs landed on the same Ray node/IP."
    echo "Teacher: ${TEACHER_HOSTNAME} (${TEACHER_IP})"
    echo "Student: ${STUDENT_HOSTNAME} (${STUDENT_IP})"
    echo "Resubmit this job with: --exclude=${TEACHER_HOSTNAME}"
    exit 2
fi

echo "Teacher: ${TEACHER_HOSTNAME} (${TEACHER_RAY_ADDRESS})"
echo "Student: ${STUDENT_HOSTNAME} (${STUDENT_IP})"

srun --container-image="${CONTAINER}" \
    --container-mounts="/mnt/data:/mnt/data" \
    env PYTHONPATH="${KDFLOW_SRC}" HF_HOME="${HF_CACHE}" \
        TRANSFORMERS_CACHE="${HF_CACHE}" \
        HF_DATASETS_CACHE="${HF_CACHE}/datasets" \
        TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
        NCCL_DEBUG=WARN NCCL_IB_DISABLE=0 \
        KDFLOW_ROLLOUT_REGEX_FILE="/mnt/data/users/truongnp5/uv_env/kdflow/config/asr_rollout.regex" \
        KDFLOW_KD_SUPPRESS_TOKEN_IDS="98,100,101" \
        KDFLOW_SRC="${KDFLOW_SRC}" STUDENT="${STUDENT}" TEACHER="${TEACHER}" \
        DATA="${DATA}" OUTPUT="${OUTPUT}" CKPT="${CKPT}" \
        WANDB_DIR="${WANDB_DIR}" TRITON_CACHE_DIR="${TRITON_CACHE_DIR}" \
        TEACHER_RAY_ADDRESS="${TEACHER_RAY_ADDRESS}" STUDENT_IP="${STUDENT_IP}" \
        TRAIN_GPUS="${TRAIN_GPUS}" TEACHER_GPUS="${TEACHER_GPUS}" \
        NUM_EPOCHS="${NUM_EPOCHS}" MAX_SAMPLES="${MAX_SAMPLES}" \
    bash -s <<'INNER_BASH'

cleanup() {
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

mkdir -p "${TRITON_CACHE_DIR}"
ray start --address="${TEACHER_RAY_ADDRESS}" \
    --node-ip-address="${STUDENT_IP}" \
    --num-gpus="${TRAIN_GPUS}" --num-cpus=64 \
    --resources="{\"student_node\": ${TRAIN_GPUS}}" \
    --disable-usage-stats || exit 1

ray status --address="${TEACHER_RAY_ADDRESS}" || exit 1

cd "${KDFLOW_SRC}" || exit 1
IMPORTED_KDFLOW="$(python -c 'import kdflow; print(kdflow.__file__)')" || exit 1
echo "Imported KDFlow: ${IMPORTED_KDFLOW}"
case "${IMPORTED_KDFLOW}" in
    "${KDFLOW_SRC}"/*) ;;
    *) echo "ERROR: Python imported stale KDFlow from ${IMPORTED_KDFLOW}"; exit 2 ;;
esac
python -c "from kdflow.arguments.distillation_args import DistillationArguments as A; assert hasattr(A, 'teacher_mode')" \
    || exit 1

EXTRA_TRAIN_ARGS=()
if [ -n "${MAX_SAMPLES}" ]; then
    EXTRA_TRAIN_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

RAY_ADDRESS="${TEACHER_RAY_ADDRESS}" python -m kdflow.cli.train_kd_on_policy \
    --student_name_or_path "${STUDENT}" \
    --teacher_name_or_path "${TEACHER}" \
    --attn_implementation sdpa --use_liger_kernel False --enable_thinking False \
    --num_nodes 1 --num_gpus_per_node "${TRAIN_GPUS}" \
    --backend fsdp2 --fsdp_size -1 --cpu_offload False \
    --train_dataset_path "${DATA}" --input_key messages \
    --apply_chat_template True --preprocess_num_workers 32 \
    --prompt_max_len 8192 --generate_max_len 256 --max_len 8448 \
    --train_batch_size 64 --micro_train_batch_size 1 --num_epochs "${NUM_EPOCHS}" \
    --use_dynamic_bsz True --max_token_len_per_gpu 12288 \
    --chunked_loss_size 1024 \
    --learning_rate 1e-6 --min_lr 1e-8 --lr_warmup_ratio 0.05 \
    --max_norm 1.0 --weight_decay 0.0 --bf16 True \
    --gradient_checkpointing True \
    --kd_algorithm vanilla_kd --kd_loss_fn rkl --kd_ratio 1.0 \
    --kd_temperature 1.0 \
    --teacher_mode persistent_remote \
    --teacher_num_nodes 1 --teacher_num_gpus_per_node "${TEACHER_GPUS}" \
    --teacher_resource_name teacher_node --student_resource_name student_node \
    --teacher_disable_radix_cache False \
    --teacher_tp_size "${TEACHER_GPUS}" --teacher_dp_size 1 --teacher_pp_size 1 \
    --teacher_mem_fraction_static 0.80 --teacher_forward_n_batches 4 \
    --rollout_num_engines "${TRAIN_GPUS}" --rollout_tp_size 1 \
    --rollout_batch_size 64 --n_samples_per_prompt 1 \
    --rollout_mem_fraction_static 0.25 \
    --temperature 0.7 --top_p 0.95 --print_rollout_sample False \
    --enable_sleep True \
    --logging_steps 5 --use_wandb True --wandb_mode offline \
    --wandb_project gemma4-asr-kd --wandb_group 31b-to-e2b-remote \
    --wandb_run_name "kdflow-${SLURM_JOB_ID}" --wandb_dir "${WANDB_DIR}" \
    --save_steps 500 --eval_steps -1 --ckpt_path "${CKPT}" \
    --save_path "${OUTPUT}" "${EXTRA_TRAIN_ARGS[@]}"

TRAIN_EXIT=$?
if [ "${TRAIN_EXIT}" -ne 0 ]; then
    echo "Student training failed with exit code ${TRAIN_EXIT}."
    exit "${TRAIN_EXIT}"
fi
echo "Student training finished successfully. Teacher job remains available."

INNER_BASH
