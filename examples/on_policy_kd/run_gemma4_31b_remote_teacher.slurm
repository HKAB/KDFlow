#!/bin/bash
#SBATCH --job-name=kdflow-gemma4-remote-teacher
#SBATCH --account=truongnp5
#SBATCH --partition=main
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=2000G
#SBATCH --output=logs/kdflow-gemma4-remote-teacher-%j.out
#SBATCH --error=logs/kdflow-gemma4-remote-teacher-%j.err

# This deliberately uses a conventional two-node allocation for portability.
# Node 0 uses all 8 GPUs for FSDP2 plus sleeping rollout engines. Node 1 exposes
# only 4 of its GPUs to Ray for the persistent TP4 teacher. If your cluster
# supports heterogeneous Slurm jobs, the second node can be requested as GPU:4
# without changing the commands below.

CONTAINER="/mnt/data/slurm/images/songmzhang+kdflow+v0.2.1-sgl0517-torch211-cu129.sqsh"
KDFLOW_SRC="/mnt/data/users/truongnp5/uv_env/kdflow/KDFlow-main"
EXPECTED_BASE_COMMIT="b83de0f64c5ca404c969e27cea45c6da3ac35485"
TRAIN_GPUS="${TRAIN_GPUS:-8}"
TEACHER_GPUS="${TEACHER_GPUS:-4}"
# For a one-iteration integration test, submit with:
#   sbatch --export=ALL,MAX_SAMPLES=64 this-script.slurm
MAX_SAMPLES="${MAX_SAMPLES:-}"
if [ -n "${MAX_SAMPLES}" ]; then
    NUM_EPOCHS="${NUM_EPOCHS:-1}"
else
    NUM_EPOCHS="${NUM_EPOCHS:-2}"
fi
if [ "${TEACHER_GPUS}" -ne 4 ]; then
    echo "This Gemma 4 31B script currently expects a TP4 teacher."
    exit 2
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

mkdir -p logs "${OUTPUT}" "${WANDB_DIR}" "${HF_CACHE}/datasets"
test -f "${KDFLOW_SRC}/kdflow/backend/sglang/hidden_state_alignment.py"
git -C "${KDFLOW_SRC}" merge-base --is-ancestor "${EXPECTED_BASE_COMMIT}" HEAD
if ! grep -q 'persistent_remote' "${KDFLOW_SRC}/kdflow/cli/train_kd_on_policy.py"; then
    echo "ERROR: KDFLOW_SRC does not contain the remote-teacher implementation."
    echo "Do not copy the old train_kd_on_policy.py ASR override over this tree."
    exit 2
fi

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [ "${#NODES[@]}" -ne 2 ]; then
    echo "Expected exactly two allocated nodes, got ${#NODES[@]}"
    exit 2
fi
TRAIN_NODE="${NODES[0]}"
TEACHER_NODE="${NODES[1]}"
HEAD_IP="$(srun --overlap -N1 -n1 -w "${TRAIN_NODE}" hostname -I | awk '{print $1}')"
RAY_PORT=6379

COMMON_ENV=(
    PYTHONPATH="${KDFLOW_SRC}"
    HF_HOME="${HF_CACHE}"
    TRANSFORMERS_CACHE="${HF_CACHE}"
    HF_DATASETS_CACHE="${HF_CACHE}/datasets"
    TOKENIZERS_PARALLELISM=false
    PYTHONUNBUFFERED=1
    NCCL_DEBUG=WARN
    NCCL_IB_DISABLE=0
    KDFLOW_ROLLOUT_REGEX_FILE="/mnt/data/users/truongnp5/uv_env/kdflow/config/asr_rollout.regex"
    KDFLOW_KD_SUPPRESS_TOKEN_IDS="98,100,101"
)

cleanup() {
    srun --overlap -N1 -n1 -w "${TRAIN_NODE}" \
        --container-image="${CONTAINER}" --container-mounts="/mnt/data:/mnt/data" \
        ray stop --force >/dev/null 2>&1 || true
    srun --overlap -N1 -n1 -w "${TEACHER_NODE}" \
        --container-image="${CONTAINER}" --container-mounts="/mnt/data:/mnt/data" \
        ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Training/rollout node: ${TRAIN_NODE} (${HEAD_IP})"
echo "Persistent teacher node: ${TEACHER_NODE} (${TEACHER_GPUS} GPUs)"

srun --overlap -N1 -n1 -w "${TRAIN_NODE}" \
    --container-image="${CONTAINER}" --container-mounts="/mnt/data:/mnt/data" \
    env "${COMMON_ENV[@]}" \
    ray start --head --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" \
        --num-gpus="${TRAIN_GPUS}" --num-cpus=64 \
        --resources="{\"student_node\": ${TRAIN_GPUS}}" \
        --disable-usage-stats --block &
HEAD_STEP_PID=$!

until srun --overlap -N1 -n1 -w "${TRAIN_NODE}" \
    --container-image="${CONTAINER}" --container-mounts="/mnt/data:/mnt/data" \
    ray status --address="${HEAD_IP}:${RAY_PORT}" >/dev/null 2>&1; do
    sleep 2
done

EXTRA_TRAIN_ARGS=()
if [ -n "${MAX_SAMPLES}" ]; then
    EXTRA_TRAIN_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

srun --overlap -N1 -n1 -w "${TEACHER_NODE}" \
    --container-image="${CONTAINER}" --container-mounts="/mnt/data:/mnt/data" \
    env "${COMMON_ENV[@]}" CUDA_VISIBLE_DEVICES=0,1,2,3 \
    ray start --address="${HEAD_IP}:${RAY_PORT}" --num-gpus="${TEACHER_GPUS}" \
        --num-cpus=32 --resources="{\"teacher_node\": ${TEACHER_GPUS}}" \
        --disable-usage-stats --block &
TEACHER_STEP_PID=$!

until srun --overlap -N1 -n1 -w "${TRAIN_NODE}" \
    --container-image="${CONTAINER}" --container-mounts="/mnt/data:/mnt/data" \
    ray status --address="${HEAD_IP}:${RAY_PORT}" 2>/dev/null | grep -q 'teacher_node'; do
    sleep 2
done

srun --overlap -N1 -n1 -w "${TRAIN_NODE}" \
    --container-image="${CONTAINER}" --container-mounts="/mnt/data:/mnt/data" \
    bash -lc "mkdir -p /tmp/${USER}/triton-${SLURM_JOB_ID}"

srun --overlap -N1 -n1 -w "${TRAIN_NODE}" \
    --container-image="${CONTAINER}" --container-mounts="/mnt/data:/mnt/data" \
    env "${COMMON_ENV[@]}" RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}" \
        TRITON_CACHE_DIR="/tmp/${USER}/triton-${SLURM_JOB_ID}" \
    python -m kdflow.cli.train_kd_on_policy \
    --student_name_or_path "${STUDENT}" \
    --teacher_name_or_path "${TEACHER}" \
    --attn_implementation sdpa --use_liger_kernel False --enable_thinking False \
    --num_nodes 1 --num_gpus_per_node "${TRAIN_GPUS}" --backend fsdp2 --fsdp_size -1 \
    --cpu_offload False \
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
    --rollout_num_engines "${TRAIN_GPUS}" --rollout_tp_size 1 --rollout_batch_size 64 \
    --n_samples_per_prompt 1 --rollout_mem_fraction_static 0.25 \
    --temperature 0.7 --top_p 0.95 --print_rollout_sample False \
    --enable_sleep True \
    --logging_steps 5 --use_wandb True --wandb_mode offline \
    --wandb_project gemma4-asr-kd --wandb_group 31b-to-e2b-remote \
    --wandb_run_name "kdflow-${SLURM_JOB_ID}" --wandb_dir "${WANDB_DIR}" \
    --save_steps 500 --eval_steps -1 --ckpt_path "${CKPT}" \
    --save_path "${OUTPUT}" "${EXTRA_TRAIN_ARGS[@]}"

echo "Training finished successfully."
