#!/bin/bash
#SBATCH --job-name=kdflow-31b-teacher
#SBATCH --account=truongnp5
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=1000G
#SBATCH --output=logs/kdflow-31b-teacher-%j.out
#SBATCH --error=logs/kdflow-31b-teacher-%j.err

# Start this job first and leave it running. It owns the Ray head and the four
# GPUs on which the student job will create the persistent TP4 teacher actor.

CONTAINER="/mnt/data/slurm/images/songmzhang+kdflow+v0.2.1-sgl0517-torch211-cu129.sqsh"
KDFLOW_SRC="/mnt/data/users/truongnp5/uv_env/kdflow/experiment/KDFlow"
CONNECTION_DIR="/mnt/data/users/truongnp5/uv_env/kdflow/ray-connections"
RAY_PORT="${RAY_PORT:-6379}"

mkdir -p logs "${CONNECTION_DIR}"
test -f "${KDFLOW_SRC}/kdflow/backend/sglang/hidden_state_alignment.py"

TEACHER_IP="$(hostname -I | awk '{print $1}')"
TEACHER_HOSTNAME="$(hostname)"
RAY_ADDRESS="${TEACHER_IP}:${RAY_PORT}"
CONNECTION_FILE="${CONNECTION_DIR}/teacher-${SLURM_JOB_ID}.env"

printf 'TEACHER_RAY_ADDRESS=%q\nTEACHER_HOSTNAME=%q\nTEACHER_JOB_ID=%q\n' \
    "${RAY_ADDRESS}" "${TEACHER_HOSTNAME}" "${SLURM_JOB_ID}" \
    > "${CONNECTION_FILE}"

cleanup() {
    srun --overlap --container-image="${CONTAINER}" \
        --container-mounts="/mnt/data:/mnt/data" \
        ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "============================================================"
echo "Teacher Ray head is starting"
echo "Hostname: ${TEACHER_HOSTNAME}"
echo "Address: ${RAY_ADDRESS}"
echo "Connection file: ${CONNECTION_FILE}"
echo
echo "Submit the student job with:"
echo "sbatch --exclude=${TEACHER_HOSTNAME} \\"
echo "  --export=ALL,TEACHER_RAY_ADDRESS=${RAY_ADDRESS},TEACHER_HOSTNAME=${TEACHER_HOSTNAME} \\"
echo "  examples/on_policy_kd/run_gemma4_e2b_student_rollout.slurm"
echo "============================================================"

srun --container-image="${CONTAINER}" \
    --container-mounts="/mnt/data:/mnt/data" \
    env PYTHONPATH="${KDFLOW_SRC}" TOKENIZERS_PARALLELISM=false \
        PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN NCCL_IB_DISABLE=0 \
    ray start --head --node-ip-address="${TEACHER_IP}" --port="${RAY_PORT}" \
        --num-gpus=4 --num-cpus=32 --resources='{"teacher_node": 4}' \
        --disable-usage-stats --block
