#!/bin/bash
#SBATCH --job-name=kdflow-teacher-cache-bench
#SBATCH --account=truongnp5
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=1000G
#SBATCH --output=logs/teacher-cache-bench-%j.out
#SBATCH --error=logs/teacher-cache-bench-%j.err

set -euo pipefail

CONTAINER="/mnt/data/slurm/images/songmzhang+kdflow+v0.2.1-sgl0517-torch211-cu129.sqsh"
KDFLOW_SRC="/mnt/data/users/truongnp5/uv_env/kdflow/KDFlow-main"
TEACHER_REPO="/mnt/data/huggingface/hub/models--google--gemma-4-31B-it"
DATA="/mnt/data/users/truongnp5/uv_env/kdflow/data/train.jsonl"
TEACHER_REV="$(cat "${TEACHER_REPO}/refs/main")"
TEACHER="${TEACHER_REPO}/snapshots/${TEACHER_REV}"

mkdir -p logs
for CACHE_MODE in enabled disabled; do
    EXTRA_ARGS=()
    if [ "${CACHE_MODE}" = disabled ]; then
        EXTRA_ARGS+=(--disable-radix-cache)
    fi
    echo "Running radix cache: ${CACHE_MODE}"
    srun --container-image="${CONTAINER}" --container-mounts="/mnt/data:/mnt/data" \
        env PYTHONPATH="${KDFLOW_SRC}" PYTHONUNBUFFERED=1 \
        python "${KDFLOW_SRC}/server_tests/teacher_prefix_cache_benchmark.py" \
        --model "${TEACHER}" --dataset "${DATA}" --tp-size 4 \
        --mem-fraction-static 0.8 "${EXTRA_ARGS[@]}"
done
