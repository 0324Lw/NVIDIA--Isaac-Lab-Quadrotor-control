#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

NUM_ENVS="${1:-64}"
TOTAL_ENV_STEPS="${2:-50000000}"
DEVICE="${3:-cuda:0}"
TASK2_CKPT="${4:-}"

echo "============================================================"
echo "Quadrotor / Crazyflie Task3 TRUE skrl PPO laptop training"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "NUM_ENVS=${NUM_ENVS}"
echo "TOTAL_ENV_STEPS=${TOTAL_ENV_STEPS}"
echo "DEVICE=${DEVICE}"
echo "TASK2_CKPT=${TASK2_CKPT:-<none>}"
echo "PYTHON=$(which python)"
echo "============================================================"

EXTRA_ARGS=()
if [ -n "${TASK2_CKPT}" ]; then
  EXTRA_ARGS+=(--pretrained-task2 "${TASK2_CKPT}")
fi

python src/quadrotor_rl/tasks/task3/task3_train.py \
  --num-envs "${NUM_ENVS}" \
  --total-env-steps "${TOTAL_ENV_STEPS}" \
  --save-freq-env-steps 5000000 \
  --rollouts 96 \
  --learning-epochs 5 \
  --mini-batches 8 \
  --lr 1.0e-4 \
  --curriculum-mode serial \
  --phase1-frac 0.34 \
  --phase2-frac 0.67 \
  --phase1-static 5 \
  --phase1-dynamic 0 \
  --phase1-max-dist 25.0 \
  --phase2-static 10 \
  --phase2-dynamic 2 \
  --phase2-max-dist 25.0 \
  --phase3-static 25 \
  --phase3-dynamic 4 \
  --phase3-max-dist 45.0 \
  --test-device "${DEVICE}" \
  --headless \
  "${EXTRA_ARGS[@]}"
