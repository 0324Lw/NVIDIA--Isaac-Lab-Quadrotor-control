#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

NUM_ENVS="${1:-128}"
TOTAL_ENV_STEPS="${2:-50000000}"
DEVICE="${3:-cuda:0}"
TASK1_CKPT="${4:-}"

echo "============================================================"
echo "Quadrotor / Crazyflie Task2 TRUE skrl PPO laptop training"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "NUM_ENVS=${NUM_ENVS}"
echo "TOTAL_ENV_STEPS=${TOTAL_ENV_STEPS}"
echo "DEVICE=${DEVICE}"
echo "TASK1_CKPT=${TASK1_CKPT:-<none>}"
echo "PYTHON=$(which python)"
echo "============================================================"

EXTRA_ARGS=()
if [ -n "${TASK1_CKPT}" ]; then
  EXTRA_ARGS+=(--pretrained-task1 "${TASK1_CKPT}")
fi

python src/quadrotor_rl/tasks/task2/task2_train.py \
  --num-envs "${NUM_ENVS}" \
  --total-env-steps "${TOTAL_ENV_STEPS}" \
  --save-freq-env-steps 5000000 \
  --rollouts 96 \
  --learning-epochs 5 \
  --mini-batches 8 \
  --lr 1.0e-4 \
  --test-device "${DEVICE}" \
  --headless \
  "${EXTRA_ARGS[@]}"
