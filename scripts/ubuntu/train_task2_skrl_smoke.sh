#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

NUM_ENVS="${1:-8}"
TOTAL_ENV_STEPS="${2:-5000}"
DEVICE="${3:-cuda:0}"

echo "============================================================"
echo "Quadrotor / Crazyflie Task2 TRUE skrl PPO smoke training"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "NUM_ENVS=${NUM_ENVS}"
echo "TOTAL_ENV_STEPS=${TOTAL_ENV_STEPS}"
echo "DEVICE=${DEVICE}"
echo "PYTHON=$(which python)"
echo "============================================================"

python src/quadrotor_rl/tasks/task2/task2_train.py \
  --num-envs "${NUM_ENVS}" \
  --total-env-steps "${TOTAL_ENV_STEPS}" \
  --save-freq-env-steps "${TOTAL_ENV_STEPS}" \
  --rollouts 16 \
  --learning-epochs 2 \
  --mini-batches 4 \
  --lr 1.0e-4 \
  --test-device "${DEVICE}" \
  --headless
