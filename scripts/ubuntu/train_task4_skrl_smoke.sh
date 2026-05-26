#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

NUM_ENVS="${1:-2}"
TOTAL_ENV_STEPS="${2:-5000}"
DEVICE="${3:-cuda:0}"

echo "============================================================"
echo "Quadrotor / Crazyflie Task4 TRUE skrl PPO smoke training"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "NUM_ENVS=${NUM_ENVS}"
echo "TOTAL_ENV_STEPS=${TOTAL_ENV_STEPS}"
echo "DEVICE=${DEVICE}"
echo "PYTHON=$(which python)"
echo "============================================================"

python src/quadrotor_rl/tasks/task4/task4_train.py \
  --num-envs "${NUM_ENVS}" \
  --total-env-steps "${TOTAL_ENV_STEPS}" \
  --save-freq-env-steps "${TOTAL_ENV_STEPS}" \
  --rollouts 8 \
  --learning-epochs 1 \
  --mini-batches 1 \
  --lr 3.0e-4 \
  --cnn-output-dim 128 \
  --compact-output-dim 64 \
  --hidden-dim 128 \
  --test-device "${DEVICE}" \
  --headless
