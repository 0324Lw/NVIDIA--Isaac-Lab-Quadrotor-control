#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

NUM_ENVS="${1:-16}"
TOTAL_ENV_STEPS="${2:-50000000}"
DEVICE="${3:-cuda:0}"
TASK3_CKPT="${4:-}"

echo "============================================================"
echo "Quadrotor / Crazyflie Task4 TRUE skrl PPO laptop training"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "NUM_ENVS=${NUM_ENVS}"
echo "TOTAL_ENV_STEPS=${TOTAL_ENV_STEPS}"
echo "DEVICE=${DEVICE}"
echo "TASK3_CKPT=${TASK3_CKPT:-<none>}"
echo "PYTHON=$(which python)"
echo "============================================================"

EXTRA_ARGS=()
if [ -n "${TASK3_CKPT}" ]; then
  EXTRA_ARGS+=(--pretrained-task3 "${TASK3_CKPT}")
fi

python src/quadrotor_rl/tasks/task4/task4_train.py \
  --num-envs "${NUM_ENVS}" \
  --total-env-steps "${TOTAL_ENV_STEPS}" \
  --save-freq-env-steps 1000000 \
  --rollouts 128 \
  --learning-epochs 6 \
  --mini-batches 8 \
  --lr 3.0e-4 \
  --min-lr 1.0e-5 \
  --max-lr 3.0e-4 \
  --entropy-loss-scale 0.01 \
  --cnn-output-dim 256 \
  --compact-output-dim 128 \
  --hidden-dim 256 \
  --enable-sensor-noise \
  --test-device "${DEVICE}" \
  --headless \
  "${EXTRA_ARGS[@]}"
