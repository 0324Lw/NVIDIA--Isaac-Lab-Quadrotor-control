#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
  echo "Usage:"
  echo "  bash scripts/ubuntu/visual/visualize_task1.sh /path/to/checkpoint_or_final_checkpoint_dir [slow_action_scale]"
  echo ""
  echo "Examples:"
  echo "  bash scripts/ubuntu/visual/visualize_task1.sh logs/task1/<run>/final_checkpoint 1.0"
  echo "  bash scripts/ubuntu/visual/visualize_task1.sh logs/task1/<run>/final_checkpoint 0.5"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

CKPT="$1"
SLOW_ACTION_SCALE="${2:-1.0}"

echo "============================================================"
echo "Quadrotor / Crazyflie Task1 TRUE skrl PPO GUI visualization"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CHECKPOINT=${CKPT}"
echo "SLOW_ACTION_SCALE=${SLOW_ACTION_SCALE}"
echo "PYTHON=$(which python)"
echo "============================================================"

python src/quadrotor_rl/tasks/task1/task1_model_test.py \
  --checkpoint "${CKPT}" \
  --num-envs 4 \
  --steps 2000 \
  --print-interval 20 \
  --max-episode-length-s 10.0 \
  --slow-action-scale "${SLOW_ACTION_SCALE}" \
  --hold-success \
  --visualize \
  --test-device cuda:0
