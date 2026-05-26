#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
  echo "Usage:"
  echo "  bash scripts/ubuntu/visual/visualize_task3.sh /path/to/checkpoint_or_final_checkpoint_dir [slow_action_scale] [eval_curriculum]"
  echo ""
  echo "Examples:"
  echo "  bash scripts/ubuntu/visual/visualize_task3.sh logs/task3/<run>/final_checkpoint 1.0 fixed_easy"
  echo "  bash scripts/ubuntu/visual/visualize_task3.sh logs/task3/<run>/final_checkpoint 0.5 fixed_hard"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

CKPT="$1"
SLOW_ACTION_SCALE="${2:-1.0}"
EVAL_CURRICULUM="${3:-fixed_easy}"

OUT_DIR="${PROJECT_ROOT}/outputs/task3_visual"
mkdir -p "${OUT_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
PLOT_PATH="${OUT_DIR}/task3_visual_navigation_${EVAL_CURRICULUM}_${STAMP}.png"
NPZ_PATH="${OUT_DIR}/task3_visual_navigation_${EVAL_CURRICULUM}_${STAMP}.npz"

echo "============================================================"
echo "Quadrotor / Crazyflie Task3 TRUE skrl PPO GUI visualization"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CHECKPOINT=${CKPT}"
echo "SLOW_ACTION_SCALE=${SLOW_ACTION_SCALE}"
echo "EVAL_CURRICULUM=${EVAL_CURRICULUM}"
echo "PLOT_PATH=${PLOT_PATH}"
echo "NPZ_PATH=${NPZ_PATH}"
echo "PYTHON=$(which python)"
echo "============================================================"

python src/quadrotor_rl/tasks/task3/task3_model_test.py \
  --checkpoint "${CKPT}" \
  --num-envs 4 \
  --steps 3000 \
  --print-interval 20 \
  --max-episode-length-s 20.0 \
  --slow-action-scale "${SLOW_ACTION_SCALE}" \
  --eval-curriculum "${EVAL_CURRICULUM}" \
  --hold-success \
  --visualize \
  --save-plot "${PLOT_PATH}" \
  --save-npz "${NPZ_PATH}" \
  --test-device cuda:0
