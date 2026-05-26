#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
  echo "Usage:"
  echo "  bash scripts/ubuntu/visual/visualize_task4.sh /path/to/checkpoint_or_final_checkpoint_dir [slow_action_scale]"
  echo ""
  echo "Examples:"
  echo "  bash scripts/ubuntu/visual/visualize_task4.sh logs/task4/<run>/final_checkpoint 1.0"
  echo "  bash scripts/ubuntu/visual/visualize_task4.sh logs/task4/<run>/final_checkpoint 0.5"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

CKPT="$1"
SLOW_ACTION_SCALE="${2:-1.0}"

OUT_DIR="${PROJECT_ROOT}/outputs/task4_visual"
mkdir -p "${OUT_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
PLOT_PATH="${OUT_DIR}/task4_visual_gate_racing_${STAMP}.png"
NPZ_PATH="${OUT_DIR}/task4_visual_gate_racing_${STAMP}.npz"

echo "============================================================"
echo "Quadrotor / Crazyflie Task4 TRUE skrl PPO GUI visualization"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CHECKPOINT=${CKPT}"
echo "SLOW_ACTION_SCALE=${SLOW_ACTION_SCALE}"
echo "PLOT_PATH=${PLOT_PATH}"
echo "NPZ_PATH=${NPZ_PATH}"
echo "PYTHON=$(which python)"
echo "============================================================"

python src/quadrotor_rl/tasks/task4/task4_model_test.py \
  --checkpoint "${CKPT}" \
  --num-envs 2 \
  --steps 3000 \
  --print-interval 20 \
  --max-episode-length-s 12.0 \
  --slow-action-scale "${SLOW_ACTION_SCALE}" \
  --visualize \
  --save-plot "${PLOT_PATH}" \
  --save-npz "${NPZ_PATH}" \
  --test-device cuda:0
