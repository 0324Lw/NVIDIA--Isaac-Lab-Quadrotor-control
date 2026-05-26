#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
  echo "Usage:"
  echo "  bash scripts/ubuntu/eval_task3_skrl.sh /path/to/checkpoint_or_final_checkpoint_dir [eval_curriculum]"
  echo ""
  echo "eval_curriculum:"
  echo "  fixed_easy | fixed_medium | fixed_hard"
  echo ""
  echo "Examples:"
  echo "  bash scripts/ubuntu/eval_task3_skrl.sh logs/task3/<run>/final_checkpoint fixed_easy"
  echo "  bash scripts/ubuntu/eval_task3_skrl.sh logs/task3/<run>/final_checkpoint fixed_hard"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

CKPT="$1"
EVAL_CURRICULUM="${2:-fixed_easy}"

OUT_DIR="${PROJECT_ROOT}/outputs/task3_eval"
mkdir -p "${OUT_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
PLOT_PATH="${OUT_DIR}/task3_navigation_eval_${EVAL_CURRICULUM}_${STAMP}.png"
NPZ_PATH="${OUT_DIR}/task3_navigation_eval_${EVAL_CURRICULUM}_${STAMP}.npz"

echo "============================================================"
echo "Quadrotor / Crazyflie Task3 TRUE skrl PPO model evaluation"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CHECKPOINT=${CKPT}"
echo "EVAL_CURRICULUM=${EVAL_CURRICULUM}"
echo "PLOT_PATH=${PLOT_PATH}"
echo "NPZ_PATH=${NPZ_PATH}"
echo "PYTHON=$(which python)"
echo "============================================================"

python - <<'PY'
import sys
print("[CHECK] Python:", sys.executable)

import torch
print("[CHECK] torch:", torch.__version__)
print("[CHECK] cuda:", torch.cuda.is_available())

import isaaclab
print("[CHECK] isaaclab: ok")

import skrl
print("[CHECK] skrl:", getattr(skrl, "__version__", "unknown"))
PY

python src/quadrotor_rl/tasks/task3/task3_model_test.py \
  --checkpoint "${CKPT}" \
  --num-envs 4 \
  --steps 1000 \
  --print-interval 20 \
  --max-episode-length-s 20.0 \
  --eval-curriculum "${EVAL_CURRICULUM}" \
  --save-plot "${PLOT_PATH}" \
  --save-npz "${NPZ_PATH}" \
  --headless \
  --test-device cuda:0
