#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
  echo "Usage:"
  echo "  bash scripts/ubuntu/eval_task2_skrl.sh /path/to/checkpoint_or_final_checkpoint_dir"
  echo ""
  echo "Examples:"
  echo "  bash scripts/ubuntu/eval_task2_skrl.sh logs/task2/<run>/final_checkpoint"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

CKPT="$1"
OUT_DIR="${PROJECT_ROOT}/outputs/task2_eval"
mkdir -p "${OUT_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
PLOT_PATH="${OUT_DIR}/task2_tracking_eval_${STAMP}.png"
NPZ_PATH="${OUT_DIR}/task2_tracking_eval_${STAMP}.npz"

echo "============================================================"
echo "Quadrotor / Crazyflie Task2 TRUE skrl PPO model evaluation"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CHECKPOINT=${CKPT}"
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

python src/quadrotor_rl/tasks/task2/task2_model_test.py \
  --checkpoint "${CKPT}" \
  --num-envs 4 \
  --steps 800 \
  --print-interval 20 \
  --max-episode-length-s 16.667 \
  --save-plot "${PLOT_PATH}" \
  --save-npz "${NPZ_PATH}" \
  --headless \
  --test-device cuda:0
