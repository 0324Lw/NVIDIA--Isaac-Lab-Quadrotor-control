#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
  echo "Usage:"
  echo "  bash scripts/ubuntu/eval_task4_skrl.sh /path/to/checkpoint_or_final_checkpoint_dir"
  echo ""
  echo "Examples:"
  echo "  bash scripts/ubuntu/eval_task4_skrl.sh logs/task4/<run>/final_checkpoint"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

CKPT="$1"

OUT_DIR="${PROJECT_ROOT}/outputs/task4_eval"
mkdir -p "${OUT_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
PLOT_PATH="${OUT_DIR}/task4_gate_racing_eval_${STAMP}.png"
NPZ_PATH="${OUT_DIR}/task4_gate_racing_eval_${STAMP}.npz"

echo "============================================================"
echo "Quadrotor / Crazyflie Task4 TRUE skrl PPO model evaluation"
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

python src/quadrotor_rl/tasks/task4/task4_model_test.py \
  --checkpoint "${CKPT}" \
  --num-envs 2 \
  --steps 1000 \
  --print-interval 20 \
  --max-episode-length-s 12.0 \
  --save-plot "${PLOT_PATH}" \
  --save-npz "${NPZ_PATH}" \
  --headless \
  --test-device cuda:0
