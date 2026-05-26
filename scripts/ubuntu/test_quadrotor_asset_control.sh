#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

echo "============================================================"
echo "Quadrotor Crazyflie asset interface and external-wrench control smoke test"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "PYTHON=$(which python)"
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<none>}"
echo "============================================================"

python - <<'PY'
import sys
print("[CHECK] Python:", sys.executable)

try:
    import torch
    print("[CHECK] torch:", torch.__version__)
    print("[CHECK] cuda available:", torch.cuda.is_available())
except Exception as e:
    raise RuntimeError("Current Python cannot import torch. Please activate conda env: isaaclab") from e

try:
    import isaaclab
    print("[CHECK] isaaclab: ok")
except Exception as e:
    raise RuntimeError("Current Python cannot import isaaclab. Please activate IsaacLab conda env.") from e
PY

python -m py_compile tests/asset/quadrotor_asset_control_test.py

python tests/asset/quadrotor_asset_control_test.py \
  --num-envs 1 \
  --steps 120 \
  --settle-steps 30 \
  --test-device cuda:0 \
  --spawn-height 1.0 \
  --force-scale 1.35 \
  --yaw-torque 0.02 \
  --headless \
  --print-names
