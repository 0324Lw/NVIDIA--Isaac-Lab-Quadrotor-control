#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

echo "============================================================"
echo "Quadrotor / Crazyflie Task1 Hover Stabilization Env Test"
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

python -m py_compile \
  src/quadrotor_rl/tasks/task1/task1_config.py \
  src/quadrotor_rl/tasks/task1/task1_scene.py \
  src/quadrotor_rl/tasks/task1/task1_env.py \
  tests/task1/task1_env_test.py

python tests/task1/task1_env_test.py \
  --num-envs 4 \
  --steps 200 \
  --collect-interval 20 \
  --test-device cuda:0 \
  --headless \
  --print-names
