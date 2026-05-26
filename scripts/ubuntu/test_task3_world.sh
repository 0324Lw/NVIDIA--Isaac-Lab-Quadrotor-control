#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

echo "============================================================"
echo "Quadrotor / Crazyflie Task3 Analytic Dynamic-Obstacle World Test"
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
    import numpy as np
    print("[CHECK] numpy:", np.__version__)
except Exception as e:
    raise RuntimeError("Current Python cannot import numpy.") from e
PY

python -m py_compile \
  src/quadrotor_rl/tasks/task3/task3_config.py \
  src/quadrotor_rl/tasks/task3/task3_world.py \
  tests/task3/task3_world_test.py

python tests/task3/task3_world_test.py \
  --num-envs 64 \
  --steps 300 \
  --test-device cuda:0 \
  --print-every 50
