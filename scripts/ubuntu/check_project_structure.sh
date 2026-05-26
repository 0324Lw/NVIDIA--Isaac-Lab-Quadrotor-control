#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "============================================================"
echo "Quadrotor project structure check"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "============================================================"

required=(
  "README.md"
  "CHANGELOG.md"
  "CONTRIBUTING.md"
  "LICENSE"
  "pyproject.toml"
  ".gitignore"

  "assets/README.md"
  "assets/gifs/README.md"
  "assets/motions/README.md"
  "assets/usd/README.md"

  "configs/local_paths.example.yaml"
  "configs/platform_ubuntu_laptop.yaml"
  "configs/platform_windows_3090.yaml"
  "configs/task1_hover_stabilization.yaml"
  "configs/task2_waypoint_tracking.yaml"
  "configs/task3_obstacle_navigation.yaml"
  "configs/task4_sim2real_robust_flight.yaml"

  "docs/project_overview.md"
  "docs/results_and_checkpoints.md"
  "docs/task1_design.md"
  "docs/task2_design.md"
  "docs/task3_design.md"
  "docs/task4_design.md"
  "docs/troubleshooting.md"
  "docs/ubuntu_training.md"
  "docs/windows_path_config.md"
  "docs/windows_training.md"

  "src/quadrotor_rl/__init__.py"
  "src/quadrotor_rl/common/__init__.py"
  "src/quadrotor_rl/data/__init__.py"
  "src/quadrotor_rl/data/README.md"
  "src/quadrotor_rl/tasks/__init__.py"
  "src/quadrotor_rl/tasks/task1/__init__.py"
  "src/quadrotor_rl/tasks/task2/__init__.py"
  "src/quadrotor_rl/tasks/task3/__init__.py"
  "src/quadrotor_rl/tasks/task4/__init__.py"

  "tests/asset/README.md"
)

missing=0
for p in "${required[@]}"; do
  if [ -e "$p" ]; then
    echo "[OK] $p"
  else
    echo "[MISSING] $p"
    missing=$((missing + 1))
  fi
done

if [ "$missing" -ne 0 ]; then
  echo "[FAIL] missing_count=${missing}"
  exit 1
fi

echo "[PASS] quadrotor project structure check passed"
