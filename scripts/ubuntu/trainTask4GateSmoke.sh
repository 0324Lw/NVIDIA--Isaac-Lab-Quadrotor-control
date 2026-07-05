#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
findProjectRoot() {
  local searchDir="$1"
  while [[ "$searchDir" != "/" ]]; do
    if [[ -f "$searchDir/src/quadrotor_rl/__init__.py" ]]; then
      printf '%s\n' "$searchDir"
      return 0
    fi
    searchDir="$(dirname "$searchDir")"
  done
  return 1
}
PROJECT_ROOT="$(findProjectRoot "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

python -m quadrotor_rl.tasks.task4.task4_train --num-envs "${NUM_ENVS:-512}" --total-env-steps "${TOTAL_ENV_STEPS:-5000}" --save-freq-env-steps "${SAVE_FREQ_ENV_STEPS:-5000}" "$@"
