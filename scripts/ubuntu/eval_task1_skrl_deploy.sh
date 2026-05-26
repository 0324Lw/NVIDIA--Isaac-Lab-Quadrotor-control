#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

if [ $# -lt 1 ]; then
  echo "Usage:"
  echo "  bash scripts/ubuntu/eval_task1_skrl_deploy.sh /path/to/checkpoint_or_final_checkpoint_dir [extra args...]"
  echo ""
  echo "This is a Go2-framework-aligned deploy entry."
  echo "It forwards to scripts/ubuntu/eval_task1_skrl.sh."
  exit 1
fi

if [ ! -f "scripts/ubuntu/eval_task1_skrl.sh" ]; then
  echo "[FATAL] scripts/ubuntu/eval_task1_skrl.sh not found."
  exit 1
fi

exec bash scripts/ubuntu/eval_task1_skrl.sh "$@"
