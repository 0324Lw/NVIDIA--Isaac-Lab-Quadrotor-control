from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from quadrotor_rl.evaluation.checkpoint_selector import resolve_checkpoint_path


REQUIRED_TOP_LEVEL_KEYS = (
    "task_name",
    "actor_obs_dim",
    "critic_obs_dim",
    "action_dim",
    "control_dt",
    "action_semantics",
    "rotor_model",
    "normalizer",
)


def check_policy_io(path: str | Path) -> Dict[str, Any]:
    policy_io_path = Path(path).expanduser().resolve()
    if policy_io_path.is_dir():
        policy_io_path = policy_io_path / "policy_io.json"
    data = json.loads(policy_io_path.read_text(encoding="utf-8"))
    errors = []
    warnings = []
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in data:
            errors.append(f"missing key: {key}")
    if int(data.get("actor_obs_dim", 0)) <= 0:
        errors.append("actor_obs_dim must be positive")
    if int(data.get("critic_obs_dim", 0)) <= 0:
        errors.append("critic_obs_dim must be positive")
    if int(data.get("action_dim", 0)) != 4:
        errors.append("action_dim must be 4 for quadrotor motor-delta control")
    semantics = data.get("action_semantics", {})
    if semantics.get("pipeline") != ["clip", "deadzone", "scale", "ema", "motor_multiplier_delta", "wrench"]:
        warnings.append("action pipeline differs from canonical quadrotor semantics")
    if float(semantics.get("action_scale", 0.0)) <= 0.0:
        errors.append("action_scale must be positive")
    if not (0.0 < float(semantics.get("action_ema_alpha", 0.0)) <= 1.0):
        errors.append("action_ema_alpha must be in (0, 1]")
    normalizer = data.get("normalizer", {})
    convention = normalizer.get("onnx_input_convention", "raw_observation")
    if convention not in {"raw_observation", "normalized_observation"}:
        warnings.append("unknown ONNX input convention")
    if normalizer.get("normalizer_required") is False:
        warnings.append("observation normalizer is marked as optional; verify deployment input convention")
    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings, "policy_io": data}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check quadrotor policy_io.json")
    parser.add_argument("--policy-io", required=True)
    args = parser.parse_args()
    report = check_policy_io(args.policy_io)
    print(json.dumps({k: v for k, v in report.items() if k != "policy_io"}, indent=2, ensure_ascii=False))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
