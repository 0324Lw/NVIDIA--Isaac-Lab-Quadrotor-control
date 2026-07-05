from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch") and hasattr(value, "detach") and hasattr(value, "cpu")


def _is_numpy_scalar(value: Any) -> bool:
    return value.__class__.__module__.startswith("numpy") and hasattr(value, "item")


def _is_numpy_array(value: Any) -> bool:
    return value.__class__.__module__.startswith("numpy") and hasattr(value, "tolist")


def _json_safe(value: Any) -> Any:
    """Convert metadata to JSON-safe Python objects without keeping device tensors alive."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value

    if isinstance(value, Path):
        return str(value)

    if _is_torch_tensor(value):
        tensor = value.detach().cpu()
        if tensor.numel() == 1:
            return tensor.item()
        return tensor.tolist()

    if _is_numpy_scalar(value):
        return value.item()

    if _is_numpy_array(value):
        return value.tolist()

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return _json_safe(dataclasses.asdict(value))
        except Exception:
            return str(value)

    if isinstance(value, dict):
        return {str(_json_safe(key)): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    if hasattr(value, "__dict__"):
        try:
            return _json_safe(vars(value))
        except Exception:
            return str(value)

    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(_json_safe(value))
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(_safe_float(value, default))
    except Exception:
        return int(default)


def _safe_tensor_mean(value: Any, default: float = 0.0) -> float:
    try:
        if hasattr(value, "mean"):
            return float(value.mean().detach().cpu().item())
        return float(value)
    except Exception:
        return float(default)


def build_policy_io_metadata(
    task_name: str,
    env_cfg: Any,
    base_env: Any,
    env: Any,
    args: Any | None = None,
    env_steps: int = 0,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    try:
        env_cfg_dict = env_cfg.to_policy_io_dict()
    except Exception:
        try:
            env_cfg_dict = dataclasses.asdict(env_cfg)
        except Exception:
            env_cfg_dict = {}

    action_deadzone = getattr(env_cfg, "action_deadzone", 0.0)
    drone = getattr(base_env, "drone", None)
    metadata: Dict[str, Any] = {
        "task_name": str(task_name),
        "global_env_steps": int(env_steps),
        "num_envs": _safe_int(getattr(base_env, "num_envs", 0)),
        "actor_obs_dim": _safe_int(env.observation_space.shape[0]),
        "critic_obs_dim": _safe_int(env.state_space.shape[0]),
        "action_dim": _safe_int(env.action_space.shape[0]),
        "control_dt": _safe_float(getattr(env_cfg, "policy_dt", 0.0)),
        "sim_dt": _safe_float(getattr(env_cfg, "sim_dt", 0.0)),
        "decimation": _safe_int(getattr(env_cfg, "decimation", 1)),
        "frame_stack": _safe_int(getattr(env_cfg, "frame_stack", 1)),
        "action_semantics": {
            "raw_action_range": [-1.0, 1.0],
            "pipeline": ["clip", "deadzone", "scale", "ema", "motor_multiplier_delta", "wrench"],
            "action_scale": _safe_float(getattr(env_cfg, "action_scale", 0.0)),
            "action_ema_alpha": _safe_float(getattr(env_cfg, "action_ema_alpha", 1.0)),
            "action_deadzone": _safe_float(action_deadzone),
            "min_motor_multiplier": _safe_float(getattr(env_cfg, "min_motor_multiplier", 0.0)),
            "max_motor_multiplier": _safe_float(getattr(env_cfg, "max_motor_multiplier", 0.0)),
        },
        "rotor_model": {
            "gravity": _safe_float(getattr(env_cfg, "gravity", 0.0)),
            "nominal_mass": _safe_float(getattr(env_cfg, "nominal_mass", 0.0)),
            "estimated_mass": _safe_tensor_mean(getattr(base_env, "estimated_mass", 0.0)),
            "hover_thrust": _safe_tensor_mean(getattr(base_env, "hover_thrust", 0.0)),
            "arm_length": _safe_float(getattr(env_cfg, "arm_length", 0.0)),
            "rotor_xy": _json_safe(getattr(env_cfg, "rotor_xy", [])),
            "rotor_yaw_signs": _json_safe(getattr(env_cfg, "rotor_yaw_signs", [])),
            "yaw_torque_per_newton": _safe_float(getattr(env_cfg, "yaw_torque_per_newton", 0.0)),
            "max_total_thrust_factor": _safe_float(getattr(env_cfg, "max_total_thrust_factor", 0.0)),
            "max_body_moment_xy": _safe_float(getattr(env_cfg, "max_body_moment_xy", 0.0)),
            "max_body_moment_z": _safe_float(getattr(env_cfg, "max_body_moment_z", 0.0)),
        },
        "normalizer": {
            "observation_preprocessor": "_observation_preprocessor.pt",
            "state_preprocessor": "_state_preprocessor.pt",
            "value_preprocessor": "_value_preprocessor.pt",
            "normalizer_required": True,
            "onnx_input_convention": "raw_observation",
            "normalizer_embedding_supported": True,
        },
        "asset": {
            "asset_source": str(getattr(base_env, "asset_source", "unknown")),
            "num_bodies": _safe_int(getattr(drone, "num_bodies", -1)),
            "num_joints": _safe_int(getattr(drone, "num_joints", -1)),
            "body_names": _json_safe(getattr(drone, "body_names", [])),
            "joint_names": _json_safe(getattr(drone, "joint_names", [])),
        },
        "env_cfg": env_cfg_dict,
        "args": vars(args) if args is not None and hasattr(args, "__dict__") else {},
        "extra": extra or {},
    }

    for optional_key in [
        "single_actor_obs_dim",
        "obs_dim_per_frame",
        "depth_dim",
        "compact_state_dim",
        "depth_channels",
        "cam_res_h",
        "cam_res_w",
        "num_gates",
    ]:
        if hasattr(env_cfg, optional_key):
            metadata[optional_key] = _safe_int(getattr(env_cfg, optional_key))
    return _json_safe(metadata)


def save_policy_io(
    save_dir: str | Path,
    task_name: str,
    env_cfg: Any,
    base_env: Any,
    env: Any,
    args: Any | None = None,
    env_steps: int = 0,
    extra: Dict[str, Any] | None = None,
) -> Path:
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    metadata = build_policy_io_metadata(task_name, env_cfg, base_env, env, args=args, env_steps=env_steps, extra=extra)
    out_path = path / "policy_io.json"
    out_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def load_policy_io(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        path = path / "policy_io.json"
    return json.loads(path.read_text(encoding="utf-8"))
