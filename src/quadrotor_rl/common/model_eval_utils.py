from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def resolve_checkpoint_path(path: str, preferred_names: tuple[str, ...] = ()) -> Path:
    """Resolve file or checkpoint directory."""

    p = Path(path).expanduser().resolve()

    if p.is_file():
        return p

    if p.is_dir():
        for name in preferred_names:
            candidate = p / name
            if candidate.exists():
                return candidate

        for name in [
            "quadrotor_task4_model.pt",
            "quadrotor_task3_model.pt",
            "quadrotor_task2_model.pt",
            "quadrotor_task1_model.pt",
            "model.pt",
            "checkpoint.pt",
        ]:
            candidate = p / name
            if candidate.exists():
                return candidate

        final_dir = p / "final_checkpoint"
        if final_dir.exists():
            return resolve_checkpoint_path(str(final_dir), preferred_names=preferred_names)

    return p


def torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def find_norm_tensors(norm_state: Dict[str, Any] | None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not isinstance(norm_state, dict):
        return None, None

    mean_keys = ("running_mean", "_running_mean", "mean", "_mean", "obs_mean")
    var_keys = ("running_variance", "_running_variance", "variance", "_variance", "var", "_var", "obs_var")

    mean = next((norm_state[k] for k in mean_keys if k in norm_state), None)
    var = next((norm_state[k] for k in var_keys if k in norm_state), None)

    if mean is not None and var is not None:
        return mean, var

    for value in norm_state.values():
        if isinstance(value, dict):
            m, v = find_norm_tensors(value)
            if m is not None and v is not None:
                return m, v

    return None, None


def normalize_with_state(obs: torch.Tensor, norm_state: Dict[str, Any] | None, clip: float = 10.0) -> torch.Tensor:
    if not norm_state:
        return obs

    mean, var = find_norm_tensors(norm_state)
    if mean is None or var is None:
        return obs

    mean = torch.as_tensor(mean, dtype=torch.float32, device=obs.device).view(-1)
    var = torch.as_tensor(var, dtype=torch.float32, device=obs.device).view(-1)

    if mean.numel() != obs.shape[-1] or var.numel() != obs.shape[-1]:
        return obs

    return torch.clamp((obs - mean) / torch.sqrt(var + 1.0e-8), -float(clip), float(clip))


def require_metadata_flag(ckpt: Dict[str, Any], key: str, expected: Any = True) -> None:
    metadata = ckpt.get("metadata", {}) if isinstance(ckpt, dict) else {}
    value = metadata.get(key, None)
    if value != expected:
        raise RuntimeError(f"checkpoint metadata mismatch: {key}={value}, expected {expected}")
