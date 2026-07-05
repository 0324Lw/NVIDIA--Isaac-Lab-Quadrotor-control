from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import torch


def to_float(value: Any):
    try:
        if torch.is_tensor(value):
            return float(value.detach().float().mean().cpu().item())
        if isinstance(value, np.ndarray):
            return float(np.mean(value))
        if isinstance(value, (list, tuple)):
            return float(np.mean(value)) if len(value) else None
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
    except Exception:
        return None
    return None


def flat_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in (data or {}).items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flat_dict(value, name))
            continue
        val = to_float(value)
        if val is not None and math.isfinite(val):
            out[name] = val
    return out


def write_scalars(writer, data: Dict[str, Any], step: int, prefix: str) -> None:
    if writer is None:
        return
    for key, value in (data or {}).items():
        val = to_float(value)
        if val is not None and math.isfinite(val):
            try:
                writer.add_scalar(f"{prefix}/{key}".replace("//", "/"), val, step)
            except Exception:
                pass


def make_table(title: str, data: Dict[str, Any], width: int = 124) -> str:
    lines = ["-" * width, f"| {title:<{width - 4}} |", "-" * width]
    if not data:
        lines += [f"| {'<empty>':<{width - 4}} |", "-" * width]
        return "\n".join(lines)

    key_width = max(60, min(87, width - 37))
    value_width = max(20, width - key_width - 7)
    for key in sorted(data.keys()):
        value = data[key]
        key_text = (str(key)[: key_width - 3] + "...") if len(str(key)) > key_width else str(key)
        if isinstance(value, float):
            if math.isnan(value):
                value_text = "nan"
            elif math.isinf(value):
                value_text = "inf"
            else:
                value_text = f"{value:.6e}" if abs(value) > 1e4 or 0 < abs(value) < 1e-3 else f"{value:.6f}"
        else:
            value_text = str(value)
        value_text = (value_text[: value_width - 3] + "...") if len(value_text) > value_width else value_text
        lines.append(f"| {key_text:<{key_width}} | {value_text:>{value_width}} |")
    lines.append("-" * width)
    return "\n".join(lines)


def tracking_mean(agent) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in getattr(agent, "tracking_data", {}).items():
        if value is None:
            continue
        try:
            if len(value) == 0:
                continue
        except Exception:
            pass
        try:
            arr = np.asarray(value, dtype=np.float64)
            if arr.size == 0:
                continue
            if key.endswith("(min)"):
                out[key] = float(np.min(arr))
            elif key.endswith("(max)"):
                out[key] = float(np.max(arr))
            else:
                out[key] = float(np.mean(arr))
        except Exception:
            val = to_float(value)
            if val is not None:
                out[key] = val
    return out


def current_lr(agent) -> float:
    for obj in [getattr(agent, "optimizer", None), getattr(getattr(agent, "scheduler", None), "optimizer", None)]:
        try:
            if obj is not None:
                return float(obj.param_groups[0]["lr"])
        except Exception:
            pass
    return float("nan")
