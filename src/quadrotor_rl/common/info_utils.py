from __future__ import annotations

import math
from typing import Any, Dict, Iterable

import numpy as np
import torch


def safe_float(x: Any, default: float | None = None) -> float | None:
    try:
        if torch.is_tensor(x):
            value = float(x.detach().float().mean().cpu().item())
        elif isinstance(x, np.ndarray):
            value = float(np.mean(x))
        elif isinstance(x, (list, tuple)):
            if len(x) == 0:
                return default
            value = float(np.mean(x))
        elif isinstance(x, (int, float, np.integer, np.floating)):
            value = float(x)
        else:
            return default
    except Exception:
        return default

    if not math.isfinite(value):
        return default
    return value


def flatten_info(info: Dict[str, Any] | None, prefix: str = "") -> Dict[str, float]:
    """Flatten nested info dict into scalar metrics."""

    if not info:
        return {}

    out: Dict[str, float] = {}

    for key, value in info.items():
        name = f"{prefix}/{key}" if prefix else str(key)

        if isinstance(value, dict):
            out.update(flatten_info(value, name))
        else:
            scalar = safe_float(value)
            if scalar is not None:
                out[name] = scalar

    return out


def mean_dict(rows: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    """Mean-reduce a list of scalar dictionaries."""

    values: Dict[str, list[float]] = {}

    for row in rows:
        for key, value in row.items():
            scalar = safe_float(value)
            if scalar is None:
                continue
            values.setdefault(key, []).append(scalar)

    return {
        key: float(np.mean(vals))
        for key, vals in values.items()
        if len(vals) > 0
    }


def prefix_keys(data: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}/{key}": value for key, value in data.items()}
