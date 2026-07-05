from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn

from quadrotor_rl.training.logging_utils import to_float


def sanitize_tensor_inplace(x: torch.Tensor, nan=0.0, posinf=1.0, neginf=-1.0, clamp_abs=None) -> None:
    if x is None or not torch.is_tensor(x):
        return
    with torch.no_grad():
        x.data = torch.nan_to_num(x.data, nan=nan, posinf=posinf, neginf=neginf)
        if clamp_abs is not None:
            x.data.clamp_(-float(clamp_abs), float(clamp_abs))


def sanitize_agent_numerics(agent, models: Dict[str, nn.Module], init_log_std: float, min_log_std: float, max_log_std: float) -> None:
    for _, model in models.items():
        for parameter in model.parameters():
            sanitize_tensor_inplace(parameter, nan=0.0, posinf=1.0, neginf=-1.0, clamp_abs=20.0)
        if hasattr(model, "log_std_parameter"):
            with torch.no_grad():
                model.log_std_parameter.data = torch.nan_to_num(
                    model.log_std_parameter.data,
                    nan=float(init_log_std),
                    posinf=float(max_log_std),
                    neginf=float(min_log_std),
                )
                model.log_std_parameter.data.clamp_(float(min_log_std), float(max_log_std))

    optimizer = getattr(agent, "optimizer", None)
    if optimizer is not None:
        for state in optimizer.state.values():
            for _, value in state.items():
                if torch.is_tensor(value):
                    with torch.no_grad():
                        value.data = torch.nan_to_num(value.data, nan=0.0, posinf=1.0, neginf=-1.0)
                        value.data.clamp_(-100.0, 100.0)


def ppo_info_has_nan(ppo_info: Dict[str, Any]) -> tuple[bool, str]:
    keys_to_check = [
        "Loss / Entropy loss",
        "Loss / Policy loss",
        "Loss / Value loss",
        "Policy / Standard deviation",
        "Learning / Learning rate",
        "learning_rate",
    ]
    for key in keys_to_check:
        if key in ppo_info:
            val = to_float(ppo_info[key])
            if val is not None and not math.isfinite(val):
                return True, key
    for key, value in ppo_info.items():
        if "Loss" in key or "Standard deviation" in key or "Learning" in key:
            val = to_float(value)
            if val is not None and not math.isfinite(val):
                return True, key
    return False, ""
