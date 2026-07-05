from __future__ import annotations

import torch

from quadrotor_rl.core.math.quat_math import quat_rotate, quat_rotate_inverse


def world_to_body(quat_wxyz: torch.Tensor, vector_w: torch.Tensor) -> torch.Tensor:
    return quat_rotate_inverse(quat_wxyz, vector_w)


def body_to_world(quat_wxyz: torch.Tensor, vector_b: torch.Tensor) -> torch.Tensor:
    return quat_rotate(quat_wxyz, vector_b)
