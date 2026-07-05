from __future__ import annotations

from typing import Tuple

import torch


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def euler_to_quat_wxyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr = torch.cos(0.5 * roll)
    sr = torch.sin(0.5 * roll)
    cp = torch.cos(0.5 * pitch)
    sp = torch.sin(0.5 * pitch)
    cy = torch.cos(0.5 * yaw)
    sy = torch.sin(0.5 * yaw)

    q = torch.zeros((roll.shape[0], 4), dtype=torch.float32, device=roll.device)
    q[:, 0] = cr * cp * cy + sr * sp * sy
    q[:, 1] = sr * cp * cy - cr * sp * sy
    q[:, 2] = cr * sp * cy + sr * cp * sy
    q[:, 3] = cr * cp * sy - sr * sp * cy
    return q


def quat_to_euler_wxyz(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1.0,
        torch.sign(sinp) * (torch.pi / 2.0),
        torch.asin(sinp),
    )

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_w = q[:, 0:1]
    q_vec = q[:, 1:4]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q_w * t + torch.cross(q_vec, t, dim=-1)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_inv = q.clone()
    q_inv[:, 1:4] *= -1.0
    return quat_rotate(q_inv, v)
