from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch


@dataclass(frozen=True)
class QuadrotorRotorLimits:
    min_motor_multiplier: float
    max_motor_multiplier: float
    max_total_thrust_factor: float
    max_body_moment_xy: float
    max_body_moment_z: float

    def validate(self) -> None:
        assert self.max_motor_multiplier > self.min_motor_multiplier
        assert self.max_total_thrust_factor > 1.0
        assert self.max_body_moment_xy > 0.0
        assert self.max_body_moment_z > 0.0


def compute_motor_multipliers(filtered_actions: torch.Tensor, limits: QuadrotorRotorLimits) -> torch.Tensor:
    limits.validate()
    motor_mult = 1.0 + filtered_actions
    return torch.clamp(
        motor_mult,
        float(limits.min_motor_multiplier),
        float(limits.max_motor_multiplier),
    )


def compute_quadrotor_wrench(
    filtered_actions: torch.Tensor,
    hover_thrust_per_rotor: torch.Tensor,
    hover_thrust: torch.Tensor,
    rotor_xy: torch.Tensor,
    rotor_yaw_signs: torch.Tensor,
    yaw_torque_per_newton: float,
    limits: QuadrotorRotorLimits,
    quat_wxyz: torch.Tensor,
    quat_rotate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert canonical filtered motor deltas to global force and torque."""

    motor_mult = compute_motor_multipliers(filtered_actions, limits)
    rotor_forces = hover_thrust_per_rotor.view(-1, 1) * motor_mult
    total_thrust = rotor_forces.sum(dim=-1)

    max_total = hover_thrust * float(limits.max_total_thrust_factor)
    total_thrust = torch.maximum(total_thrust, torch.zeros_like(total_thrust))
    total_thrust = torch.minimum(total_thrust, max_total)

    x = rotor_xy[:, 0].view(1, 4)
    y = rotor_xy[:, 1].view(1, 4)

    torque_x = torch.sum(y * rotor_forces, dim=-1)
    torque_y = -torch.sum(x * rotor_forces, dim=-1)
    torque_z = float(yaw_torque_per_newton) * torch.sum(
        rotor_yaw_signs.view(1, 4) * rotor_forces,
        dim=-1,
    )

    torque_b = torch.stack([torque_x, torque_y, torque_z], dim=-1)
    torque_b[:, 0:2] = torch.clamp(
        torque_b[:, 0:2],
        -float(limits.max_body_moment_xy),
        float(limits.max_body_moment_xy),
    )
    torque_b[:, 2] = torch.clamp(
        torque_b[:, 2],
        -float(limits.max_body_moment_z),
        float(limits.max_body_moment_z),
    )

    force_b = torch.zeros((filtered_actions.shape[0], 3), dtype=torch.float32, device=filtered_actions.device)
    force_b[:, 2] = total_thrust

    force_w = quat_rotate_fn(quat_wxyz, force_b)
    torque_w = quat_rotate_fn(quat_wxyz, torque_b)
    return force_w, torque_w, torque_b, motor_mult
