from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class QuadrotorActionSemantics:
    """Canonical quadrotor action pipeline description.

    The policy always emits raw actions in [-1, 1]. The framework then applies
    optional dead-zone, scaling, exponential smoothing, and finally interprets
    the filtered value as a delta around the hover motor multiplier.
    """

    action_scale: float
    action_ema_alpha: float
    action_deadzone: float = 0.0
    min_action: float = -1.0
    max_action: float = 1.0

    def validate(self) -> None:
        assert self.action_scale > 0.0
        assert 0.0 < self.action_ema_alpha <= 1.0
        assert self.action_deadzone >= 0.0
        assert self.max_action > self.min_action


def sanitize_quadrotor_actions(actions: torch.Tensor, min_action: float = -1.0, max_action: float = 1.0) -> torch.Tensor:
    actions = torch.nan_to_num(actions, nan=0.0, posinf=float(max_action), neginf=float(min_action))
    return torch.clamp(actions, float(min_action), float(max_action))


def update_quadrotor_action_buffers(
    actions: torch.Tensor,
    previous_filtered_actions: torch.Tensor,
    semantics: QuadrotorActionSemantics,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the shared raw-action to filtered-action pipeline.

    Returns:
        raw_actions: sanitized raw policy output in [-1, 1].
        filtered_actions: scaled and smoothed motor-multiplier delta.
        prev_filtered_actions: previous scaled filtered action snapshot.
    """

    semantics.validate()
    raw_actions = sanitize_quadrotor_actions(
        actions,
        min_action=semantics.min_action,
        max_action=semantics.max_action,
    )

    if semantics.action_deadzone > 0.0:
        active_actions = torch.where(
            torch.abs(raw_actions) < float(semantics.action_deadzone),
            torch.zeros_like(raw_actions),
            raw_actions,
        )
    else:
        active_actions = raw_actions

    target_action = active_actions * float(semantics.action_scale)
    prev_filtered_actions = previous_filtered_actions.clone()
    alpha = float(semantics.action_ema_alpha)
    filtered_actions = alpha * target_action + (1.0 - alpha) * previous_filtered_actions
    filtered_actions = torch.nan_to_num(filtered_actions, nan=0.0, posinf=1.0, neginf=-1.0)
    return raw_actions.clone(), filtered_actions, prev_filtered_actions


def reset_quadrotor_action_buffers(
    raw_actions: torch.Tensor,
    filtered_actions: torch.Tensor,
    prev_filtered_actions: torch.Tensor,
    motor_multipliers: torch.Tensor,
    env_ids: torch.Tensor,
) -> None:
    raw_actions[env_ids] = 0.0
    filtered_actions[env_ids] = 0.0
    prev_filtered_actions[env_ids] = 0.0
    motor_multipliers[env_ids] = 1.0
