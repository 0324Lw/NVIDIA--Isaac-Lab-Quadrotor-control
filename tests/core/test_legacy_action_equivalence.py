from __future__ import annotations

import torch

from quadrotor_rl.core.physics.action_semantics import (
    QuadrotorActionSemantics,
    sanitize_quadrotor_actions,
    update_quadrotor_action_buffers,
)


def test_task1_raw_space_filter_keeps_observation_semantics():
    raw_actions = torch.tensor([[1.0, -1.0, 0.5, -0.5]], dtype=torch.float32)
    previous_raw_filtered = torch.tensor([[0.2, -0.2, 0.1, -0.1]], dtype=torch.float32)
    action_scale = 0.65
    alpha = 0.7

    sanitized = sanitize_quadrotor_actions(raw_actions)
    next_raw_filtered = alpha * sanitized + (1.0 - alpha) * previous_raw_filtered
    motor_delta = action_scale * next_raw_filtered

    assert torch.allclose(next_raw_filtered, torch.tensor([[0.76, -0.76, 0.38, -0.38]]), atol=1.0e-6)
    assert torch.allclose(motor_delta, torch.tensor([[0.494, -0.494, 0.247, -0.247]]), atol=1.0e-6)


def test_task2_to_task4_scaled_filter_matches_previous_motor_delta_logic():
    raw_actions = torch.tensor([[1.0, -2.0, float("nan"), float("inf")]], dtype=torch.float32)
    previous_motor_delta = torch.tensor([[0.2, -0.2, 0.1, -0.1]], dtype=torch.float32)
    semantics = QuadrotorActionSemantics(action_scale=0.5, action_ema_alpha=0.25)

    raw, filtered_delta, old_delta = update_quadrotor_action_buffers(
        actions=raw_actions,
        previous_filtered_actions=previous_motor_delta,
        semantics=semantics,
    )

    expected_raw = torch.tensor([[1.0, -1.0, 0.0, 1.0]], dtype=torch.float32)
    expected_delta = 0.25 * (expected_raw * 0.5) + 0.75 * previous_motor_delta
    assert torch.allclose(raw, expected_raw)
    assert torch.allclose(old_delta, previous_motor_delta)
    assert torch.allclose(filtered_delta, expected_delta)
