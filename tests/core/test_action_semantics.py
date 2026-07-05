from __future__ import annotations

import torch

from quadrotor_rl.core.physics.action_semantics import QuadrotorActionSemantics, update_quadrotor_action_buffers
from quadrotor_rl.core.physics.rotor_model import QuadrotorRotorLimits, compute_quadrotor_wrench
from quadrotor_rl.core.math.quat_math import quat_rotate


def test_action_semantics_scales_before_filtering():
    actions = torch.tensor([[1.0, -1.0, 0.5, -0.5]], dtype=torch.float32)
    prev = torch.zeros_like(actions)
    raw, filtered, old = update_quadrotor_action_buffers(
        actions=actions,
        previous_filtered_actions=prev,
        semantics=QuadrotorActionSemantics(action_scale=0.5, action_ema_alpha=0.5),
    )
    assert torch.allclose(raw, actions)
    assert torch.allclose(old, prev)
    assert torch.allclose(filtered, torch.tensor([[0.25, -0.25, 0.125, -0.125]]))


def test_rotor_wrench_returns_expected_shapes():
    filtered = torch.zeros((2, 4), dtype=torch.float32)
    hover_per = torch.full((2,), 0.25, dtype=torch.float32)
    hover = torch.full((2,), 1.0, dtype=torch.float32)
    rotor_xy = torch.tensor([[0.05, 0.05], [-0.05, 0.05], [-0.05, -0.05], [0.05, -0.05]], dtype=torch.float32)
    yaw_signs = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float32)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    limits = QuadrotorRotorLimits(0.0, 2.0, 2.4, 0.1, 0.05)
    force_w, torque_w, torque_b, motor = compute_quadrotor_wrench(
        filtered, hover_per, hover, rotor_xy, yaw_signs, 0.006, limits, quat, quat_rotate
    )
    assert force_w.shape == (2, 3)
    assert torque_w.shape == (2, 3)
    assert torque_b.shape == (2, 3)
    assert motor.shape == (2, 4)
    assert torch.allclose(force_w[:, 2], hover)
