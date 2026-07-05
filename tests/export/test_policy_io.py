from __future__ import annotations

import json
from pathlib import Path

from quadrotor_rl.export.check_policy_io import check_policy_io


def test_policy_io_checker_accepts_minimal_valid_file(tmp_path: Path):
    policy_io = {
        "task_name": "quadrotor_task1_hover_stabilization",
        "actor_obs_dim": 108,
        "critic_obs_dim": 108,
        "action_dim": 4,
        "control_dt": 0.02,
        "action_semantics": {
            "pipeline": ["clip", "deadzone", "scale", "ema", "motor_multiplier_delta", "wrench"],
            "action_scale": 0.5,
            "action_ema_alpha": 0.5,
        },
        "rotor_model": {},
        "normalizer": {},
    }
    path = tmp_path / "policy_io.json"
    path.write_text(json.dumps(policy_io), encoding="utf-8")
    report = check_policy_io(path)
    assert report["ok"]


def test_policy_io_metadata_is_json_safe_with_tensor_extra(tmp_path: Path):
    import torch
    from types import SimpleNamespace
    from quadrotor_rl.export.policy_io import save_policy_io

    cfg = SimpleNamespace(
        policy_dt=0.02,
        sim_dt=0.005,
        decimation=4,
        frame_stack=4,
        action_scale=0.25,
        action_ema_alpha=0.8,
        action_deadzone=0.0,
        min_motor_multiplier=0.0,
        max_motor_multiplier=2.0,
        gravity=9.81,
        nominal_mass=0.034,
        arm_length=0.046,
        rotor_xy=((0.1, 0.1), (-0.1, 0.1)),
        rotor_yaw_signs=(1.0, -1.0),
        yaw_torque_per_newton=0.01,
        max_total_thrust_factor=3.0,
        max_body_moment_xy=1.0,
        max_body_moment_z=1.0,
    )
    drone = SimpleNamespace(num_bodies=1, num_joints=0, body_names=["base"], joint_names=[])
    base_env = SimpleNamespace(
        num_envs=2,
        estimated_mass=torch.tensor([0.034, 0.035]),
        hover_thrust=torch.tensor([0.083, 0.084]),
        asset_source="test_asset",
        drone=drone,
    )
    space = SimpleNamespace(shape=(4,))
    env = SimpleNamespace(observation_space=space, state_space=space, action_space=space)
    args = SimpleNamespace(num_envs=2, device="cuda:0")

    out = save_policy_io(
        tmp_path,
        task_name="quadrotor_task_test",
        env_cfg=cfg,
        base_env=base_env,
        env=env,
        args=args,
        env_steps=128,
        extra={"last_info": {"reward_components": {"Total": torch.tensor(1.25)}}},
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["extra"]["last_info"]["reward_components"]["Total"] == 1.25
    assert isinstance(data["rotor_model"]["estimated_mass"], float)
