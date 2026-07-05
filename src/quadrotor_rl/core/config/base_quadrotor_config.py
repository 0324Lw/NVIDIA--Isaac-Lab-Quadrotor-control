from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Tuple


@dataclass
class BaseQuadrotorConfig:
    """Common quadrotor configuration contract shared by all tasks.

    Task configs may override any field, but every quadrotor task is expected
    to expose these fields so the control, export, evaluation, and sim2sim
    layers can read a stable policy interface.
    """

    num_envs: int = 512
    device: str = "cuda:0"
    seed: int = 42

    sim_dt: float = 0.005
    decimation: int = 4
    max_episode_length_s: float = 10.0

    spawn_height: float = 1.0
    env_spacing: float = 2.5
    crazyflie_usd_url: str = ""

    gravity: float = 9.81
    nominal_mass: float = 0.0282
    arm_length: float = 0.046
    rotor_xy_m1: Tuple[float, float] = (0.046, 0.046)
    rotor_xy_m2: Tuple[float, float] = (-0.046, 0.046)
    rotor_xy_m3: Tuple[float, float] = (-0.046, -0.046)
    rotor_xy_m4: Tuple[float, float] = (0.046, -0.046)
    rotor_yaw_signs: Tuple[float, float, float, float] = (1.0, -1.0, 1.0, -1.0)
    yaw_torque_per_newton: float = 0.006

    action_dim: int = 4
    action_scale: float = 0.5
    action_ema_alpha: float = 0.5
    min_motor_multiplier: float = 0.0
    max_motor_multiplier: float = 2.2
    max_total_thrust_factor: float = 2.4
    max_body_moment_xy: float = 0.025
    max_body_moment_z: float = 0.006

    frame_stack: int = 4
    obs_clip: float = 10.0
    priv_clip: float = 20.0
    print_debug_info: bool = True

    @property
    def policy_dt(self) -> float:
        return float(self.sim_dt * self.decimation)

    @property
    def max_episode_length(self) -> int:
        return int(self.max_episode_length_s / max(self.policy_dt, 1.0e-6))

    @property
    def hover_thrust(self) -> float:
        return float(self.nominal_mass * self.gravity)

    @property
    def hover_thrust_per_rotor(self) -> float:
        return float(self.hover_thrust / 4.0)

    @property
    def rotor_xy(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        return (
            self.rotor_xy_m1,
            self.rotor_xy_m2,
            self.rotor_xy_m3,
            self.rotor_xy_m4,
        )

    def validate_common(self) -> None:
        assert self.num_envs > 0, f"num_envs must be positive, got {self.num_envs}"
        assert self.device, "device must not be empty"
        assert self.sim_dt > 0.0
        assert self.decimation >= 1
        assert self.policy_dt > 0.0
        assert self.max_episode_length_s > 0.0
        assert self.max_episode_length >= 1
        assert self.spawn_height > 0.0
        assert self.env_spacing > 0.0
        assert self.gravity > 0.0
        assert self.nominal_mass > 0.0
        assert self.hover_thrust > 0.0
        assert self.hover_thrust_per_rotor > 0.0
        assert self.arm_length > 0.0
        assert self.action_dim == 4
        assert 0.0 < self.action_ema_alpha <= 1.0
        assert self.action_scale > 0.0
        assert self.max_motor_multiplier > self.min_motor_multiplier
        assert self.max_total_thrust_factor > 1.0
        assert self.max_body_moment_xy > 0.0
        assert self.max_body_moment_z > 0.0
        assert self.frame_stack >= 1
        assert self.obs_clip > 0.0
        assert self.priv_clip > 0.0

    def to_policy_io_dict(self) -> Dict[str, Any]:
        return {
            "num_envs": int(self.num_envs),
            "sim_dt": float(self.sim_dt),
            "decimation": int(self.decimation),
            "policy_dt": float(self.policy_dt),
            "max_episode_length_s": float(self.max_episode_length_s),
            "max_episode_length": int(self.max_episode_length),
            "action_dim": int(self.action_dim),
            "action_scale": float(self.action_scale),
            "action_ema_alpha": float(self.action_ema_alpha),
            "min_motor_multiplier": float(self.min_motor_multiplier),
            "max_motor_multiplier": float(self.max_motor_multiplier),
            "max_total_thrust_factor": float(self.max_total_thrust_factor),
            "max_body_moment_xy": float(self.max_body_moment_xy),
            "max_body_moment_z": float(self.max_body_moment_z),
            "gravity": float(self.gravity),
            "nominal_mass": float(self.nominal_mass),
            "arm_length": float(self.arm_length),
            "rotor_xy": [list(x) for x in self.rotor_xy],
            "rotor_yaw_signs": list(self.rotor_yaw_signs),
            "yaw_torque_per_newton": float(self.yaw_torque_per_newton),
            "obs_clip": float(self.obs_clip),
            "priv_clip": float(self.priv_clip),
            "frame_stack": int(self.frame_stack),
        }

    def public_config_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for item in fields(self):
            name = item.name
            value = getattr(self, name)
            if isinstance(value, tuple):
                value = list(value)
            out[name] = value
        return out
