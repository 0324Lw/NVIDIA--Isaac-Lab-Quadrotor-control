from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Task2Config:
    """Quadrotor / Crazyflie Task2: 3D trajectory tracking.

    Task logic:
        - Generate a smooth random 3D trajectory at reset.
        - Track the nearest forward waypoint rather than a fixed time index.
        - Provide 5 lookahead relative waypoint vectors.
        - Use 4-frame observation stacking.
        - Control through 4 normalized rotor-thrust corrections.
        - Reward path tracking, tangent velocity alignment, heading alignment,
          smooth control, and terminal success / crash / deviation events.

    Isaac Lab implementation:
        - Asset: Bitcraze Crazyflie 2.X USD.
        - Control: external root force and torque from rotor thrusts.
        - Observation: 100 dimensions = 4 frames × 25 features.
    """

    # ------------------------------------------------------------------
    # Basic
    # ------------------------------------------------------------------
    num_envs: int = 512
    device: str = "cuda:0"
    seed: int = 42

    sim_dt: float = 0.005
    decimation: int = 4
    max_episode_length_s: float = 16.667

    # ------------------------------------------------------------------
    # Asset
    # ------------------------------------------------------------------
    spawn_height: float = 1.2
    env_spacing: float = 8.0

    crazyflie_usd_url: str = (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
        "Assets/Isaac/5.1/Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd"
    )

    # ------------------------------------------------------------------
    # Physical constants / nominal model
    # ------------------------------------------------------------------
    gravity: float = 9.81
    nominal_mass: float = 0.0282

    arm_length: float = 0.046
    rotor_xy_m1: Tuple[float, float] = (0.046, 0.046)
    rotor_xy_m2: Tuple[float, float] = (-0.046, 0.046)
    rotor_xy_m3: Tuple[float, float] = (-0.046, -0.046)
    rotor_xy_m4: Tuple[float, float] = (0.046, -0.046)
    rotor_yaw_signs: Tuple[float, float, float, float] = (1.0, -1.0, 1.0, -1.0)
    yaw_torque_per_newton: float = 0.006

    # ------------------------------------------------------------------
    # Trajectory generation
    # ------------------------------------------------------------------
    trajectory_num_points_factor: int = 2

    amp_x_range: Tuple[float, float] = (1.5, 3.0)
    amp_y_range: Tuple[float, float] = (1.5, 3.0)
    amp_z_range: Tuple[float, float] = (0.1, 0.4)

    freq_x_range: Tuple[float, float] = (0.02, 0.08)
    freq_y_range: Tuple[float, float] = (0.02, 0.08)
    freq_z_range: Tuple[float, float] = (0.01, 0.05)

    base_altitude: float = 1.2
    min_trajectory_z: float = 0.80
    max_trajectory_z: float = 1.80

    # Forward nearest-point search window.
    target_search_range: int = 30

    # ------------------------------------------------------------------
    # Action model
    # ------------------------------------------------------------------
    action_dim: int = 4
    action_scale: float = 0.50
    action_ema_alpha: float = 0.50

    min_motor_multiplier: float = 0.0
    max_motor_multiplier: float = 2.2

    max_total_thrust_factor: float = 2.4
    max_body_moment_xy: float = 0.025
    max_body_moment_z: float = 0.006

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    frame_stack: int = 4
    lookahead_steps: int = 5
    lookahead_interval: int = 10

    # Single-frame observation:
    #   roll / pitch / yaw             3
    #   filtered action                4
    #   current relative target        3
    #   5 lookahead relative targets   15
    # total                            25
    obs_dim_per_frame: int = 25

    obs_clip: float = 10.0
    priv_clip: float = 20.0

    rpy_scale: float = 1.5
    rel_pos_scale: float = 4.0
    lookahead_rel_pos_scale: float = 4.0

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    r_survival: float = 0.10
    r_track_sigma: float = 1.0
    r_vel_coef: float = 0.15
    r_heading_coef: float = 0.05
    r_smooth_coef: float = -0.05

    # Terminal event rewards.
    r_crash: float = -20.0
    r_deviate: float = -20.0
    r_success_base: float = 50.0
    time_bonus_coef: float = 0.10

    continuous_reward_clip_min: float = -1.0
    continuous_reward_clip_max: float = 1.0
    final_reward_clip_min: float = -30.0
    final_reward_clip_max: float = 120.0

    # Weighted Z error for path tracking reward.
    z_error_weight: float = 2.5

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------
    max_dev_err: float = 2.0
    max_roll_pitch: float = 1.05
    min_z: float = 0.05
    max_z: float = 3.0

    # Success means the target cursor reaches the end region.
    success_end_margin: int = 8

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------
    print_debug_info: bool = True

    @property
    def policy_dt(self) -> float:
        return float(self.sim_dt * self.decimation)

    @property
    def max_episode_length(self) -> int:
        return int(self.max_episode_length_s / max(self.policy_dt, 1.0e-6))

    @property
    def trajectory_num_points(self) -> int:
        return int(self.max_episode_length * self.trajectory_num_points_factor)

    @property
    def single_actor_obs_dim(self) -> int:
        return int(self.obs_dim_per_frame)

    @property
    def actor_obs_dim(self) -> int:
        return int(self.obs_dim_per_frame * self.frame_stack)

    @property
    def critic_obs_dim(self) -> int:
        return int(self.actor_obs_dim)

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

    def validate(self) -> None:
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
        assert self.yaw_torque_per_newton >= 0.0

        assert self.trajectory_num_points_factor >= 1
        assert self.trajectory_num_points >= self.max_episode_length
        assert self.target_search_range >= 1

        for name in [
            "amp_x_range",
            "amp_y_range",
            "amp_z_range",
            "freq_x_range",
            "freq_y_range",
            "freq_z_range",
        ]:
            lo, hi = getattr(self, name)
            assert hi >= lo, f"{name} invalid: {getattr(self, name)}"

        assert self.base_altitude > 0.0
        assert self.min_trajectory_z > 0.0
        assert self.max_trajectory_z > self.min_trajectory_z

        assert self.action_dim == 4
        assert 0.0 < self.action_ema_alpha <= 1.0
        assert self.action_scale > 0.0
        assert self.max_motor_multiplier > self.min_motor_multiplier
        assert self.max_total_thrust_factor > 1.0
        assert self.max_body_moment_xy > 0.0
        assert self.max_body_moment_z > 0.0

        assert self.frame_stack == 4
        assert self.lookahead_steps == 5
        assert self.lookahead_interval >= 1
        assert self.obs_dim_per_frame == 25
        assert self.actor_obs_dim == 100
        assert self.critic_obs_dim == 100

        for name in [
            "obs_clip",
            "priv_clip",
            "rpy_scale",
            "rel_pos_scale",
            "lookahead_rel_pos_scale",
            "r_track_sigma",
            "z_error_weight",
        ]:
            assert float(getattr(self, name)) > 0.0, f"{name} must be positive"

        assert self.continuous_reward_clip_min < self.continuous_reward_clip_max
        assert self.final_reward_clip_min < self.final_reward_clip_max

        assert self.max_dev_err > 0.0
        assert self.max_roll_pitch > 0.0
        assert self.min_z >= 0.0
        assert self.max_z > self.min_z
        assert self.success_end_margin >= self.lookahead_steps


QuadrotorTask2Config = Task2Config
CrazyflieTask2Config = Task2Config
