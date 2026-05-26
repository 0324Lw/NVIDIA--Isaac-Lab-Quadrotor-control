from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Task3Config:
    """Quadrotor / Crazyflie Task3: dynamic obstacle navigation world config.

    This file only defines configuration for the analytic world layer.

    World logic:
        - 50m x 50m square arena.
        - Random start / goal with 7m to 10m separation.
        - Safe zones around start and goal.
        - 30 static cylindrical obstacles.
        - 4 moving cylindrical obstacles.
        - 24-ray 2D LiDAR with 10m range.
        - Torch-native batched obstacle / LiDAR / collision calculations.

    Isaac Lab policy:
        Environment code will use this world together with the Crazyflie
        articulation and root-wrench control, but the world itself is analytic
        and does not create heavy obstacle prims.
    """

    # ------------------------------------------------------------------
    # Basic
    # ------------------------------------------------------------------
    num_envs: int = 512
    device: str = "cuda:0"
    seed: int = 42

    sim_dt: float = 0.005
    decimation: int = 4
    max_episode_length_s: float = 20.0

    # ------------------------------------------------------------------
    # Asset
    # ------------------------------------------------------------------
    spawn_height: float = 1.0
    env_spacing: float = 60.0

    crazyflie_usd_url: str = (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
        "Assets/Isaac/5.1/Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd"
    )

    # ------------------------------------------------------------------
    # Quadrotor physical constants
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
    # World map
    # ------------------------------------------------------------------
    arena_size: float = 50.0
    wall_height: float = 3.0

    min_start_goal_dist: float = 7.0
    max_start_goal_dist: float = 10.0
    safe_zone_radius: float = 5.0

    start_goal_z: float = 1.0

    # ------------------------------------------------------------------
    # Static obstacles
    # ------------------------------------------------------------------
    num_static_obs: int = 30
    static_radius_min: float = 0.5
    static_radius_max: float = 1.5
    obs_height: float = 3.0
    min_obs_gap: float = 2.0

    # ------------------------------------------------------------------
    # Dynamic obstacles
    # ------------------------------------------------------------------
    num_dynamic_obs: int = 4
    dynamic_radius: float = 1.0
    dynamic_speed: float = 1.5
    dynamic_spawn_line_t_min: float = 0.15
    dynamic_spawn_line_t_max: float = 0.85
    dynamic_lateral_spread: float = 5.0

    # ------------------------------------------------------------------
    # LiDAR
    # ------------------------------------------------------------------
    lidar_num_rays: int = 24
    lidar_max_range: float = 10.0
    lidar_z_offset: float = 0.0
    lidar_start_offset: float = 0.15

    # ------------------------------------------------------------------
    # Collision / navigation
    # ------------------------------------------------------------------
    robot_radius: float = 0.18
    success_radius: float = 0.60

    min_flight_z: float = 0.10
    max_flight_z: float = 2.80
    max_roll_pitch: float = 1.05

    # ------------------------------------------------------------------
    # Action / control
    # ------------------------------------------------------------------
    action_scale: float = 0.50
    action_ema_alpha: float = 0.50

    min_motor_multiplier: float = 0.0
    max_motor_multiplier: float = 2.2

    max_total_thrust_factor: float = 2.4
    max_body_moment_xy: float = 0.025
    max_body_moment_z: float = 0.006

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    r_step: float = -0.03
    r_height_scale: float = 0.25
    r_height_k: float = 5.0
    r_att_penalty: float = -0.10
    r_smooth_penalty: float = -0.002
    r_approach: float = 0.50
    r_dir: float = 0.10
    r_repulsion_max: float = -0.40

    r_crash: float = -50.0
    r_deviate: float = -50.0
    r_success_base: float = 50.0
    time_bonus_coef: float = 0.10

    safe_lidar_dist: float = 0.25
    max_z_err: float = 0.80
    success_xy_tolerance: float = 0.40
    success_z_tolerance: float = 0.40

    continuous_reward_clip_min: float = -1.0
    continuous_reward_clip_max: float = 1.0
    final_reward_clip_min: float = -80.0
    final_reward_clip_max: float = 160.0

    # ------------------------------------------------------------------
    # Observation planning for later Task3 env
    # ------------------------------------------------------------------
    action_dim: int = 4

    # Planned single-frame observation layout:
    #   relative goal in body frame      3
    #   distance to goal                 1
    #   heading sin/cos                  2
    #   linear velocity body             3
    #   angular velocity body            3
    #   projected gravity body           3
    #   last filtered action             4
    #   lidar                            24
    #   lidar delta                      24
    #   risk features                    8
    # total                              75
    single_actor_obs_dim: int = 75
    frame_stack: int = 4

    obs_clip: float = 10.0
    priv_clip: float = 20.0

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
    def actor_obs_dim(self) -> int:
        return int(self.single_actor_obs_dim * self.frame_stack)

    @property
    def critic_obs_dim(self) -> int:
        return int(self.actor_obs_dim)

    @property
    def arena_half(self) -> float:
        return float(self.arena_size / 2.0)

    @property
    def start_goal_bound(self) -> float:
        return float(self.arena_half - self.safe_zone_radius)

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
        assert self.env_spacing > self.arena_size

        assert self.gravity > 0.0
        assert self.nominal_mass > 0.0
        assert self.hover_thrust > 0.0
        assert self.hover_thrust_per_rotor > 0.0
        assert self.arm_length > 0.0
        assert self.yaw_torque_per_newton >= 0.0

        assert self.arena_size > 0.0
        assert self.wall_height > 0.0
        assert self.arena_half > 0.0
        assert self.safe_zone_radius > 0.0
        assert self.start_goal_bound > 0.0

        assert 0.0 < self.min_start_goal_dist <= self.max_start_goal_dist
        assert self.max_start_goal_dist < self.arena_size
        assert self.start_goal_z > 0.0

        assert self.num_static_obs >= 0
        assert 0.0 < self.static_radius_min <= self.static_radius_max
        assert self.obs_height > 0.0
        assert self.min_obs_gap >= 0.0

        assert self.num_dynamic_obs >= 0
        assert self.dynamic_radius > 0.0
        assert self.dynamic_speed > 0.0
        assert 0.0 <= self.dynamic_spawn_line_t_min < self.dynamic_spawn_line_t_max <= 1.0
        assert self.dynamic_lateral_spread >= 0.0

        assert self.lidar_num_rays >= 4
        assert self.lidar_max_range > 0.0
        assert self.lidar_start_offset >= 0.0
        assert self.lidar_start_offset < self.lidar_max_range

        assert self.robot_radius > 0.0
        assert self.success_radius > 0.0

        assert self.min_flight_z >= 0.0
        assert self.max_flight_z > self.min_flight_z
        assert self.max_roll_pitch > 0.0

        assert self.action_dim == 4
        assert self.single_actor_obs_dim == 75
        assert self.frame_stack == 4
        assert self.actor_obs_dim == 300
        assert self.critic_obs_dim == 300

        assert self.obs_clip > 0.0
        assert self.priv_clip > 0.0


QuadrotorTask3Config = Task3Config
CrazyflieTask3Config = Task3Config
