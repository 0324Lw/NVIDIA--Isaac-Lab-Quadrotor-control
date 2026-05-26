from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Task4Config:
    """Quadrotor / Crazyflie Task4: vision-based narrow-gate racing.

    This config migrates the old PyBullet visual racing world to an Isaac Lab
    compatible analytic world layer.

    World logic:
        - 30m x 10m x 5m enclosed racing arena.
        - Fixed spawn at [-12, 0, 1.5].
        - Five procedural gates along the X-axis.
        - Gate inner opening size 0.5m with 0.05m frame thickness.
        - Random gate roll / pitch perturbation for aggressive racing.
        - Analytic 64x64 depth vision in [0, 1].
        - All tensors are batched for vectorized RL environments.
    """

    # ------------------------------------------------------------------
    # Basic
    # ------------------------------------------------------------------
    num_envs: int = 512
    device: str = "cuda:0"
    seed: int = 42

    sim_dt: float = 0.005
    decimation: int = 4
    max_episode_length_s: float = 12.0

    # ------------------------------------------------------------------
    # Asset / Isaac Lab scene spacing
    # ------------------------------------------------------------------
    spawn_height: float = 1.5
    env_spacing: float = 36.0

    crazyflie_usd_url: str = (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
        "Assets/Isaac/5.1/Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd"
    )

    # ------------------------------------------------------------------
    # Arena
    # ------------------------------------------------------------------
    arena_length: float = 30.0
    arena_width: float = 10.0
    arena_height: float = 5.0
    wall_thickness: float = 0.5

    start_pos: Tuple[float, float, float] = (-12.0, 0.0, 1.5)

    # ------------------------------------------------------------------
    # Gate track
    # ------------------------------------------------------------------
    num_gates: int = 5
    gate_size: float = 0.5
    gate_thickness: float = 0.05

    max_roll_pitch_deg: float = 45.0
    max_pitch_offset_deg: float = 22.5

    first_gate_x_offset: float = 4.0
    last_gate_margin: float = 4.0

    gate_y_range: Tuple[float, float] = (-2.0, 2.0)
    gate_z_range: Tuple[float, float] = (1.5, 3.5)

    # A soft centerline is used for curriculum / reward shaping later.
    centerline_samples: int = 100

    # ------------------------------------------------------------------
    # Analytic depth camera
    # ------------------------------------------------------------------
    cam_res_w: int = 64
    cam_res_h: int = 64
    cam_fov_deg: float = 110.0
    cam_near: float = 0.1
    cam_far: float = 10.0

    # Camera is fixed on body +X, with body +Z as up direction.
    camera_forward_axis: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    camera_up_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Collision / racing events
    # ------------------------------------------------------------------
    robot_radius: float = 0.08
    pass_gate_margin: float = 0.02
    gate_plane_tolerance: float = 0.15

    min_flight_z: float = 0.10
    max_flight_z: float = 4.80
    max_roll_pitch: float = 1.05

    success_after_last_gate: bool = True

    # ------------------------------------------------------------------
    # Action / actuator dynamics / sim2real
    # ------------------------------------------------------------------
    action_scale: float = 0.50
    action_deadzone: float = 0.05
    action_ema_alpha: float = 0.60

    idle_motor_multiplier: float = 0.20
    min_motor_multiplier: float = 0.20
    max_motor_multiplier: float = 2.00

    max_total_thrust_factor: float = 2.4
    max_body_moment_xy: float = 0.030
    max_body_moment_z: float = 0.008

    dr_mass_range: float = 0.10
    noise_imu_std: float = 0.02
    noise_depth_prob: float = 0.01
    noise_depth_std: float = 0.03
    enable_sensor_noise: bool = False

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    r_step: float = -0.25
    r_track_v_scale: float = 0.20
    r_track_k: float = 2.0
    r_align_pose: float = 0.15
    r_smooth: float = -0.01
    r_depth_safety: float = -0.10

    r_gate_base: float = 15.0
    r_crash: float = -100.0
    r_timeout_penalty: float = -100.0
    r_success: float = 200.0

    max_centerline_dist: float = 3.0
    crash_z_min: float = 0.20
    crash_roll_pitch_max: float = 1.20
    pass_gate_dist: float = 0.80

    continuous_reward_clip_min: float = -1.0
    continuous_reward_clip_max: float = 1.0
    final_reward_clip_min: float = -150.0
    final_reward_clip_max: float = 350.0

    # ------------------------------------------------------------------
    # Quadrotor physical constants for later env
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
    # Planned observation for later Task4 env
    # ------------------------------------------------------------------
    action_dim: int = 4

    # Vision + compact proprio for later env:
    #   depth image        1 x 64 x 64 = 4096
    #   compact state      32
    # total                4128
    depth_channels: int = 1
    compact_state_dim: int = 32
    frame_stack: int = 1

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
    def arena_half_length(self) -> float:
        return float(self.arena_length / 2.0)

    @property
    def arena_half_width(self) -> float:
        return float(self.arena_width / 2.0)

    @property
    def gate_inner_half(self) -> float:
        return float(self.gate_size / 2.0)

    @property
    def gate_outer_half(self) -> float:
        return float(self.gate_size / 2.0 + self.gate_thickness)

    @property
    def gate_start_x(self) -> float:
        return float(self.start_pos[0] + self.first_gate_x_offset)

    @property
    def gate_end_x(self) -> float:
        return float(self.arena_half_length - self.last_gate_margin)

    @property
    def depth_dim(self) -> int:
        return int(self.depth_channels * self.cam_res_h * self.cam_res_w)

    @property
    def single_actor_obs_dim(self) -> int:
        return int(self.depth_dim + self.compact_state_dim)

    @property
    def actor_obs_dim(self) -> int:
        return int(self.single_actor_obs_dim * self.frame_stack)

    @property
    def critic_obs_dim(self) -> int:
        # Later env may append gate privileged features. For now keep aligned.
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
        assert self.env_spacing > self.arena_length

        assert self.arena_length > 0.0
        assert self.arena_width > 0.0
        assert self.arena_height > 0.0
        assert self.wall_thickness > 0.0

        sx, sy, sz = self.start_pos
        assert -self.arena_half_length < sx < self.arena_half_length
        assert -self.arena_half_width < sy < self.arena_half_width
        assert 0.0 < sz < self.arena_height

        assert self.num_gates >= 1
        assert self.gate_size > 0.0
        assert self.gate_thickness > 0.0
        assert self.gate_outer_half > self.gate_inner_half

        assert self.gate_start_x < self.gate_end_x
        assert -self.arena_half_length < self.gate_start_x < self.arena_half_length
        assert -self.arena_half_length < self.gate_end_x < self.arena_half_length

        gy0, gy1 = self.gate_y_range
        gz0, gz1 = self.gate_z_range
        assert gy0 < gy1
        assert gz0 < gz1
        assert -self.arena_half_width < gy0 < self.arena_half_width
        assert -self.arena_half_width < gy1 < self.arena_half_width
        assert 0.0 < gz0 < self.arena_height
        assert 0.0 < gz1 < self.arena_height

        assert self.max_roll_pitch_deg >= 0.0
        assert self.max_pitch_offset_deg >= 0.0
        assert self.centerline_samples >= self.num_gates + 1

        assert self.cam_res_w == 64
        assert self.cam_res_h == 64
        assert self.depth_channels == 1
        assert self.cam_fov_deg > 0.0 and self.cam_fov_deg < 180.0
        assert 0.0 < self.cam_near < self.cam_far

        assert self.robot_radius > 0.0
        assert self.pass_gate_margin >= 0.0
        assert self.gate_plane_tolerance > 0.0
        assert self.min_flight_z >= 0.0
        assert self.max_flight_z > self.min_flight_z
        assert self.max_flight_z <= self.arena_height + self.wall_thickness
        assert self.max_roll_pitch > 0.0

        assert self.gravity > 0.0
        assert self.nominal_mass > 0.0
        assert self.hover_thrust > 0.0
        assert self.hover_thrust_per_rotor > 0.0
        assert self.arm_length > 0.0
        assert self.yaw_torque_per_newton >= 0.0

        assert self.action_dim == 4
        assert self.compact_state_dim == 32
        assert self.depth_dim == 4096
        assert self.single_actor_obs_dim == 4128
        assert self.actor_obs_dim == 4128
        assert self.critic_obs_dim == 4128

        assert self.obs_clip > 0.0
        assert self.priv_clip > 0.0


QuadrotorTask4Config = Task4Config
CrazyflieTask4Config = Task4Config
