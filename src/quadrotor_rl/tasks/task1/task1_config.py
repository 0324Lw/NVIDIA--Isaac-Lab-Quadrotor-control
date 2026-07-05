from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from quadrotor_rl.core.config.base_quadrotor_config import BaseQuadrotorConfig


@dataclass
class Task1Config(BaseQuadrotorConfig):
    """Quadrotor / Crazyflie Task1: Hover and Attitude Stabilization.

    Migration source:
        The legacy legacy simulator task used the legacy hover simulator, target altitude
        tracking, 4-frame stacking, EMA action smoothing, altitude reward,
        attitude penalty, smoothness penalty, and terminal events for crash,
        deviation, timeout, and stable hover success.

    Isaac Lab version:
        - Asset: Bitcraze Crazyflie 2.X USD.
        - Action: 4 normalized rotor thrust corrections in [-1, 1].
        - Low-level control: convert rotor thrusts into root force and torque.
        - Observation: stacked proprioceptive state for hover control.
    """

    # ------------------------------------------------------------------
    # Basic
    # ------------------------------------------------------------------
    num_envs: int = 512
    device: str = "cuda:0"
    seed: int = 42

    sim_dt: float = 0.005
    decimation: int = 4
    max_episode_length_s: float = 10.0

    # ------------------------------------------------------------------
    # Asset
    # ------------------------------------------------------------------
    spawn_height: float = 1.0
    env_spacing: float = 2.5

    # Smoke test verified fallback USD on Isaac 5.1:
    #   num_bodies=5, num_joints=4, mass≈0.0282 kg.
    crazyflie_usd_url: str = (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
        "Assets/Isaac/5.1/Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd"
    )

    # ------------------------------------------------------------------
    # Physical constants / nominal model
    # ------------------------------------------------------------------
    gravity: float = 9.81
    nominal_mass: float = 0.0282

    # Approximate Crazyflie arm length, used for rotor force -> torque mapping.
    arm_length: float = 0.046

    # Rotor layout in body frame, meters.
    # The exact visual prop order is not critical for Task1 hover because the
    # policy learns through the resulting root wrench. The signs are kept
    # deterministic and documented.
    rotor_xy_m1: Tuple[float, float] = (0.046, 0.046)
    rotor_xy_m2: Tuple[float, float] = (-0.046, 0.046)
    rotor_xy_m3: Tuple[float, float] = (-0.046, -0.046)
    rotor_xy_m4: Tuple[float, float] = (0.046, -0.046)

    # Alternating rotor yaw direction signs.
    rotor_yaw_signs: Tuple[float, float, float, float] = (1.0, -1.0, 1.0, -1.0)

    # Converts rotor thrust difference to yaw torque. Conservative because
    # the smoke test showed yaw torque 0.02 N*m is already very strong.
    yaw_torque_per_newton: float = 0.006

    # ------------------------------------------------------------------
    # Task target
    # ------------------------------------------------------------------
    target_pos: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    target_yaw: float = 0.0

    # ------------------------------------------------------------------
    # Action model
    # ------------------------------------------------------------------
    action_dim: int = 4

    # Network action u in [-1, 1].
    # motor_multiplier = 1 + action_scale * u
    # rotor_force = hover_force_per_rotor * motor_multiplier
    action_scale: float = 0.65

    # EMA action smoothing. Legacy value was 0.5.
    action_ema_alpha: float = 0.55

    min_motor_multiplier: float = 0.0
    max_motor_multiplier: float = 2.2

    # Clamp root wrench for numeric safety.
    max_total_thrust_factor: float = 2.4
    max_body_moment_xy: float = 0.025
    max_body_moment_z: float = 0.006

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    frame_stack: int = 4

    # Single-frame observation layout:
    #   position error in body frame             3
    #   linear velocity in body frame            3
    #   angular velocity in body frame           3
    #   projected gravity in body frame          3
    #   yaw error sin/cos                        2
    #   previous filtered action                 4
    #   action delta                             4
    #   motor multiplier                         4
    #   stable progress                          1
    # total                                      27
    single_actor_obs_dim: int = 27

    obs_clip: float = 10.0
    priv_clip: float = 20.0

    pos_error_scale: float = 2.0
    lin_vel_scale: float = 3.0
    ang_vel_scale: float = 8.0
    stable_progress_scale: float = 1.0

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    alive_reward: float = 0.015

    w_pos: float = 2.50
    w_height: float = 1.20
    w_lin_vel: float = 0.25
    w_ang_vel: float = 0.06
    w_upright: float = 0.80
    w_yaw: float = 0.08
    w_action_smooth: float = 0.030
    w_action_mag: float = 0.002

    # Event rewards
    rew_success: float = 10.0
    rew_crash: float = -10.0
    rew_deviation: float = -10.0
    rew_timeout: float = -1.0

    # Reward safety clamp. Legacy final clamp was widened to preserve event
    # rewards; we keep that idea but use a moderate clamp for skrl PPO.
    reward_clip_min: float = -20.0
    reward_clip_max: float = 80.0

    # ------------------------------------------------------------------
    # Termination thresholds
    # ------------------------------------------------------------------
    min_z: float = 0.12
    max_z: float = 2.40

    max_xy_error: float = 1.50
    max_z_error: float = 0.75

    # Legacy roll/pitch threshold was 0.4 rad. Isaac first version uses a
    # slightly wider crash threshold for exploration, while success remains strict.
    max_roll_pitch: float = 0.75

    success_pos_error: float = 0.10
    success_lin_vel: float = 0.20
    success_ang_vel: float = 0.35
    success_roll_pitch: float = 0.15
    success_steps_req: int = 150

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
        # Task1 has no asymmetric privileged state yet.
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
        self.validate_common()
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

        assert self.action_dim == 4
        assert 0.0 < self.action_ema_alpha <= 1.0
        assert self.action_scale > 0.0
        assert self.max_motor_multiplier > self.min_motor_multiplier
        assert self.max_total_thrust_factor > 1.0
        assert self.max_body_moment_xy > 0.0
        assert self.max_body_moment_z > 0.0

        assert self.frame_stack >= 1
        assert self.single_actor_obs_dim == 27
        assert self.actor_obs_dim == 108
        assert self.critic_obs_dim == 108

        for name in [
            "obs_clip",
            "priv_clip",
            "pos_error_scale",
            "lin_vel_scale",
            "ang_vel_scale",
            "stable_progress_scale",
        ]:
            assert float(getattr(self, name)) > 0.0, f"{name} must be positive"

        assert self.reward_clip_min < self.reward_clip_max
        assert self.min_z >= 0.0
        assert self.max_z > self.min_z
        assert self.max_xy_error > 0.0
        assert self.max_z_error > 0.0
        assert self.max_roll_pitch > 0.0
        assert self.success_pos_error > 0.0
        assert self.success_lin_vel > 0.0
        assert self.success_ang_vel > 0.0
        assert self.success_roll_pitch > 0.0
        assert self.success_steps_req >= 1


QuadrotorTask1Config = Task1Config
CrazyflieTask1Config = Task1Config
