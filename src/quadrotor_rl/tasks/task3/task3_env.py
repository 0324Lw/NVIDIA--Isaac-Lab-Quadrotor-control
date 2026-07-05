from __future__ import annotations

from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from quadrotor_rl.core.env.asset_state import estimate_articulation_mass
from quadrotor_rl.core.env.base_quadrotor_env import QuadrotorBaseEnvMixin
from quadrotor_rl.core.env.reset_manager import write_root_pose_to_sim
from quadrotor_rl.core.math.quat_math import (
    euler_to_quat_wxyz,
    quat_rotate,
    quat_rotate_inverse,
    quat_to_euler_wxyz,
    wrap_to_pi,
)
from quadrotor_rl.core.physics.action_semantics import (
    QuadrotorActionSemantics,
    reset_quadrotor_action_buffers,
    update_quadrotor_action_buffers,
)
from quadrotor_rl.core.physics.rotor_model import QuadrotorRotorLimits, compute_quadrotor_wrench

from quadrotor_rl.tasks.task3.task3_config import Task3Config
from quadrotor_rl.tasks.task3.task3_scene import (
    get_quadrotor_task3_asset_source,
    make_quadrotor_task3_scene_cfg,
)
from quadrotor_rl.tasks.task3.task3_world import QuadrotorTask3World


class QuadrotorTask3Env(QuadrotorBaseEnvMixin, gym.Env):
    """Quadrotor / Crazyflie Task3: dynamic-obstacle navigation.

    Action:
        [num_envs, 4], normalized rotor thrust correction in [-1, 1].

    Observation:
        [num_envs, 300] = 4 stacked frames × 75 features.

    Single-frame layout:
        relative goal in body frame      3
        distance to goal                 1
        goal heading sin/cos             2
        linear velocity body             3
        angular velocity body            3
        projected gravity body           3
        filtered action                  4
        lidar                            24
        lidar delta                      24
        risk features                    8
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Task3Config):
        super().__init__()

        cfg.validate()
        self.cfg = cfg
        self.num_envs = int(cfg.num_envs)
        self.device = str(cfg.device)
        self.dt = float(cfg.policy_dt)

        torch.manual_seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))

        sim_cfg = sim_utils.SimulationCfg(
            dt=float(cfg.sim_dt),
            device=self.device,
            physx=sim_utils.PhysxCfg(
                enable_external_forces_every_iteration=True,
                min_position_iteration_count=4,
                max_position_iteration_count=8,
                min_velocity_iteration_count=1,
                max_velocity_iteration_count=2,
            ),
        )
        self.sim = sim_utils.SimulationContext(sim_cfg)

        SceneCfg = make_quadrotor_task3_scene_cfg(cfg)
        self.asset_source = get_quadrotor_task3_asset_source()

        scene_cfg = SceneCfg(
            num_envs=int(cfg.num_envs),
            env_spacing=float(cfg.env_spacing),
        )
        self.scene = InteractiveScene(scene_cfg)

        self.sim.reset()
        self.scene.update(0.0)

        try:
            self.drone = self.scene["drone"]
        except Exception:
            self.drone = self.scene.articulations["drone"]

        self.env_origins = self.scene.env_origins.to(self.device)

        self.world = QuadrotorTask3World(
            cfg=cfg,
            num_envs=self.num_envs,
            device=self.device,
        )

        self.action_dim = int(cfg.action_dim)
        self.num_actions = int(cfg.action_dim)
        self.num_observations = int(cfg.actor_obs_dim)
        self.num_privileged_obs = int(cfg.critic_obs_dim)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_observations,),
            dtype=np.float32,
        )
        self.state_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_privileged_obs,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32,
        )

        self.rotor_xy = torch.tensor(
            cfg.rotor_xy,
            dtype=torch.float32,
            device=self.device,
        )
        self.rotor_yaw_signs = torch.tensor(
            cfg.rotor_yaw_signs,
            dtype=torch.float32,
            device=self.device,
        )

        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.raw_actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)
        self.filtered_actions = torch.zeros_like(self.raw_actions)
        self.prev_filtered_actions = torch.zeros_like(self.raw_actions)
        self.prev_raw_actions = torch.zeros_like(self.raw_actions)
        self.motor_multipliers = torch.ones_like(self.raw_actions)

        self.action_semantics = QuadrotorActionSemantics(
            action_scale=float(cfg.action_scale),
            action_ema_alpha=float(cfg.action_ema_alpha),
            action_deadzone=float(getattr(cfg, "action_deadzone", 0.0)),
        )
        self.rotor_limits = QuadrotorRotorLimits(
            min_motor_multiplier=float(cfg.min_motor_multiplier),
            max_motor_multiplier=float(cfg.max_motor_multiplier),
            max_total_thrust_factor=float(cfg.max_total_thrust_factor),
            max_body_moment_xy=float(cfg.max_body_moment_xy),
            max_body_moment_z=float(cfg.max_body_moment_z),
        )

        self.last_force_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.last_torque_w = torch.zeros_like(self.last_force_w)
        self.last_torque_b = torch.zeros_like(self.last_force_w)

        self.current_lidar = torch.ones((self.num_envs, int(cfg.lidar_num_rays)), dtype=torch.float32, device=self.device)
        self.prev_lidar = torch.ones_like(self.current_lidar)

        self.prev_goal_distance = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.obs_buffer = torch.zeros(
            (
                self.num_envs,
                int(cfg.frame_stack),
                int(cfg.single_actor_obs_dim),
            ),
            dtype=torch.float32,
            device=self.device,
        )

        self.total_done_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_success_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_crash_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_timeout_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_deviation_episodes = torch.zeros((), dtype=torch.float32, device=self.device)

        self.estimated_mass = self._estimate_mass()
        self.hover_thrust = self.estimated_mass * float(cfg.gravity)
        self.hover_thrust_per_rotor = self.hover_thrust / 4.0

        self.last_info: Dict = {}

        self.reset()

        if bool(cfg.print_debug_info):
            self._print_debug_info()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def reset(
        self,
        env_ids: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ):
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
            full_reset = True
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
            full_reset = int(env_ids.numel()) == self.num_envs

        if env_ids.numel() == 0:
            obs = self.compute_obs()
            return obs, {"state": self.compute_privileged_obs()}

        self.world.reset(env_ids)

        start_pos = self.world.start_pos[env_ids].clone()

        self._write_root_pose_to_sim(
            env_ids=env_ids,
            pos_local=start_pos,
            roll=torch.zeros(env_ids.numel(), dtype=torch.float32, device=self.device),
            pitch=torch.zeros(env_ids.numel(), dtype=torch.float32, device=self.device),
            yaw=torch.zeros(env_ids.numel(), dtype=torch.float32, device=self.device),
            zero_vel=True,
        )

        try:
            joint_pos = self.drone.data.default_joint_pos[env_ids].clone()
            joint_vel = torch.zeros_like(self.drone.data.default_joint_vel[env_ids])
            self.drone.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        except Exception:
            pass

        try:
            self.drone.reset(env_ids)
        except Exception:
            pass

        self.episode_steps[env_ids] = 0
        self.episode_return[env_ids] = 0.0

        self.raw_actions[env_ids] = 0.0
        self.filtered_actions[env_ids] = 0.0
        self.prev_filtered_actions[env_ids] = 0.0
        self.prev_raw_actions[env_ids] = 0.0
        self.motor_multipliers[env_ids] = 1.0

        self.last_force_w[env_ids] = 0.0
        self.last_torque_w[env_ids] = 0.0
        self.last_torque_b[env_ids] = 0.0

        self._apply_external_wrench(
            env_ids=env_ids,
            force_w=torch.zeros((env_ids.numel(), 3), dtype=torch.float32, device=self.device),
            torque_w=torch.zeros((env_ids.numel(), 3), dtype=torch.float32, device=self.device),
        )

        self.scene.write_data_to_sim()
        self.scene.update(0.0)

        pos_local = self._root_pos_w() - self.env_origins
        yaw = self._quat_to_euler_wxyz(self._root_quat_wxyz())[2]
        lidar = self.world.get_lidar_scan(pos_local, yaw)

        self.current_lidar[env_ids] = lidar[env_ids]
        self.prev_lidar[env_ids] = lidar[env_ids]
        self.prev_goal_distance[env_ids] = self.world.distance_to_goal(pos_local)[env_ids]

        self.obs_buffer[env_ids] = 0.0

        single_obs = self._compute_single_obs(update_lidar=False)
        for k in range(int(self.cfg.frame_stack)):
            self.obs_buffer[env_ids, k, :] = single_obs[env_ids]

        obs = self.compute_obs()
        state = self.compute_privileged_obs()

        self.last_info = {"state": state}

        if full_reset:
            return obs, {"state": state}
        return obs[env_ids], {"state": state[env_ids]}

    @torch.no_grad()
    def step(self, actions: torch.Tensor):
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)

        if actions.shape != (self.num_envs, self.num_actions):
            raise RuntimeError(
                f"expected action shape {(self.num_envs, self.num_actions)}, got {tuple(actions.shape)}"
            )

        self.raw_actions, self.filtered_actions, self.prev_filtered_actions = update_quadrotor_action_buffers(
            actions=actions,
            previous_filtered_actions=self.filtered_actions,
            semantics=self.action_semantics,
        )

        force_w, torque_w, torque_b, motor_mult = self._actions_to_wrench(self.filtered_actions)

        self.motor_multipliers = motor_mult.clone()
        self.last_force_w = force_w.clone()
        self.last_torque_w = torque_w.clone()
        self.last_torque_b = torque_b.clone()

        all_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        for _ in range(int(self.cfg.decimation)):
            self._apply_external_wrench(all_env_ids, force_w, torque_w)

            try:
                self.drone.write_data_to_sim()
            except Exception:
                pass

            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(float(self.cfg.sim_dt))

        self.world.step_dynamics(dt=float(self.cfg.policy_dt))

        self.episode_steps += 1

        single_obs = self._compute_single_obs(update_lidar=True)
        reward, terminated, truncated, info = self._compute_reward_done_info()
        done = terminated | truncated

        self.episode_return += reward

        self.obs_buffer = torch.roll(self.obs_buffer, shifts=-1, dims=1)
        self.obs_buffer[:, -1, :] = single_obs

        obs = self.compute_obs()
        state = self.compute_privileged_obs()
        info["state"] = state
        self.last_info = info

        if done.any():
            reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
            info["terminal_observation"] = obs[reset_ids].clone()
            info["terminal_state"] = state[reset_ids].clone()
            self.reset(reset_ids)
            obs = self.compute_obs()
            info["state"] = self.compute_privileged_obs()
            self.last_info = info

        self.prev_raw_actions = self.raw_actions.clone()

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        pass

    def set_curriculum(self, num_static: int, num_dynamic: int, max_sg_dist: float) -> None:
        """Compatibility curriculum hook from the legacy Task3 design.

        It changes the number of active analytic obstacles and the start-goal
        distance range for future resets. Tensor storage keeps its original
        maximum size, while valid masks select the active subset.
        """

        max_static = int(self.world.static_pos.shape[1])
        max_dynamic = int(self.world.dynamic_pos.shape[1])

        num_static = int(max(0, min(int(num_static), max_static)))
        num_dynamic = int(max(0, min(int(num_dynamic), max_dynamic)))
        max_sg_dist = float(max(1.0, min(float(max_sg_dist), float(self.cfg.arena_size) * 0.90)))

        self.cfg.num_static_obs = num_static
        self.cfg.num_dynamic_obs = num_dynamic
        self.cfg.max_start_goal_dist = max_sg_dist
        self.cfg.min_start_goal_dist = max(1.0, 0.70 * max_sg_dist)

        # Keep world cfg synchronized because it holds the same design knobs.
        self.world.cfg.num_static_obs = self.cfg.num_static_obs
        self.world.cfg.num_dynamic_obs = self.cfg.num_dynamic_obs
        self.world.cfg.max_start_goal_dist = self.cfg.max_start_goal_dist
        self.world.cfg.min_start_goal_dist = self.cfg.min_start_goal_dist

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def compute_obs(self) -> torch.Tensor:
        obs = self.obs_buffer.reshape(self.num_envs, -1)

        if obs.shape[-1] != self.num_observations:
            raise RuntimeError(
                f"obs dim mismatch: got {obs.shape[-1]}, expected {self.num_observations}"
            )

        return torch.nan_to_num(
            torch.clamp(obs, -float(self.cfg.obs_clip), float(self.cfg.obs_clip)),
            nan=0.0,
            posinf=float(self.cfg.obs_clip),
            neginf=-float(self.cfg.obs_clip),
        )

    def compute_privileged_obs(self) -> torch.Tensor:
        return self.compute_obs()

    def get_privileged_observations(self) -> torch.Tensor:
        return self.compute_privileged_obs()

    def _compute_states(self) -> torch.Tensor:
        return self.compute_privileged_obs()

    def _compute_single_obs(self, update_lidar: bool = True) -> torch.Tensor:
        pos_w = self._root_pos_w()
        quat = self._root_quat_wxyz()
        lin_vel_w = self._root_lin_vel_w()
        ang_vel_w = self._root_ang_vel_w()

        pos_local = pos_w - self.env_origins
        roll, pitch, yaw = self._quat_to_euler_wxyz(quat)

        goal_vec_w = self.world.goal_vector(pos_local)
        goal_vec_b = self._quat_rotate_inverse(quat, goal_vec_w)

        goal_dist = torch.norm(goal_vec_w, dim=-1)
        heading_angle = torch.atan2(goal_vec_b[:, 1], goal_vec_b[:, 0])

        lin_vel_b = self._quat_rotate_inverse(quat, lin_vel_w)
        ang_vel_b = self._quat_rotate_inverse(quat, ang_vel_w)

        gravity_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        gravity_w[:, 2] = -1.0
        projected_gravity_b = self._quat_rotate_inverse(quat, gravity_w)

        if update_lidar:
            new_lidar = self.world.get_lidar_scan(pos_local, yaw)
            lidar_delta = new_lidar - self.current_lidar
            self.prev_lidar = self.current_lidar.clone()
            self.current_lidar = new_lidar.clone()
        else:
            new_lidar = self.current_lidar
            lidar_delta = self.current_lidar - self.prev_lidar

        risk = self._risk_features_from_current(pos_local, yaw)

        obs = torch.cat(
            [
                torch.clamp(goal_vec_b / float(self.cfg.max_start_goal_dist), -5.0, 5.0),
                torch.clamp(goal_dist.unsqueeze(-1) / float(self.cfg.max_start_goal_dist), 0.0, 5.0),
                torch.sin(heading_angle).unsqueeze(-1),
                torch.cos(heading_angle).unsqueeze(-1),
                torch.clamp(lin_vel_b / 5.0, -5.0, 5.0),
                torch.clamp(ang_vel_b / 8.0, -5.0, 5.0),
                projected_gravity_b,
                self.filtered_actions,
                torch.clamp(new_lidar, 0.0, 1.0),
                torch.clamp(lidar_delta, -1.0, 1.0),
                torch.clamp(risk, -5.0, 5.0),
            ],
            dim=-1,
        )

        if obs.shape[-1] != int(self.cfg.single_actor_obs_dim):
            raise RuntimeError(
                f"single obs dim mismatch: got {obs.shape[-1]}, expected {self.cfg.single_actor_obs_dim}"
            )

        return torch.nan_to_num(
            torch.clamp(obs, -float(self.cfg.obs_clip), float(self.cfg.obs_clip)),
            nan=0.0,
            posinf=float(self.cfg.obs_clip),
            neginf=-float(self.cfg.obs_clip),
        )

    def _risk_features_from_current(self, pos_local: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        goal_vec = self.world.goal_vector(pos_local)
        goal_dist = torch.norm(goal_vec, dim=-1)
        nearest = self.world.nearest_obstacle_distance(pos_local)

        lidar = self.current_lidar
        n = int(self.cfg.lidar_num_rays)

        front = lidar[:, 0]
        left = lidar[:, n // 4]
        back = lidar[:, n // 2]
        right = lidar[:, (3 * n) // 4]

        lidar_min = lidar.min(dim=-1).values
        lidar_mean = lidar.mean(dim=-1)

        features = torch.stack(
            [
                torch.clamp(lidar_min, 0.0, 1.0),
                torch.clamp(lidar_mean, 0.0, 1.0),
                torch.clamp(front, 0.0, 1.0),
                torch.clamp(left, 0.0, 1.0),
                torch.clamp(back, 0.0, 1.0),
                torch.clamp(right, 0.0, 1.0),
                torch.clamp(nearest / float(self.cfg.lidar_max_range), 0.0, 1.0),
                torch.clamp(goal_dist / float(self.cfg.max_start_goal_dist), 0.0, 5.0),
            ],
            dim=-1,
        )

        return torch.nan_to_num(features, nan=1.0, posinf=1.0, neginf=0.0)

    # ------------------------------------------------------------------
    # Reward / events
    # ------------------------------------------------------------------
    def _compute_reward_done_info(self):
        pos_w = self._root_pos_w()
        quat = self._root_quat_wxyz()
        lin_vel_w = self._root_lin_vel_w()
        ang_vel_w = self._root_ang_vel_w()

        pos_local = pos_w - self.env_origins
        roll, pitch, yaw = self._quat_to_euler_wxyz(quat)

        goal_vec = self.world.goal_vector(pos_local)
        goal_dist = torch.norm(goal_vec, dim=-1)
        goal_dist_xy = torch.norm(goal_vec[:, :2], dim=-1)

        unit_goal = goal_vec / torch.clamp(goal_dist.unsqueeze(-1), min=1.0e-6)
        v_toward = torch.sum(lin_vel_w * unit_goal, dim=-1)
        v_toward_safe = torch.clamp(v_toward, -5.0, 5.0)

        heading_vec = self._quat_rotate(
            quat,
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device).view(1, 3).repeat(self.num_envs, 1),
        )
        heading_xy = heading_vec[:, :2]
        goal_xy = unit_goal[:, :2]
        heading_xy_norm = torch.norm(heading_xy, dim=-1)
        goal_xy_norm = torch.norm(goal_xy, dim=-1)

        cos_theta_xy = torch.sum(heading_xy * goal_xy, dim=-1) / torch.clamp(
            heading_xy_norm * goal_xy_norm,
            min=1.0e-6,
        )

        v_xy_norm = torch.norm(lin_vel_w[:, :2], dim=-1)

        dz = pos_local[:, 2] - float(self.cfg.start_goal_z)
        roll_pitch_abs = torch.maximum(torch.abs(roll), torch.abs(pitch))

        min_lidar_norm = self.current_lidar.min(dim=-1).values
        min_lidar_dist = min_lidar_norm * float(self.cfg.lidar_max_range)

        r_step = torch.full((self.num_envs,), float(self.cfg.r_step), dtype=torch.float32, device=self.device)
        r_height = float(self.cfg.r_height_scale) * torch.exp(-float(self.cfg.r_height_k) * torch.square(dz))
        r_att = float(self.cfg.r_att_penalty) * (torch.abs(roll) + torch.abs(pitch))
        r_smooth = float(self.cfg.r_smooth_penalty) * torch.sum(torch.square(self.raw_actions - self.prev_raw_actions), dim=-1)
        r_approach = float(self.cfg.r_approach) * v_toward_safe
        r_dir = float(self.cfg.r_dir) * v_xy_norm * cos_theta_xy

        repulsion_active = min_lidar_dist < 1.0
        r_repulsion_raw = float(self.cfg.r_repulsion_max) * torch.exp(
            -1.5 * (min_lidar_dist - float(self.cfg.safe_lidar_dist))
        )
        r_repulsion = torch.where(repulsion_active, r_repulsion_raw, torch.zeros_like(r_repulsion_raw))

        raw_cont_reward = r_step + r_height + r_att + r_smooth + r_approach + r_dir + r_repulsion
        cont_reward = torch.clamp(
            raw_cont_reward,
            float(self.cfg.continuous_reward_clip_min),
            float(self.cfg.continuous_reward_clip_max),
        )

        lidar_crash = min_lidar_dist < float(self.cfg.safe_lidar_dist)
        obstacle_collision = self.world.check_obstacle_collision(pos_local)
        out_of_bounds = self.world.check_out_of_bounds(pos_local)
        flip_crash = roll_pitch_abs > float(self.cfg.max_roll_pitch)
        z_deviation = torch.abs(dz) > float(self.cfg.max_z_err)

        floor_crash = pos_local[:, 2] < float(self.cfg.min_flight_z)
        height_deviation = z_deviation & (~floor_crash)

        success_xy = goal_dist_xy < float(self.cfg.success_xy_tolerance)
        success_z = torch.abs(pos_local[:, 2] - self.world.goal_pos[:, 2]) < float(self.cfg.success_z_tolerance)
        success_world = self.world.check_success(pos_local)
        success = (success_xy & success_z) | success_world

        timeout = self.episode_steps >= int(self.cfg.max_episode_length)

        crash = lidar_crash | obstacle_collision | floor_crash | flip_crash
        deviation = (height_deviation | out_of_bounds) & (~crash)

        terminated = crash | deviation
        truncated = success | (timeout & (~terminated))

        time_bonus = torch.clamp(
            (float(self.cfg.max_episode_length) - self.episode_steps.float()) * float(self.cfg.time_bonus_coef),
            min=0.0,
        )

        terminal_reward = torch.zeros_like(cont_reward)
        terminal_reward = torch.where(crash, torch.full_like(terminal_reward, float(self.cfg.r_crash)), terminal_reward)
        terminal_reward = torch.where(deviation, torch.full_like(terminal_reward, float(self.cfg.r_deviate)), terminal_reward)
        terminal_reward = torch.where(success, torch.full_like(terminal_reward, float(self.cfg.r_success_base)) + time_bonus, terminal_reward)

        reward = cont_reward + terminal_reward
        reward = torch.clamp(reward, float(self.cfg.final_reward_clip_min), float(self.cfg.final_reward_clip_max))
        reward = torch.nan_to_num(reward, nan=0.0, posinf=float(self.cfg.final_reward_clip_max), neginf=float(self.cfg.final_reward_clip_min))

        done = terminated | truncated
        done_count = done.float().sum()

        self.total_done_episodes += done_count.detach()
        self.total_success_episodes += success.float().sum().detach()
        self.total_crash_episodes += crash.float().sum().detach()
        self.total_deviation_episodes += deviation.float().sum().detach()
        self.total_timeout_episodes += (timeout & (~terminated) & (~success)).float().sum().detach()

        denom = torch.clamp(self.total_done_episodes, min=1.0)

        progress = self.prev_goal_distance - goal_dist
        self.prev_goal_distance = goal_dist.detach().clone()

        reason = "ALIVE"
        if crash.any().item():
            reason = "CRASH"
        elif deviation.any().item():
            reason = "DEVIATION"
        elif success.any().item():
            reason = "SUCCESS"
        elif timeout.any().item():
            reason = "TIMEOUT"

        info = {
            "reward_components": {
                "R_Step": r_step.mean().item(),
                "R_Height": r_height.mean().item(),
                "R_Att": r_att.mean().item(),
                "R_Smooth": r_smooth.mean().item(),
                "R_Approach": r_approach.mean().item(),
                "R_Dir": r_dir.mean().item(),
                "R_Repulsion": r_repulsion.mean().item(),
                "R_Continuous_Clipped": cont_reward.mean().item(),
                "R_Terminal": terminal_reward.mean().item(),
                "Total": reward.mean().item(),
            },
            "events": {
                "Success_Rate": success.float().mean().item(),
                "Crash_Rate": crash.float().mean().item(),
                "Lidar_Crash_Rate": lidar_crash.float().mean().item(),
                "Obstacle_Collision_Rate": obstacle_collision.float().mean().item(),
                "Floor_Crash_Rate": floor_crash.float().mean().item(),
                "Flip_Crash_Rate": flip_crash.float().mean().item(),
                "Deviation_Rate": deviation.float().mean().item(),
                "Out_Of_Bounds_Rate": out_of_bounds.float().mean().item(),
                "Timeout_Rate": (timeout & (~terminated) & (~success)).float().mean().item(),
                "Done_Rate": done.float().mean().item(),
                "Episode_Success_Rate": (self.total_success_episodes / denom).item(),
                "Episode_Crash_Rate": (self.total_crash_episodes / denom).item(),
                "Episode_Deviation_Rate": (self.total_deviation_episodes / denom).item(),
                "Episode_Timeout_Rate": (self.total_timeout_episodes / denom).item(),
                "Episode_Done_Count": self.total_done_episodes.item(),
            },
            "telemetry": {
                "Goal_Dist": goal_dist.mean().item(),
                "Goal_Dist_XY": goal_dist_xy.mean().item(),
                "Progress": progress.mean().item(),
                "Pos_Z": pos_local[:, 2].mean().item(),
                "Z_Error": dz.mean().item(),
                "Min_Lidar": min_lidar_dist.mean().item(),
                "Lidar_Norm_Min": min_lidar_norm.mean().item(),
                "Nearest_Obstacle": self.world.nearest_obstacle_distance(pos_local).mean().item(),
                "Roll": roll.mean().item(),
                "Pitch": pitch.mean().item(),
                "Yaw": yaw.mean().item(),
                "RollPitchAbs": roll_pitch_abs.mean().item(),
                "Vel_Toward": v_toward_safe.mean().item(),
                "Heading_Align": cos_theta_xy.mean().item(),
                "Lin_Vel": torch.norm(lin_vel_w, dim=-1).mean().item(),
                "Ang_Vel": torch.norm(ang_vel_w, dim=-1).mean().item(),
                "Action_Mean": self.filtered_actions.mean().item(),
                "Action_Abs": self.filtered_actions.abs().mean().item(),
                "Motor_Multiplier": self.motor_multipliers.mean().item(),
                "Force_Z_W": self.last_force_w[:, 2].mean().item(),
                "Torque_X_W": self.last_torque_w[:, 0].mean().item(),
                "Torque_Y_W": self.last_torque_w[:, 1].mean().item(),
                "Torque_Z_W": self.last_torque_w[:, 2].mean().item(),
                "Episode_Length": self.episode_steps.float().mean().item(),
                "Episode_Return": self.episode_return.mean().item(),
            },
            "debug": {
                "Actor_Obs_Dim": float(self.num_observations),
                "Critic_Obs_Dim": float(self.num_privileged_obs),
                "Action_Dim": float(self.num_actions),
                "Single_Obs_Dim": float(self.cfg.single_actor_obs_dim),
                "Frame_Stack": float(self.cfg.frame_stack),
                "Lidar_Rays": float(self.cfg.lidar_num_rays),
                "Static_Obs": float(self.cfg.num_static_obs),
                "Dynamic_Obs": float(self.cfg.num_dynamic_obs),
                "Estimated_Mass": self.estimated_mass.mean().item(),
                "Hover_Thrust": self.hover_thrust.mean().item(),
                "Hover_Thrust_Per_Rotor": self.hover_thrust_per_rotor.mean().item(),
                "Asset_Source_Is_Fallback": float("fallback_usd" in str(self.asset_source)),
                "Reward_Min": reward.min().item(),
                "Reward_Max": reward.max().item(),
            },
            "task3_stats": {
                "r_step": r_step.mean().item(),
                "r_height": r_height.mean().item(),
                "r_att": r_att.mean().item(),
                "r_smooth": r_smooth.mean().item(),
                "r_approach": r_approach.mean().item(),
                "r_dir": r_dir.mean().item(),
                "r_repulsion": r_repulsion.mean().item(),
                "r_cont_clipped": cont_reward.mean().item(),
                "r_terminal": terminal_reward.mean().item(),
                "r_final_total": reward.mean().item(),
                "dist_xy": goal_dist_xy.mean().item(),
                "pos_z": pos_local[:, 2].mean().item(),
                "min_lidar": min_lidar_dist.mean().item(),
                "reason": reason,
            },
            "is_success": success.detach().clone(),
        }

        return reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    def _actions_to_wrench(self, filtered_actions: torch.Tensor):
        return compute_quadrotor_wrench(
            filtered_actions=filtered_actions,
            hover_thrust_per_rotor=self.hover_thrust_per_rotor,
            hover_thrust=self.hover_thrust,
            rotor_xy=self.rotor_xy,
            rotor_yaw_signs=self.rotor_yaw_signs,
            yaw_torque_per_newton=float(self.cfg.yaw_torque_per_newton),
            limits=self.rotor_limits,
            quat_wxyz=self._root_quat_wxyz(),
            quat_rotate_fn=self._quat_rotate,
        )

    def _apply_external_wrench(self, env_ids: torch.Tensor, force_w: torch.Tensor, torque_w: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

        if env_ids.numel() == 0:
            return

        forces = torch.zeros((env_ids.numel(), 1, 3), dtype=torch.float32, device=self.device)
        torques = torch.zeros_like(forces)
        forces[:, 0, :] = force_w
        torques[:, 0, :] = torque_w

        try:
            self.drone.set_external_force_and_torque(
                forces=forces,
                torques=torques,
                body_ids=[0],
                env_ids=env_ids,
                is_global=True,
            )
            return
        except TypeError:
            pass
        except Exception:
            pass

        try:
            self.drone.set_external_force_and_torque(
                forces=forces,
                torques=torques,
                body_ids=[0],
                is_global=True,
            )
            return
        except TypeError:
            pass
        except Exception:
            pass

        try:
            self.drone.set_external_force_and_torque(
                forces=forces,
                torques=torques,
                body_ids=[0],
            )
            return
        except Exception:
            pass

        num_bodies = int(getattr(self.drone, "num_bodies", 1))
        forces_full = torch.zeros((env_ids.numel(), num_bodies, 3), dtype=torch.float32, device=self.device)
        torques_full = torch.zeros_like(forces_full)
        forces_full[:, 0, :] = force_w
        torques_full[:, 0, :] = torque_w
        self.drone.set_external_force_and_torque(forces_full, torques_full)

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def set_root_pose_for_test(
        self,
        pos_local: torch.Tensor,
        roll: float | torch.Tensor = 0.0,
        pitch: float | torch.Tensor = 0.0,
        yaw: float | torch.Tensor = 0.0,
        zero_vel: bool = True,
    ) -> None:
        pos_local = torch.as_tensor(pos_local, dtype=torch.float32, device=self.device)

        if pos_local.shape == (3,):
            pos_local = pos_local.view(1, 3).repeat(self.num_envs, 1)

        assert pos_local.shape == (self.num_envs, 3)

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        roll_t = self._as_env_vector(roll)
        pitch_t = self._as_env_vector(pitch)
        yaw_t = self._as_env_vector(yaw)

        self._write_root_pose_to_sim(env_ids, pos_local, roll_t, pitch_t, yaw_t, zero_vel=zero_vel)
        self.scene.write_data_to_sim()
        self.scene.update(0.0)

        yaw_now = self._quat_to_euler_wxyz(self._root_quat_wxyz())[2]
        lidar = self.world.get_lidar_scan(pos_local, yaw_now)
        self.prev_lidar = self.current_lidar.clone()
        self.current_lidar = lidar.clone()

    def check_events_for_test(self):
        return self._compute_reward_done_info()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _estimate_mass(self) -> torch.Tensor:
        return estimate_articulation_mass(
            drone=self.drone,
            num_envs=self.num_envs,
            device=self.device,
            fallback_mass=float(self.cfg.nominal_mass),
        )

    def _write_root_pose_to_sim(
        self,
        env_ids: torch.Tensor,
        pos_local: torch.Tensor,
        roll: torch.Tensor,
        pitch: torch.Tensor,
        yaw: torch.Tensor,
        zero_vel: bool = True,
    ) -> None:
        write_root_pose_to_sim(
            drone=self.drone,
            env_origins=self.env_origins,
            env_ids=env_ids,
            pos_local=pos_local,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            zero_vel=zero_vel,
        )

    def _root_pos_w(self) -> torch.Tensor:
        return self.drone.data.root_pos_w.clone()

    def _root_quat_wxyz(self) -> torch.Tensor:
        return self.drone.data.root_quat_w.clone()

    def _root_lin_vel_w(self) -> torch.Tensor:
        if hasattr(self.drone.data, "root_lin_vel_w"):
            return self.drone.data.root_lin_vel_w.clone()
        return self._quat_rotate(self._root_quat_wxyz(), self.drone.data.root_lin_vel_b.clone())

    def _root_ang_vel_w(self) -> torch.Tensor:
        if hasattr(self.drone.data, "root_ang_vel_w"):
            return self.drone.data.root_ang_vel_w.clone()
        return self._quat_rotate(self._root_quat_wxyz(), self.drone.data.root_ang_vel_b.clone())

    def _as_env_vector(self, x) -> torch.Tensor:
        if torch.is_tensor(x):
            x = x.to(device=self.device, dtype=torch.float32)
            if x.numel() == 1:
                return x.reshape(1).repeat(self.num_envs)
            return x.reshape(self.num_envs)

        return torch.full((self.num_envs,), float(x), dtype=torch.float32, device=self.device)

    @staticmethod
    def _euler_to_quat_wxyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        return euler_to_quat_wxyz(roll, pitch, yaw)

    @staticmethod
    def _quat_to_euler_wxyz(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return quat_to_euler_wxyz(q)

    @staticmethod
    def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return quat_rotate(q, v)

    @staticmethod
    def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return quat_rotate_inverse(q, v)

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------
    def _print_debug_info(self) -> None:
        print("\n" + "=" * 120)
        print("✅ [Task3] Quadrotor / Crazyflie Dynamic Obstacle Navigation Env Initialized")
        print(f"  asset_source            : {self.asset_source}")
        print(f"  num_envs                : {self.num_envs}")
        print(f"  device                  : {self.device}")
        print(f"  num_bodies              : {getattr(self.drone, 'num_bodies', '<unknown>')}")
        print(f"  num_joints              : {getattr(self.drone, 'num_joints', '<unknown>')}")
        print(f"  body_names              : {list(getattr(self.drone, 'body_names', []))}")
        print(f"  joint_names             : {list(getattr(self.drone, 'joint_names', []))}")
        print(f"  estimated_mass          : {self.estimated_mass.mean().item():.6f} kg")
        print(f"  hover_thrust            : {self.hover_thrust.mean().item():.6f} N")
        print(f"  action_dim              : {self.num_actions}")
        print(f"  single_actor_obs_dim    : {self.cfg.single_actor_obs_dim}")
        print(f"  frame_stack             : {self.cfg.frame_stack}")
        print(f"  actor_obs_dim           : {self.num_observations}")
        print(f"  critic_obs_dim          : {self.num_privileged_obs}")
        print(f"  lidar_num_rays          : {self.cfg.lidar_num_rays}")
        print(f"  lidar_max_range         : {self.cfg.lidar_max_range}")
        print(f"  static_obs              : {self.cfg.num_static_obs}")
        print(f"  dynamic_obs             : {self.cfg.num_dynamic_obs}")
        print(f"  arena_size              : {self.cfg.arena_size}")
        print(f"  sim_dt                  : {self.cfg.sim_dt}")
        print(f"  policy_dt               : {self.cfg.policy_dt}")
        print(f"  max_episode_length      : {self.cfg.max_episode_length}")
        print("=" * 120 + "\n")


Task3Env = QuadrotorTask3Env
CrazyflieTask3Env = QuadrotorTask3Env
