from __future__ import annotations

from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from quadrotor_rl.tasks.task2.task2_config import Task2Config
from quadrotor_rl.tasks.task2.task2_scene import (
    get_quadrotor_task2_asset_source,
    make_quadrotor_task2_scene_cfg,
)


class QuadrotorTask2Env(gym.Env):
    """Quadrotor / Crazyflie Task2: procedural 3D trajectory tracking.

    Action:
        [num_envs, 4], normalized rotor thrust correction in [-1, 1].

    Observation:
        [num_envs, 100] = 4 stacked frames × 25 features.

    Per-frame layout:
        roll / pitch / yaw             3
        filtered action                4
        current relative target        3
        5 lookahead relative targets   15
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Task2Config):
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

        SceneCfg = make_quadrotor_task2_scene_cfg(cfg)
        self.asset_source = get_quadrotor_task2_asset_source()

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

        self.num_points = int(cfg.trajectory_num_points)

        self.waypoints = torch.zeros(
            (self.num_envs, self.num_points, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.tangents = torch.zeros_like(self.waypoints)

        self.target_idx = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
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

        self.episode_steps = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self.episode_return = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )

        self.raw_actions = torch.zeros(
            (self.num_envs, self.num_actions),
            dtype=torch.float32,
            device=self.device,
        )
        self.filtered_actions = torch.zeros_like(self.raw_actions)
        self.prev_filtered_actions = torch.zeros_like(self.raw_actions)
        self.prev_raw_actions = torch.zeros_like(self.raw_actions)
        self.motor_multipliers = torch.ones_like(self.raw_actions)

        self.last_force_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.last_torque_w = torch.zeros_like(self.last_force_w)
        self.last_torque_b = torch.zeros_like(self.last_force_w)

        self.obs_buffer = torch.zeros(
            (
                self.num_envs,
                int(cfg.frame_stack),
                int(cfg.obs_dim_per_frame),
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

        self._generate_static_trajectory(env_ids)

        start_pos = self.waypoints[env_ids, 0, :]
        self.target_idx[env_ids] = 0

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
        self.obs_buffer[env_ids] = 0.0

        self._apply_external_wrench(
            env_ids=env_ids,
            force_w=torch.zeros((env_ids.numel(), 3), dtype=torch.float32, device=self.device),
            torque_w=torch.zeros((env_ids.numel(), 3), dtype=torch.float32, device=self.device),
        )

        self.scene.write_data_to_sim()
        self.scene.update(0.0)

        single_obs = self._compute_single_obs()
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

        actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        actions = torch.clamp(actions, -1.0, 1.0)

        self.raw_actions = actions.clone()
        self.prev_filtered_actions = self.filtered_actions.clone()

        target_action = actions * float(self.cfg.action_scale)
        alpha = float(self.cfg.action_ema_alpha)
        self.filtered_actions = alpha * target_action + (1.0 - alpha) * self.filtered_actions

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

        self.episode_steps += 1

        reward, terminated, truncated, info = self._compute_reward_done_info()
        done = terminated | truncated

        self.episode_return += reward

        single_obs = self._compute_single_obs()
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

        self.prev_raw_actions = actions.clone()

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Trajectory
    # ------------------------------------------------------------------
    def _generate_static_trajectory(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        n = int(env_ids.numel())

        if n == 0:
            return

        t = torch.arange(self.num_points, dtype=torch.float32, device=self.device).view(1, -1) * float(self.cfg.policy_dt)

        ax = self._rand_range((n, 1), self.cfg.amp_x_range)
        ay = self._rand_range((n, 1), self.cfg.amp_y_range)
        az = self._rand_range((n, 1), self.cfg.amp_z_range)

        fx = self._rand_range((n, 1), self.cfg.freq_x_range)
        fy = self._rand_range((n, 1), self.cfg.freq_y_range)
        fz = self._rand_range((n, 1), self.cfg.freq_z_range)

        dx = torch.rand((n, 1), dtype=torch.float32, device=self.device) * (2.0 * torch.pi)
        dy = torch.rand((n, 1), dtype=torch.float32, device=self.device) * (2.0 * torch.pi)
        dz = torch.rand((n, 1), dtype=torch.float32, device=self.device) * (2.0 * torch.pi)

        px = ax * torch.sin(2.0 * torch.pi * fx * t + dx)
        py = ay * torch.sin(2.0 * torch.pi * fy * t + dy)
        pz = float(self.cfg.base_altitude) + az * torch.sin(2.0 * torch.pi * fz * t + dz)
        pz = torch.clamp(pz, float(self.cfg.min_trajectory_z), float(self.cfg.max_trajectory_z))

        points = torch.stack([px, py, pz], dim=-1)

        diff = torch.zeros_like(points)
        diff[:, :-1, :] = points[:, 1:, :] - points[:, :-1, :]
        diff[:, -1, :] = diff[:, -2, :]

        norm = torch.norm(diff, dim=-1, keepdim=True)
        tangents = torch.where(
            norm > 1.0e-6,
            diff / torch.clamp(norm, min=1.0e-6),
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device).view(1, 1, 3),
        )

        self.waypoints[env_ids] = points
        self.tangents[env_ids] = tangents

    def _rand_range(self, shape, range_tuple) -> torch.Tensor:
        lo, hi = range_tuple
        return float(lo) + (float(hi) - float(lo)) * torch.rand(shape, dtype=torch.float32, device=self.device)

    def _update_target_idx(self, pos_local: torch.Tensor) -> None:
        search = int(self.cfg.target_search_range)
        offsets = torch.arange(search, dtype=torch.long, device=self.device).view(1, -1)

        idx = self.target_idx.view(-1, 1) + offsets
        idx = torch.clamp(idx, 0, self.num_points - 1)

        env_arange = torch.arange(self.num_envs, dtype=torch.long, device=self.device).view(-1, 1)
        candidates = self.waypoints[env_arange, idx, :]
        dist = torch.norm(candidates - pos_local.view(self.num_envs, 1, 3), dim=-1)

        closest = torch.argmin(dist, dim=-1)
        self.target_idx = torch.clamp(self.target_idx + closest, 0, self.num_points - 1)

    def _gather_current_target(self) -> torch.Tensor:
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        return self.waypoints[env_ids, self.target_idx, :]

    def _gather_current_tangent(self) -> torch.Tensor:
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        return self.tangents[env_ids, self.target_idx, :]

    def _gather_lookahead_targets(self) -> torch.Tensor:
        k = torch.arange(
            1,
            int(self.cfg.lookahead_steps) + 1,
            dtype=torch.long,
            device=self.device,
        ).view(1, -1)

        idx = self.target_idx.view(-1, 1) + k * int(self.cfg.lookahead_interval)
        idx = torch.clamp(idx, 0, self.num_points - 1)

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device).view(-1, 1)
        return self.waypoints[env_ids, idx, :]

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def compute_obs(self) -> torch.Tensor:
        obs = self.obs_buffer.reshape(self.num_envs, -1)

        if obs.shape[-1] != self.num_observations:
            raise RuntimeError(f"obs dim mismatch: got {obs.shape[-1]}, expected {self.num_observations}")

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

    def _compute_single_obs(self) -> torch.Tensor:
        pos_w = self._root_pos_w()
        quat = self._root_quat_wxyz()

        pos_local = pos_w - self.env_origins
        self._update_target_idx(pos_local)

        roll, pitch, yaw = self._quat_to_euler_wxyz(quat)

        target = self._gather_current_target()
        rel_target = target - pos_local

        lookahead = self._gather_lookahead_targets()
        rel_lookahead = lookahead - pos_local.view(self.num_envs, 1, 3)
        rel_lookahead_flat = rel_lookahead.reshape(self.num_envs, -1)

        obs = torch.cat(
            [
                torch.stack([roll, pitch, yaw], dim=-1) / float(self.cfg.rpy_scale),
                self.filtered_actions,
                torch.clamp(rel_target / float(self.cfg.rel_pos_scale), -5.0, 5.0),
                torch.clamp(rel_lookahead_flat / float(self.cfg.lookahead_rel_pos_scale), -5.0, 5.0),
            ],
            dim=-1,
        )

        if obs.shape[-1] != int(self.cfg.obs_dim_per_frame):
            raise RuntimeError(
                f"single obs dim mismatch: got {obs.shape[-1]}, expected {self.cfg.obs_dim_per_frame}"
            )

        return torch.nan_to_num(
            torch.clamp(obs, -float(self.cfg.obs_clip), float(self.cfg.obs_clip)),
            nan=0.0,
            posinf=float(self.cfg.obs_clip),
            neginf=-float(self.cfg.obs_clip),
        )

    # ------------------------------------------------------------------
    # Reward / events
    # ------------------------------------------------------------------
    def _compute_reward_done_info(self):
        pos_w = self._root_pos_w()
        quat = self._root_quat_wxyz()
        lin_vel_w = self._root_lin_vel_w()
        ang_vel_w = self._root_ang_vel_w()

        pos_local = pos_w - self.env_origins
        self._update_target_idx(pos_local)

        target = self._gather_current_target()
        tangent = self._gather_current_tangent()

        pos_err_vec = pos_local - target
        dist_err = torch.norm(pos_err_vec, dim=-1)

        weighted_dist = torch.sqrt(
            torch.square(pos_err_vec[:, 0])
            + torch.square(pos_err_vec[:, 1])
            + torch.square(float(self.cfg.z_error_weight) * pos_err_vec[:, 2])
        )

        roll, pitch, yaw = self._quat_to_euler_wxyz(quat)
        roll_pitch_abs = torch.maximum(torch.abs(roll), torch.abs(pitch))

        heading_vec = self._quat_rotate(
            quat,
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device).view(1, 3).repeat(self.num_envs, 1),
        )

        raw_v_align = torch.sum(lin_vel_w * tangent, dim=-1)
        safe_v_align = torch.clamp(raw_v_align, -5.0, 5.0)
        heading_align = torch.sum(heading_vec * tangent, dim=-1)

        r_surv = torch.full((self.num_envs,), float(self.cfg.r_survival), dtype=torch.float32, device=self.device)
        r_track = 0.5 * torch.exp(
            -torch.square(weighted_dist) / (2.0 * float(self.cfg.r_track_sigma) ** 2)
        ) - 0.1
        r_vel = float(self.cfg.r_vel_coef) * safe_v_align
        r_heading = float(self.cfg.r_heading_coef) * heading_align
        r_smooth = float(self.cfg.r_smooth_coef) * torch.sum(torch.square(self.raw_actions - self.prev_raw_actions), dim=-1)

        raw_cont_reward = r_surv + r_track + r_vel + r_heading + r_smooth
        cont_reward = torch.clamp(
            raw_cont_reward,
            float(self.cfg.continuous_reward_clip_min),
            float(self.cfg.continuous_reward_clip_max),
        )

        crash = (
            (roll_pitch_abs > float(self.cfg.max_roll_pitch))
            | (pos_local[:, 2] < float(self.cfg.min_z))
            | (pos_local[:, 2] > float(self.cfg.max_z))
        )

        deviation = (dist_err > float(self.cfg.max_dev_err)) & (~crash)

        success = self.target_idx >= int(self.num_points - self.cfg.success_end_margin)

        timeout = self.episode_steps >= int(self.cfg.max_episode_length)

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

        completion = self.target_idx.float() / max(float(self.num_points - 1), 1.0)

        info = {
            "reward_components": {
                "R_Survival": r_surv.mean().item(),
                "R_Track": r_track.mean().item(),
                "R_Vel": r_vel.mean().item(),
                "R_Heading": r_heading.mean().item(),
                "R_Smooth": r_smooth.mean().item(),
                "R_Continuous_Clipped": cont_reward.mean().item(),
                "R_Terminal": terminal_reward.mean().item(),
                "Total": reward.mean().item(),
            },
            "events": {
                "Success_Rate": success.float().mean().item(),
                "Crash_Rate": crash.float().mean().item(),
                "Deviation_Rate": deviation.float().mean().item(),
                "Timeout_Rate": (timeout & (~terminated) & (~success)).float().mean().item(),
                "Done_Rate": done.float().mean().item(),
                "Episode_Success_Rate": (self.total_success_episodes / denom).item(),
                "Episode_Crash_Rate": (self.total_crash_episodes / denom).item(),
                "Episode_Deviation_Rate": (self.total_deviation_episodes / denom).item(),
                "Episode_Timeout_Rate": (self.total_timeout_episodes / denom).item(),
                "Episode_Done_Count": self.total_done_episodes.item(),
            },
            "telemetry": {
                "Dist_Error": dist_err.mean().item(),
                "Weighted_Dist_Error": weighted_dist.mean().item(),
                "Completion_Rate": completion.mean().item(),
                "Target_Idx": self.target_idx.float().mean().item(),
                "Z": pos_local[:, 2].mean().item(),
                "Roll": roll.mean().item(),
                "Pitch": pitch.mean().item(),
                "Yaw": yaw.mean().item(),
                "RollPitchAbs": roll_pitch_abs.mean().item(),
                "Vel_Align": safe_v_align.mean().item(),
                "Heading_Align": heading_align.mean().item(),
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
                "Obs_Dim_Per_Frame": float(self.cfg.obs_dim_per_frame),
                "Lookahead_Steps": float(self.cfg.lookahead_steps),
                "Trajectory_Num_Points": float(self.num_points),
                "Estimated_Mass": self.estimated_mass.mean().item(),
                "Hover_Thrust": self.hover_thrust.mean().item(),
                "Hover_Thrust_Per_Rotor": self.hover_thrust_per_rotor.mean().item(),
                "Asset_Source_Is_Fallback": float("fallback_usd" in str(self.asset_source)),
                "Reward_Min": reward.min().item(),
                "Reward_Max": reward.max().item(),
            },
            "task2_stats": {
                "dist_err": dist_err.mean().item(),
                "completion_rate": (completion.mean().item() * 100.0),
                "r_final_total": reward.mean().item(),
                "r_terminal": terminal_reward.mean().item(),
                "pos_z": pos_local[:, 2].mean().item(),
            },
            "is_success": success.detach().clone(),
        }

        return reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    def _actions_to_wrench(self, filtered_actions: torch.Tensor):
        motor_mult = 1.0 + filtered_actions
        motor_mult = torch.clamp(
            motor_mult,
            float(self.cfg.min_motor_multiplier),
            float(self.cfg.max_motor_multiplier),
        )

        rotor_forces = self.hover_thrust_per_rotor.view(-1, 1) * motor_mult
        total_thrust = rotor_forces.sum(dim=-1)

        max_total = self.hover_thrust * float(self.cfg.max_total_thrust_factor)
        total_thrust = torch.maximum(total_thrust, torch.zeros_like(total_thrust))
        total_thrust = torch.minimum(total_thrust, max_total)

        x = self.rotor_xy[:, 0].view(1, 4)
        y = self.rotor_xy[:, 1].view(1, 4)

        torque_x = torch.sum(y * rotor_forces, dim=-1)
        torque_y = -torch.sum(x * rotor_forces, dim=-1)
        torque_z = float(self.cfg.yaw_torque_per_newton) * torch.sum(
            self.rotor_yaw_signs.view(1, 4) * rotor_forces,
            dim=-1,
        )

        torque_b = torch.stack([torque_x, torque_y, torque_z], dim=-1)
        torque_b[:, 0:2] = torch.clamp(
            torque_b[:, 0:2],
            -float(self.cfg.max_body_moment_xy),
            float(self.cfg.max_body_moment_xy),
        )
        torque_b[:, 2] = torch.clamp(
            torque_b[:, 2],
            -float(self.cfg.max_body_moment_z),
            float(self.cfg.max_body_moment_z),
        )

        force_b = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        force_b[:, 2] = total_thrust

        quat = self._root_quat_wxyz()
        force_w = self._quat_rotate(quat, force_b)
        torque_w = self._quat_rotate(quat, torque_b)

        return force_w, torque_w, torque_b, motor_mult

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

    def check_events_for_test(self):
        return self._compute_reward_done_info()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _estimate_mass(self) -> torch.Tensor:
        try:
            masses = self.drone.root_physx_view.get_masses()
            masses = torch.as_tensor(masses, dtype=torch.float32, device=self.device)

            if masses.ndim == 2:
                total = masses.sum(dim=-1)
            else:
                total = masses.reshape(self.num_envs, -1).sum(dim=-1)

            if torch.isfinite(total).all().item() and total.mean().item() > 1.0e-5:
                return total

        except Exception:
            pass

        return torch.full(
            (self.num_envs,),
            float(self.cfg.nominal_mass),
            dtype=torch.float32,
            device=self.device,
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
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

        root_state = self.drone.data.default_root_state[env_ids].clone()
        root_state[:, :3] = self.env_origins[env_ids] + pos_local
        root_state[:, 3:7] = self._euler_to_quat_wxyz(roll, pitch, yaw)

        if zero_vel:
            root_state[:, 7:13] = 0.0

        self.drone.write_root_state_to_sim(root_state, env_ids=env_ids)

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
        cr = torch.cos(0.5 * roll)
        sr = torch.sin(0.5 * roll)
        cp = torch.cos(0.5 * pitch)
        sp = torch.sin(0.5 * pitch)
        cy = torch.cos(0.5 * yaw)
        sy = torch.sin(0.5 * yaw)

        q = torch.zeros((roll.shape[0], 4), dtype=torch.float32, device=roll.device)
        q[:, 0] = cr * cp * cy + sr * sp * sy
        q[:, 1] = sr * cp * cy - cr * sp * sy
        q[:, 2] = cr * sp * cy + sr * cp * sy
        q[:, 3] = cr * cp * sy - sr * sp * cy
        return q

    @staticmethod
    def _quat_to_euler_wxyz(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        pitch = torch.where(
            torch.abs(sinp) >= 1.0,
            torch.sign(sinp) * (torch.pi / 2.0),
            torch.asin(sinp),
        )

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    @staticmethod
    def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_w = q[:, 0:1]
        q_vec = q[:, 1:4]
        t = 2.0 * torch.cross(q_vec, v, dim=-1)
        return v + q_w * t + torch.cross(q_vec, t, dim=-1)

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------
    def _print_debug_info(self) -> None:
        print("\n" + "=" * 120)
        print("✅ [Task2] Quadrotor / Crazyflie 3D Trajectory Tracking Env Initialized")
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
        print(f"  obs_dim_per_frame       : {self.cfg.obs_dim_per_frame}")
        print(f"  frame_stack             : {self.cfg.frame_stack}")
        print(f"  actor_obs_dim           : {self.num_observations}")
        print(f"  critic_obs_dim          : {self.num_privileged_obs}")
        print(f"  lookahead_steps         : {self.cfg.lookahead_steps}")
        print(f"  lookahead_interval      : {self.cfg.lookahead_interval}")
        print(f"  trajectory_num_points   : {self.num_points}")
        print(f"  sim_dt                  : {self.cfg.sim_dt}")
        print(f"  policy_dt               : {self.cfg.policy_dt}")
        print(f"  max_episode_length      : {self.cfg.max_episode_length}")
        print("=" * 120 + "\n")


Task2Env = QuadrotorTask2Env
CrazyflieTask2Env = QuadrotorTask2Env
