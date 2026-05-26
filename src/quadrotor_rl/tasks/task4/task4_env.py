from __future__ import annotations

from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from quadrotor_rl.tasks.task4.task4_config import Task4Config
from quadrotor_rl.tasks.task4.task4_scene import (
    get_quadrotor_task4_asset_source,
    make_quadrotor_task4_scene_cfg,
)
from quadrotor_rl.tasks.task4.task4_world import QuadrotorTask4World


class QuadrotorTask4Env(gym.Env):
    """Quadrotor / Crazyflie Task4: vision-based narrow-gate racing.

    Action:
        [num_envs, 4], normalized rotor correction in [-1, 1].

    Observation:
        [num_envs, 4128] = 64x64 analytic depth + 32 compact proprio/gate features.

    Compact state layout:
        target gate analytic features       14
        linear velocity body                 3
        angular velocity body                3
        projected gravity body               3
        filtered action                      4
        roll / pitch / yaw normalized        3
        target gate progress                 1
        speed norm                           1
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Task4Config):
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

        SceneCfg = make_quadrotor_task4_scene_cfg(cfg)
        self.asset_source = get_quadrotor_task4_asset_source()

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

        self.world = QuadrotorTask4World(
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

        self.last_force_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.last_torque_w = torch.zeros_like(self.last_force_w)
        self.last_torque_b = torch.zeros_like(self.last_force_w)

        self.current_depth = torch.ones(
            (self.num_envs, 1, int(cfg.cam_res_h), int(cfg.cam_res_w)),
            dtype=torch.float32,
            device=self.device,
        )
        self.prev_depth = self.current_depth.clone()

        self.prev_pos_local = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.prev_centerline_dist = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.obs_buffer = torch.zeros(
            (self.num_envs, int(cfg.actor_obs_dim)),
            dtype=torch.float32,
            device=self.device,
        )

        self.total_done_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_success_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_crash_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_timeout_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_missed_gate_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
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
        quat = self._root_quat_wxyz()

        depth = self.world.get_depth_vision(pos_local, quat)
        self.current_depth[env_ids] = depth[env_ids]
        self.prev_depth[env_ids] = depth[env_ids]

        self.prev_pos_local[env_ids] = pos_local[env_ids]

        centerline_dist, _, _ = self._centerline_terms(pos_local, self._root_lin_vel_w())
        self.prev_centerline_dist[env_ids] = centerline_dist[env_ids]

        single_obs = self._compute_single_obs(update_depth=False)
        self.obs_buffer[env_ids] = single_obs[env_ids]

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

        prev_pos_local = self._root_pos_w() - self.env_origins

        self.raw_actions = actions.clone()
        self.prev_filtered_actions = self.filtered_actions.clone()

        active_actions = torch.where(
            torch.abs(actions) < float(self.cfg.action_deadzone),
            torch.zeros_like(actions),
            actions,
        )

        target_action = active_actions * float(self.cfg.action_scale)
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

        single_obs = self._compute_single_obs(update_depth=True)
        reward, terminated, truncated, info = self._compute_reward_done_info(prev_pos_local=prev_pos_local)
        done = terminated | truncated

        self.episode_return += reward

        self.obs_buffer = single_obs
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
        self.prev_pos_local = self._root_pos_w() - self.env_origins

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def compute_obs(self) -> torch.Tensor:
        obs = self.obs_buffer

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

    def _compute_single_obs(self, update_depth: bool = True) -> torch.Tensor:
        pos_w = self._root_pos_w()
        quat = self._root_quat_wxyz()
        lin_vel_w = self._root_lin_vel_w()
        ang_vel_w = self._root_ang_vel_w()

        pos_local = pos_w - self.env_origins
        roll, pitch, yaw = self._quat_to_euler_wxyz(quat)

        if update_depth:
            depth = self.world.get_depth_vision(pos_local, quat)
            if bool(getattr(self.cfg, "enable_sensor_noise", False)):
                depth = self._apply_depth_noise(depth)
            self.prev_depth = self.current_depth.clone()
            self.current_depth = depth.clone()
        else:
            depth = self.current_depth

        target_features = self.world.current_target_gate_features(pos_local, quat)

        lin_vel_b = self._quat_rotate_inverse(quat, lin_vel_w)
        ang_vel_b = self._quat_rotate_inverse(quat, ang_vel_w)

        gravity_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        gravity_w[:, 2] = -1.0
        projected_gravity_b = self._quat_rotate_inverse(quat, gravity_w)

        rpy_norm = torch.stack([roll, pitch, yaw], dim=-1) / torch.pi
        gate_progress = self.world.target_gate_idx.float().view(-1, 1) / max(float(self.cfg.num_gates), 1.0)
        speed = torch.norm(lin_vel_w, dim=-1, keepdim=True) / 10.0

        compact = torch.cat(
            [
                torch.clamp(target_features, -5.0, 5.0),
                torch.clamp(lin_vel_b / 5.0, -5.0, 5.0),
                torch.clamp(ang_vel_b / 8.0, -5.0, 5.0),
                projected_gravity_b,
                torch.clamp(self.filtered_actions, -2.0, 2.0),
                torch.clamp(rpy_norm, -2.0, 2.0),
                torch.clamp(gate_progress, 0.0, 2.0),
                torch.clamp(speed, 0.0, 5.0),
            ],
            dim=-1,
        )

        if compact.shape[-1] != int(self.cfg.compact_state_dim):
            raise RuntimeError(
                f"compact state dim mismatch: got {compact.shape[-1]}, expected {self.cfg.compact_state_dim}"
            )

        obs = torch.cat(
            [
                torch.clamp(depth.view(self.num_envs, -1), 0.0, 1.0),
                compact,
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

    def _apply_depth_noise(self, depth: torch.Tensor) -> torch.Tensor:
        out = depth.clone()

        if float(self.cfg.noise_depth_std) > 0.0:
            out = out + torch.randn_like(out) * float(self.cfg.noise_depth_std)

        if float(self.cfg.noise_depth_prob) > 0.0:
            mask = torch.rand_like(out) < float(self.cfg.noise_depth_prob)
            salt = torch.rand_like(out) > 0.5
            out = torch.where(mask, salt.float(), out)

        return torch.clamp(out, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Reward / events
    # ------------------------------------------------------------------
    def _centerline_terms(self, pos_local: torch.Tensor, lin_vel_w: torch.Tensor):
        centerline = self.world.centerline
        diff = centerline - pos_local[:, None, :]
        dist = torch.norm(diff, dim=-1)
        idx = torch.argmin(dist, dim=-1)

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        nearest = centerline[env_ids, idx]
        next_idx = torch.clamp(idx + 1, max=int(self.cfg.centerline_samples) - 1)
        next_pt = centerline[env_ids, next_idx]

        tangent = next_pt - nearest
        tangent = tangent / torch.clamp(torch.norm(tangent, dim=-1, keepdim=True), min=1.0e-6)

        centerline_dist = dist[env_ids, idx]
        v_tangent = torch.sum(lin_vel_w * tangent, dim=-1)

        return centerline_dist, tangent, v_tangent

    def _missed_gate_mask(self, pos_local: torch.Tensor, passed_this_step: torch.Tensor, target_idx_before: torch.Tensor) -> torch.Tensor:
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        idx = torch.clamp(target_idx_before, 0, int(self.cfg.num_gates) - 1)

        gate_pos = self.world.gate_pos[env_ids, idx]
        gate_rot = self.world.gate_rot[env_ids, idx]

        local = torch.bmm(
            gate_rot.transpose(1, 2),
            (pos_local - gate_pos).unsqueeze(-1),
        ).squeeze(-1)

        plane_crossed = local[:, 2] > float(self.cfg.gate_plane_tolerance)
        outside_opening = (
            (torch.abs(local[:, 0]) > float(self.cfg.pass_gate_dist))
            | (torch.abs(local[:, 1]) > float(self.cfg.pass_gate_dist))
        )

        active = target_idx_before < int(self.cfg.num_gates)
        return active & plane_crossed & outside_opening & (~passed_this_step)

    def _compute_reward_done_info(self, prev_pos_local: torch.Tensor):
        pos_w = self._root_pos_w()
        quat = self._root_quat_wxyz()
        lin_vel_w = self._root_lin_vel_w()
        ang_vel_w = self._root_ang_vel_w()

        pos_local = pos_w - self.env_origins
        roll, pitch, yaw = self._quat_to_euler_wxyz(quat)

        centerline_dist, tangent, v_tangent = self._centerline_terms(pos_local, lin_vel_w)
        progress = self.prev_centerline_dist - centerline_dist
        self.prev_centerline_dist = centerline_dist.detach().clone()

        target_idx_before = self.world.target_gate_idx.clone()
        passed_this_step = self.world.update_gate_progress(prev_pos_local, pos_local)
        target_idx_after = self.world.target_gate_idx.clone()

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        idx = torch.clamp(target_idx_before, 0, int(self.cfg.num_gates) - 1)

        target_tangent = self.world.gate_tangent[env_ids, idx]
        x_body_world = self._quat_rotate(
            quat,
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device).view(1, 3).repeat(self.num_envs, 1),
        )
        pose_align = torch.sum(x_body_world * target_tangent, dim=-1)

        v_tangent_clipped = torch.clamp(v_tangent, -5.0, 5.0)
        r_track_forward = (
            float(self.cfg.r_track_v_scale)
            * v_tangent_clipped
            * torch.exp(-float(self.cfg.r_track_k) * torch.square(centerline_dist))
        )
        r_track_backward = float(self.cfg.r_track_v_scale) * v_tangent_clipped * 1.5
        r_track = torch.where(v_tangent_clipped < 0.0, r_track_backward, r_track_forward)

        r_step = torch.full((self.num_envs,), float(self.cfg.r_step), dtype=torch.float32, device=self.device)
        r_smooth = float(self.cfg.r_smooth) * torch.sum(torch.square(self.raw_actions - self.prev_raw_actions), dim=-1)
        r_align = float(self.cfg.r_align_pose) * pose_align

        depth_min = self.current_depth.view(self.num_envs, -1).min(dim=-1).values
        r_depth = float(self.cfg.r_depth_safety) * torch.exp(-4.0 * depth_min)

        raw_cont_reward = r_step + r_smooth + r_track + r_align + r_depth
        cont_reward = torch.clamp(
            raw_cont_reward,
            float(self.cfg.continuous_reward_clip_min),
            float(self.cfg.continuous_reward_clip_max),
        )

        roll_pitch_abs = torch.maximum(torch.abs(roll), torch.abs(pitch))

        floor_crash = pos_local[:, 2] < float(self.cfg.crash_z_min)
        flip_crash = roll_pitch_abs > float(self.cfg.crash_roll_pitch_max)
        gate_collision = self.world.check_gate_collision(pos_local)
        out_of_bounds = self.world.check_out_of_bounds(pos_local)
        centerline_deviation = centerline_dist > float(self.cfg.max_centerline_dist)
        missed_gate = self._missed_gate_mask(pos_local, passed_this_step, target_idx_before)

        success = self.world.check_success()
        timeout = self.episode_steps >= int(self.cfg.max_episode_length)

        crash = floor_crash | flip_crash | gate_collision | missed_gate
        deviation = (out_of_bounds | centerline_deviation) & (~crash)

        terminated = crash | deviation
        truncated = success | (timeout & (~terminated))

        gate_bonus = torch.where(
            passed_this_step,
            float(self.cfg.r_gate_base) * torch.clamp(target_idx_after.float(), min=1.0),
            torch.zeros_like(cont_reward),
        )

        terminal_reward = gate_bonus.clone()
        terminal_reward = torch.where(crash, torch.full_like(terminal_reward, float(self.cfg.r_crash)), terminal_reward)
        terminal_reward = torch.where(deviation, torch.full_like(terminal_reward, float(self.cfg.r_crash)), terminal_reward)
        terminal_reward = torch.where(timeout & (~terminated) & (~success), torch.full_like(terminal_reward, float(self.cfg.r_timeout_penalty)), terminal_reward)
        terminal_reward = torch.where(success, terminal_reward + float(self.cfg.r_success), terminal_reward)

        reward = cont_reward + terminal_reward
        reward = torch.clamp(
            reward,
            float(self.cfg.final_reward_clip_min),
            float(self.cfg.final_reward_clip_max),
        )
        reward = torch.nan_to_num(
            reward,
            nan=0.0,
            posinf=float(self.cfg.final_reward_clip_max),
            neginf=float(self.cfg.final_reward_clip_min),
        )

        done = terminated | truncated
        done_count = done.float().sum()

        self.total_done_episodes += done_count.detach()
        self.total_success_episodes += success.float().sum().detach()
        self.total_crash_episodes += crash.float().sum().detach()
        self.total_missed_gate_episodes += missed_gate.float().sum().detach()
        self.total_deviation_episodes += deviation.float().sum().detach()
        self.total_timeout_episodes += (timeout & (~terminated) & (~success)).float().sum().detach()

        denom = torch.clamp(self.total_done_episodes, min=1.0)

        reason = "ALIVE"
        if crash.any().item():
            reason = "CRASH"
        elif deviation.any().item():
            reason = "DEVIATION"
        elif success.any().item():
            reason = "SUCCESS_ALL_GATES"
        elif timeout.any().item():
            reason = "TIMEOUT"

        info = {
            "reward_components": {
                "R_Step": r_step.mean().item(),
                "R_Track": r_track.mean().item(),
                "R_Align": r_align.mean().item(),
                "R_Smooth": r_smooth.mean().item(),
                "R_Depth": r_depth.mean().item(),
                "R_Continuous_Clipped": cont_reward.mean().item(),
                "R_Gate_Bonus": gate_bonus.mean().item(),
                "R_Terminal": terminal_reward.mean().item(),
                "Total": reward.mean().item(),
            },
            "events": {
                "Success_Rate": success.float().mean().item(),
                "Crash_Rate": crash.float().mean().item(),
                "Floor_Crash_Rate": floor_crash.float().mean().item(),
                "Flip_Crash_Rate": flip_crash.float().mean().item(),
                "Gate_Collision_Rate": gate_collision.float().mean().item(),
                "Missed_Gate_Rate": missed_gate.float().mean().item(),
                "Deviation_Rate": deviation.float().mean().item(),
                "Out_Of_Bounds_Rate": out_of_bounds.float().mean().item(),
                "Centerline_Deviation_Rate": centerline_deviation.float().mean().item(),
                "Timeout_Rate": (timeout & (~terminated) & (~success)).float().mean().item(),
                "Done_Rate": done.float().mean().item(),
                "Gate_Pass_Rate": passed_this_step.float().mean().item(),
                "Episode_Success_Rate": (self.total_success_episodes / denom).item(),
                "Episode_Crash_Rate": (self.total_crash_episodes / denom).item(),
                "Episode_Missed_Gate_Rate": (self.total_missed_gate_episodes / denom).item(),
                "Episode_Deviation_Rate": (self.total_deviation_episodes / denom).item(),
                "Episode_Timeout_Rate": (self.total_timeout_episodes / denom).item(),
                "Episode_Done_Count": self.total_done_episodes.item(),
            },
            "telemetry": {
                "Target_Gate_Idx": self.world.target_gate_idx.float().mean().item(),
                "Passed_Gates": self.world.target_gate_idx.float().mean().item(),
                "Centerline_Dist": centerline_dist.mean().item(),
                "Progress": progress.mean().item(),
                "V_Tangent": v_tangent_clipped.mean().item(),
                "Pose_Align": pose_align.mean().item(),
                "Depth_Min": depth_min.mean().item(),
                "Depth_Mean": self.current_depth.mean().item(),
                "Pos_X": pos_local[:, 0].mean().item(),
                "Pos_Y": pos_local[:, 1].mean().item(),
                "Pos_Z": pos_local[:, 2].mean().item(),
                "Roll": roll.mean().item(),
                "Pitch": pitch.mean().item(),
                "Yaw": yaw.mean().item(),
                "RollPitchAbs": roll_pitch_abs.mean().item(),
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
                "Depth_Dim": float(self.cfg.depth_dim),
                "Compact_State_Dim": float(self.cfg.compact_state_dim),
                "Num_Gates": float(self.cfg.num_gates),
                "Camera_Res_W": float(self.cfg.cam_res_w),
                "Camera_Res_H": float(self.cfg.cam_res_h),
                "Estimated_Mass": self.estimated_mass.mean().item(),
                "Hover_Thrust": self.hover_thrust.mean().item(),
                "Hover_Thrust_Per_Rotor": self.hover_thrust_per_rotor.mean().item(),
                "Asset_Source_Is_Fallback": float("fallback_usd" in str(self.asset_source)),
                "Reward_Min": reward.min().item(),
                "Reward_Max": reward.max().item(),
            },
            "task4_stats": {
                "r_cont_clipped": cont_reward.mean().item(),
                "r_track": r_track.mean().item(),
                "r_align": r_align.mean().item(),
                "r_smooth": r_smooth.mean().item(),
                "r_depth": r_depth.mean().item(),
                "r_terminal": terminal_reward.mean().item(),
                "total_reward": reward.mean().item(),
                "passed_gates": self.world.target_gate_idx.float().mean().item(),
                "centerline_dist": centerline_dist.mean().item(),
                "depth_min": depth_min.mean().item(),
                "reason": reason,
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

        quat = self._root_quat_wxyz()
        depth = self.world.get_depth_vision(pos_local, quat)
        self.prev_depth = self.current_depth.clone()
        self.current_depth = depth.clone()

    def check_events_for_test(self):
        prev = self.prev_pos_local.clone()
        return self._compute_reward_done_info(prev_pos_local=prev)

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

    @staticmethod
    def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_inv = q.clone()
        q_inv[:, 1:4] *= -1.0
        return QuadrotorTask4Env._quat_rotate(q_inv, v)

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------
    def _print_debug_info(self) -> None:
        print("\n" + "=" * 120)
        print("✅ [Task4] Quadrotor / Crazyflie Vision Gate Racing Env Initialized")
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
        print(f"  depth_dim               : {self.cfg.depth_dim}")
        print(f"  compact_state_dim       : {self.cfg.compact_state_dim}")
        print(f"  actor_obs_dim           : {self.num_observations}")
        print(f"  critic_obs_dim          : {self.num_privileged_obs}")
        print(f"  gates                   : {self.cfg.num_gates}")
        print(f"  camera                  : {self.cfg.cam_res_w} x {self.cfg.cam_res_h}, fov={self.cfg.cam_fov_deg}")
        print(f"  arena                   : {self.cfg.arena_length} x {self.cfg.arena_width} x {self.cfg.arena_height}")
        print(f"  sim_dt                  : {self.cfg.sim_dt}")
        print(f"  policy_dt               : {self.cfg.policy_dt}")
        print(f"  max_episode_length      : {self.cfg.max_episode_length}")
        print("=" * 120 + "\n")


Task4Env = QuadrotorTask4Env
CrazyflieTask4Env = QuadrotorTask4Env
