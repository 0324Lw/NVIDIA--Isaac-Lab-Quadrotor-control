from __future__ import annotations

from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from quadrotor_rl.tasks.task1.task1_config import Task1Config
from quadrotor_rl.tasks.task1.task1_scene import make_quadrotor_task1_scene_cfg, get_quadrotor_task1_asset_source


class QuadrotorTask1Env(gym.Env):
    """Quadrotor / Crazyflie Task1: hover and attitude stabilization.

    Action:
        [num_envs, 4], normalized rotor thrust correction in [-1, 1].

    Observation:
        [num_envs, 108] = 4 stacked frames × 27 features.

    Control:
        Convert four normalized rotor commands into a root force and body
        moment, then apply the corresponding global wrench to the Crazyflie
        root body.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Task1Config):
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

        SceneCfg = make_quadrotor_task1_scene_cfg(cfg)
        self.asset_source = get_quadrotor_task1_asset_source()

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

        self.target_pos_local = torch.tensor(
            cfg.target_pos,
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3).repeat(self.num_envs, 1)

        self.target_yaw = torch.full(
            (self.num_envs,),
            float(cfg.target_yaw),
            dtype=torch.float32,
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
        self.success_counter = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )

        self.raw_actions = torch.zeros(
            (self.num_envs, self.num_actions),
            dtype=torch.float32,
            device=self.device,
        )
        self.filtered_actions = torch.zeros_like(self.raw_actions)
        self.prev_filtered_actions = torch.zeros_like(self.raw_actions)
        self.motor_multipliers = torch.ones_like(self.raw_actions)

        self.last_force_w = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.last_torque_w = torch.zeros_like(self.last_force_w)
        self.last_torque_b = torch.zeros_like(self.last_force_w)

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

        self.estimated_mass = self._estimate_mass()
        self.hover_thrust = self.estimated_mass * float(cfg.gravity)
        self.hover_thrust_per_rotor = self.hover_thrust / 4.0

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

        self._write_root_pose_to_sim(
            env_ids=env_ids,
            pos_local=self.target_pos_local[env_ids],
            roll=torch.zeros(env_ids.numel(), dtype=torch.float32, device=self.device),
            pitch=torch.zeros(env_ids.numel(), dtype=torch.float32, device=self.device),
            yaw=self.target_yaw[env_ids],
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
        self.success_counter[env_ids] = 0

        self.raw_actions[env_ids] = 0.0
        self.filtered_actions[env_ids] = 0.0
        self.prev_filtered_actions[env_ids] = 0.0
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

        alpha = float(self.cfg.action_ema_alpha)
        self.filtered_actions = alpha * actions + (1.0 - alpha) * self.filtered_actions

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

        if done.any():
            reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
            info["terminal_observation"] = obs[reset_ids].clone()
            info["terminal_state"] = state[reset_ids].clone()
            self.reset(reset_ids)
            obs = self.compute_obs()
            info["state"] = self.compute_privileged_obs()

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        pass

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

    def _compute_single_obs(self) -> torch.Tensor:
        pos_w = self._root_pos_w()
        quat_wxyz = self._root_quat_wxyz()
        lin_vel_w = self._root_lin_vel_w()
        ang_vel_w = self._root_ang_vel_w()

        pos_local = pos_w - self.env_origins
        pos_error_w = pos_local - self.target_pos_local
        pos_error_b = self._quat_rotate_inverse(quat_wxyz, pos_error_w)

        lin_vel_b = self._quat_rotate_inverse(quat_wxyz, lin_vel_w)
        ang_vel_b = self._quat_rotate_inverse(quat_wxyz, ang_vel_w)

        gravity_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        gravity_w[:, 2] = -1.0
        projected_gravity_b = self._quat_rotate_inverse(quat_wxyz, gravity_w)

        _, _, yaw = self._quat_to_euler_wxyz(quat_wxyz)
        yaw_error = self._wrap_to_pi(yaw - self.target_yaw)

        action_delta = self.filtered_actions - self.prev_filtered_actions

        stable_progress = torch.clamp(
            self.success_counter.float() / max(float(self.cfg.success_steps_req), 1.0),
            0.0,
            1.0,
        ).unsqueeze(-1)

        obs = torch.cat(
            [
                torch.clamp(pos_error_b / float(self.cfg.pos_error_scale), -5.0, 5.0),
                torch.clamp(lin_vel_b / float(self.cfg.lin_vel_scale), -5.0, 5.0),
                torch.clamp(ang_vel_b / float(self.cfg.ang_vel_scale), -5.0, 5.0),
                projected_gravity_b,
                torch.sin(yaw_error).unsqueeze(-1),
                torch.cos(yaw_error).unsqueeze(-1),
                self.filtered_actions,
                action_delta,
                torch.clamp(self.motor_multipliers, 0.0, float(self.cfg.max_motor_multiplier)),
                stable_progress,
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

    # ------------------------------------------------------------------
    # Reward / events
    # ------------------------------------------------------------------
    def _compute_reward_done_info(self):
        pos_w = self._root_pos_w()
        quat = self._root_quat_wxyz()
        lin_vel_w = self._root_lin_vel_w()
        ang_vel_w = self._root_ang_vel_w()

        pos_local = pos_w - self.env_origins
        pos_error = pos_local - self.target_pos_local

        xy_error = torch.norm(pos_error[:, :2], dim=-1)
        z_error = pos_error[:, 2]
        pos_error_norm = torch.norm(pos_error, dim=-1)

        roll, pitch, yaw = self._quat_to_euler_wxyz(quat)
        roll_pitch_abs = torch.maximum(torch.abs(roll), torch.abs(pitch))
        yaw_error = self._wrap_to_pi(yaw - self.target_yaw)

        lin_vel_norm = torch.norm(lin_vel_w, dim=-1)
        ang_vel_norm = torch.norm(ang_vel_w, dim=-1)

        r_alive = torch.full((self.num_envs,), float(self.cfg.alive_reward), dtype=torch.float32, device=self.device)
        r_pos = torch.exp(-2.2 * torch.square(pos_error_norm))
        r_height = torch.exp(-5.0 * torch.square(z_error))
        r_upright = torch.exp(-6.0 * (torch.square(roll) + torch.square(pitch)))
        r_yaw = torch.exp(-1.5 * torch.square(yaw_error))

        p_lin_vel = -torch.square(lin_vel_norm)
        p_ang_vel = -torch.square(ang_vel_norm)
        p_action_smooth = -torch.mean(torch.square(self.filtered_actions - self.prev_filtered_actions), dim=-1)
        p_action_mag = -torch.mean(torch.square(self.filtered_actions), dim=-1)

        reward_raw = (
            r_alive
            + float(self.cfg.w_pos) * r_pos
            + float(self.cfg.w_height) * r_height
            + float(self.cfg.w_upright) * r_upright
            + float(self.cfg.w_yaw) * r_yaw
            + float(self.cfg.w_lin_vel) * p_lin_vel
            + float(self.cfg.w_ang_vel) * p_ang_vel
            + float(self.cfg.w_action_smooth) * p_action_smooth
            + float(self.cfg.w_action_mag) * p_action_mag
        )

        crash_low = pos_local[:, 2] < float(self.cfg.min_z)
        crash_high = pos_local[:, 2] > float(self.cfg.max_z)
        crash_att = roll_pitch_abs > float(self.cfg.max_roll_pitch)
        crash = crash_low | crash_high | crash_att

        deviation = (
            (xy_error > float(self.cfg.max_xy_error))
            | (torch.abs(z_error) > float(self.cfg.max_z_error))
        ) & (~crash)

        stable = (
            (pos_error_norm < float(self.cfg.success_pos_error))
            & (lin_vel_norm < float(self.cfg.success_lin_vel))
            & (ang_vel_norm < float(self.cfg.success_ang_vel))
            & (roll_pitch_abs < float(self.cfg.success_roll_pitch))
        )

        self.success_counter = torch.where(
            stable,
            self.success_counter + 1,
            torch.zeros_like(self.success_counter),
        )

        success = self.success_counter >= int(self.cfg.success_steps_req)
        timeout = self.episode_steps >= int(self.cfg.max_episode_length)

        terminated = crash | deviation | success
        truncated = timeout & (~terminated)

        event_reward = torch.zeros_like(reward_raw)
        event_reward = torch.where(crash, torch.full_like(event_reward, float(self.cfg.rew_crash)), event_reward)
        event_reward = torch.where(deviation, torch.full_like(event_reward, float(self.cfg.rew_deviation)), event_reward)
        event_reward = torch.where(success, torch.full_like(event_reward, float(self.cfg.rew_success)), event_reward)
        event_reward = torch.where(truncated, torch.full_like(event_reward, float(self.cfg.rew_timeout)), event_reward)

        reward = reward_raw + event_reward
        reward = torch.clamp(reward, float(self.cfg.reward_clip_min), float(self.cfg.reward_clip_max))
        reward = torch.nan_to_num(reward, nan=0.0, posinf=float(self.cfg.reward_clip_max), neginf=float(self.cfg.reward_clip_min))

        done = terminated | truncated
        done_count = done.float().sum()

        self.total_done_episodes += done_count.detach()
        self.total_success_episodes += success.float().sum().detach()
        self.total_crash_episodes += crash.float().sum().detach()
        self.total_timeout_episodes += truncated.float().sum().detach()

        denom = torch.clamp(self.total_done_episodes, min=1.0)

        info = {
            "reward_components": {
                "R_Alive": r_alive.mean().item(),
                "R_Pos": (float(self.cfg.w_pos) * r_pos).mean().item(),
                "R_Height": (float(self.cfg.w_height) * r_height).mean().item(),
                "R_Upright": (float(self.cfg.w_upright) * r_upright).mean().item(),
                "R_Yaw": (float(self.cfg.w_yaw) * r_yaw).mean().item(),
                "P_Lin_Vel": (float(self.cfg.w_lin_vel) * p_lin_vel).mean().item(),
                "P_Ang_Vel": (float(self.cfg.w_ang_vel) * p_ang_vel).mean().item(),
                "P_Action_Smooth": (float(self.cfg.w_action_smooth) * p_action_smooth).mean().item(),
                "P_Action_Mag": (float(self.cfg.w_action_mag) * p_action_mag).mean().item(),
                "Event": event_reward.mean().item(),
                "Total": reward.mean().item(),
            },
            "events": {
                "Success_Rate": success.float().mean().item(),
                "Stable_Rate": stable.float().mean().item(),
                "Crash_Rate": crash.float().mean().item(),
                "Deviation_Rate": deviation.float().mean().item(),
                "Timeout_Rate": truncated.float().mean().item(),
                "Done_Rate": done.float().mean().item(),
                "Episode_Success_Rate": (self.total_success_episodes / denom).item(),
                "Episode_Crash_Rate": (self.total_crash_episodes / denom).item(),
                "Episode_Timeout_Rate": (self.total_timeout_episodes / denom).item(),
                "Episode_Done_Count": self.total_done_episodes.item(),
            },
            "telemetry": {
                "Pos_Error": pos_error_norm.mean().item(),
                "XY_Error": xy_error.mean().item(),
                "Z_Error": z_error.mean().item(),
                "Z": pos_local[:, 2].mean().item(),
                "Roll": roll.mean().item(),
                "Pitch": pitch.mean().item(),
                "Yaw_Error": yaw_error.mean().item(),
                "RollPitchAbs": roll_pitch_abs.mean().item(),
                "Lin_Vel": lin_vel_norm.mean().item(),
                "Ang_Vel": ang_vel_norm.mean().item(),
                "Action_Mean": self.filtered_actions.mean().item(),
                "Action_Abs": self.filtered_actions.abs().mean().item(),
                "Motor_Multiplier": self.motor_multipliers.mean().item(),
                "Force_Z_W": self.last_force_w[:, 2].mean().item(),
                "Torque_X_W": self.last_torque_w[:, 0].mean().item(),
                "Torque_Y_W": self.last_torque_w[:, 1].mean().item(),
                "Torque_Z_W": self.last_torque_w[:, 2].mean().item(),
                "Success_Counter": self.success_counter.float().mean().item(),
                "Episode_Length": self.episode_steps.float().mean().item(),
                "Episode_Return": self.episode_return.mean().item(),
            },
            "debug": {
                "Actor_Obs_Dim": float(self.num_observations),
                "Critic_Obs_Dim": float(self.num_privileged_obs),
                "Action_Dim": float(self.num_actions),
                "Estimated_Mass": self.estimated_mass.mean().item(),
                "Hover_Thrust": self.hover_thrust.mean().item(),
                "Hover_Thrust_Per_Rotor": self.hover_thrust_per_rotor.mean().item(),
                "Asset_Source_Is_Fallback": float("fallback_usd" in str(self.asset_source)),
                "Reward_Min": reward.min().item(),
                "Reward_Max": reward.max().item(),
            },
            "is_success": success.detach().clone(),
        }

        return reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    def _actions_to_wrench(self, filtered_actions: torch.Tensor):
        motor_mult = 1.0 + float(self.cfg.action_scale) * filtered_actions
        motor_mult = torch.clamp(
            motor_mult,
            float(self.cfg.min_motor_multiplier),
            float(self.cfg.max_motor_multiplier),
        )

        rotor_forces = self.hover_thrust_per_rotor.view(-1, 1) * motor_mult
        total_thrust = rotor_forces.sum(dim=-1)

        max_total = self.hover_thrust * float(self.cfg.max_total_thrust_factor)
        # torch.clamp does not allow min=Number and max=Tensor together.
        # Use elementwise maximum/minimum because hover_thrust is per-env.
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
    def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(x), torch.cos(x))

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
        return QuadrotorTask1Env._quat_rotate(q_inv, v)

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------
    def _print_debug_info(self) -> None:
        print("\n" + "=" * 120)
        print("✅ [Task1] Quadrotor / Crazyflie Hover Stabilization Env Initialized")
        print(f"  asset_source            : {self.asset_source}")
        print(f"  num_envs                : {self.num_envs}")
        print(f"  device                  : {self.device}")
        print(f"  num_bodies              : {getattr(self.drone, 'num_bodies', '<unknown>')}")
        print(f"  num_joints              : {getattr(self.drone, 'num_joints', '<unknown>')}")
        print(f"  body_names              : {list(getattr(self.drone, 'body_names', []))}")
        print(f"  joint_names             : {list(getattr(self.drone, 'joint_names', []))}")
        print(f"  estimated_mass          : {self.estimated_mass.mean().item():.6f} kg")
        print(f"  hover_thrust            : {self.hover_thrust.mean().item():.6f} N")
        print(f"  hover_thrust_per_rotor  : {self.hover_thrust_per_rotor.mean().item():.6f} N")
        print(f"  action_dim              : {self.num_actions}")
        print(f"  single_actor_obs_dim    : {self.cfg.single_actor_obs_dim}")
        print(f"  frame_stack             : {self.cfg.frame_stack}")
        print(f"  actor_obs_dim           : {self.num_observations}")
        print(f"  critic_obs_dim          : {self.num_privileged_obs}")
        print(f"  sim_dt                  : {self.cfg.sim_dt}")
        print(f"  policy_dt               : {self.cfg.policy_dt}")
        print(f"  max_episode_length      : {self.cfg.max_episode_length}")
        print("=" * 120 + "\n")


Task1Env = QuadrotorTask1Env
CrazyflieTask1Env = QuadrotorTask1Env
