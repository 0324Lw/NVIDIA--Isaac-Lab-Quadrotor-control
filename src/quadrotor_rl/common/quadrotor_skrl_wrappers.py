from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
import torch


class IsaacLabDictObsWrapper(gym.Env):
    """Wrap tensor-based Gym env into skrl Isaac Lab dict observation protocol.

    The wrapped env is expected to return:
        obs, reward, terminated, truncated, info

    The skrl Isaac Lab wrapper expects:
        {"policy": obs, "critic": state}
    """

    metadata = {"render_modes": []}

    def __init__(self, env: gym.Env):
        super().__init__()
        self.env = env
        self.num_envs = int(getattr(env, "num_envs"))
        self.device = getattr(env, "device", "cpu")

        self.observation_space = env.observation_space
        self.state_space = getattr(env, "state_space", env.observation_space)
        self.action_space = env.action_space

        self.single_observation_space = gym.spaces.Dict(
            {
                "policy": self.observation_space,
                "critic": self.state_space,
            }
        )
        self.single_action_space = self.action_space
        self.last_info: Dict[str, Any] = {}

    @property
    def unwrapped(self):
        return self

    def reset(self, seed=None, options=None, **kwargs):
        obs, info = self.env.reset(seed=seed, options=options, **kwargs)
        if info is None:
            info = {}
        state = info.get("state", self.env.compute_privileged_obs())

        obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        state = torch.nan_to_num(state, nan=0.0, posinf=20.0, neginf=-20.0)

        self.last_info = info
        return {"policy": obs.clone(), "critic": state.clone()}, info

    def step(self, actions):
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        actions = torch.clamp(actions, -1.0, 1.0)

        obs, reward, terminated, truncated, info = self.env.step(actions)
        if info is None:
            info = {}

        state = info.get("state", self.env.compute_privileged_obs())

        obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        state = torch.nan_to_num(state, nan=0.0, posinf=20.0, neginf=-20.0)
        reward = torch.nan_to_num(reward, nan=0.0, posinf=350.0, neginf=-150.0)

        self.last_info = info
        return {"policy": obs.clone(), "critic": state.clone()}, reward, terminated, truncated, info

    def close(self):
        try:
            return self.env.close()
        except Exception:
            return None


QuadrotorSkrlWrapper = IsaacLabDictObsWrapper
