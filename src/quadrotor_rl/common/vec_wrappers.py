from __future__ import annotations

import torch


class ActionClipWrapper:
    """Minimal action clipping wrapper for tensor-vectorized environments."""

    def __init__(self, env, low: float = -1.0, high: float = 1.0):
        self.env = env
        self.low = float(low)
        self.high = float(high)

        self.num_envs = getattr(env, "num_envs", None)
        self.device = getattr(env, "device", "cpu")
        self.observation_space = getattr(env, "observation_space", None)
        self.state_space = getattr(env, "state_space", None)
        self.action_space = getattr(env, "action_space", None)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, actions):
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=self.high, neginf=self.low)
        actions = torch.clamp(actions, self.low, self.high)
        return self.env.step(actions)

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return getattr(self.env, "unwrapped", self.env)


class ObservationClampWrapper:
    """Clamp tensor observations returned by a vectorized environment."""

    def __init__(self, env, obs_clip: float = 10.0):
        self.env = env
        self.obs_clip = float(obs_clip)

        self.num_envs = getattr(env, "num_envs", None)
        self.device = getattr(env, "device", "cpu")
        self.observation_space = getattr(env, "observation_space", None)
        self.state_space = getattr(env, "state_space", None)
        self.action_space = getattr(env, "action_space", None)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        return self._clamp(obs), info

    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions)
        return self._clamp(obs), reward, terminated, truncated, info

    def _clamp(self, obs):
        if torch.is_tensor(obs):
            return torch.nan_to_num(
                torch.clamp(obs, -self.obs_clip, self.obs_clip),
                nan=0.0,
                posinf=self.obs_clip,
                neginf=-self.obs_clip,
            )
        return obs

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return getattr(self.env, "unwrapped", self.env)
