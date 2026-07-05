from __future__ import annotations

from typing import Any, Callable, Dict

import torch

from quadrotor_rl.evaluation.eval_metrics import QuadrotorEvalMetrics


@torch.no_grad()
def run_tensor_policy_eval(env, policy_fn: Callable[[torch.Tensor], torch.Tensor], num_steps: int) -> Dict[str, float]:
    obs, info = env.reset()
    del info
    total_reward = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    done_count = 0.0
    success_count = 0.0
    crash_count = 0.0
    action_abs = []
    for _ in range(int(num_steps)):
        actions = policy_fn(obs)
        action_abs.append(actions.abs().mean().detach())
        obs, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        done = terminated | truncated
        done_count += float(done.float().sum().detach().cpu().item())
        events = info.get("events", {}) if isinstance(info, dict) else {}
        success_count += float(events.get("Success_Rate", 0.0)) * env.num_envs
        crash_count += float(events.get("Crash_Rate", 0.0)) * env.num_envs
    denom = max(done_count, 1.0)
    metrics = QuadrotorEvalMetrics(
        success_rate=success_count / denom,
        crash_rate=crash_count / denom,
        mean_episode_return=float(total_reward.mean().detach().cpu().item()),
        mean_action_abs=float(torch.stack(action_abs).mean().detach().cpu().item()) if action_abs else 0.0,
    )
    return metrics.to_dict()
