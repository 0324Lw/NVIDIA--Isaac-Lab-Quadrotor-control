from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logging.getLogger("isaaclab.assets.articulation").setLevel(logging.ERROR)
logging.getLogger("omni.physx.plugin").setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Train Quadrotor / Crazyflie Task4 Vision Gate Racing with TRUE skrl PPO"
)

# Env / run
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--total-env-steps", type=int, default=50_000_000)
parser.add_argument("--save-freq-env-steps", type=int, default=1_000_000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test-device", type=str, default="cuda:0")

# Checkpoint / warm-start
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--pretrained", type=str, default="")
parser.add_argument("--pretrained-task1", type=str, default="")
parser.add_argument("--pretrained-task2", type=str, default="")
parser.add_argument("--pretrained-task3", type=str, default="")
parser.add_argument("--start-env-steps", type=int, default=0)

# PPO
parser.add_argument("--rollouts", type=int, default=128)
parser.add_argument("--learning-epochs", type=int, default=6)
parser.add_argument("--mini-batches", type=int, default=8)
parser.add_argument("--lr", type=float, default=3.0e-4)
parser.add_argument("--min-lr", type=float, default=1.0e-5)
parser.add_argument("--max-lr", type=float, default=3.0e-4)
parser.add_argument("--discount-factor", type=float, default=0.99)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--kl-threshold", type=float, default=0.015)
parser.add_argument("--ratio-clip", type=float, default=0.20)
parser.add_argument("--value-clip", type=float, default=0.20)
parser.add_argument("--entropy-loss-scale", type=float, default=0.01)
parser.add_argument("--value-loss-scale", type=float, default=1.0)
parser.add_argument("--grad-norm-clip", type=float, default=0.5)

# Policy distribution
parser.add_argument("--init-log-std", type=float, default=-0.55)
parser.add_argument("--min-log-std", type=float, default=-3.0)
parser.add_argument("--max-log-std", type=float, default=0.5)

# Vision network
parser.add_argument("--cnn-output-dim", type=int, default=256)
parser.add_argument("--compact-output-dim", type=int, default=128)
parser.add_argument("--hidden-dim", type=int, default=256)

# Env overrides
parser.add_argument("--max-episode-length-s", type=float, default=12.0)
parser.add_argument("--action-scale", type=float, default=0.50)
parser.add_argument("--action-deadzone", type=float, default=0.05)
parser.add_argument("--action-ema-alpha", type=float, default=0.60)
parser.add_argument("--enable-sensor-noise", action="store_true")
parser.add_argument("--print-debug-info", action="store_true")

# Reward overrides
parser.add_argument("--r-step", type=float, default=-0.25)
parser.add_argument("--r-track-v-scale", type=float, default=0.20)
parser.add_argument("--r-align-pose", type=float, default=0.15)
parser.add_argument("--r-gate-base", type=float, default=15.0)
parser.add_argument("--r-crash", type=float, default=-100.0)
parser.add_argument("--r-success", type=float, default=200.0)

# Logging
parser.add_argument("--log-interval-updates", type=int, default=1)

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.device = str(args_cli.test_device)
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import StepTrainer
from skrl.utils import set_seed

try:
    from skrl.agents.torch.ppo import PPO, PPO_CFG
except ImportError:
    from skrl.agents.torch.ppo import PPO
    from skrl.agents.torch.ppo.ppo_cfg import PPO_CFG

try:
    from skrl.resources.schedulers.torch import KLAdaptiveLR
except ImportError:
    KLAdaptiveLR = None

from quadrotor_rl.tasks.task4.task4_config import Task4Config
from quadrotor_rl.tasks.task4.task4_env import QuadrotorTask4Env
from quadrotor_rl.export.policy_io import save_policy_io


# ======================================================================
# Utilities
# ======================================================================

def to_float(x: Any):
    try:
        if torch.is_tensor(x):
            return float(x.detach().float().mean().cpu().item())
        if isinstance(x, np.ndarray):
            return float(np.mean(x))
        if isinstance(x, (list, tuple)):
            return float(np.mean(x)) if len(x) else None
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
    except Exception:
        return None
    return None


def flat_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in (d or {}).items():
        name = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flat_dict(v, name))
        else:
            val = to_float(v)
            if val is not None and math.isfinite(val):
                out[name] = val
    return out


def write_scalars(writer, data: Dict[str, Any], step: int, prefix: str) -> None:
    if writer is None:
        return

    for k, v in (data or {}).items():
        val = to_float(v)
        if val is not None and math.isfinite(val):
            try:
                writer.add_scalar(f"{prefix}/{k}".replace("//", "/"), val, step)
            except Exception:
                pass


def make_table(title: str, data: Dict[str, Any], width: int = 124) -> str:
    lines = [
        "-" * width,
        f"| {title:<{width - 4}} |",
        "-" * width,
    ]

    if not data:
        lines += [f"| {'<empty>':<{width - 4}} |", "-" * width]
        return "\n".join(lines)

    for k in sorted(data.keys()):
        v = data[k]
        ks = (str(k)[:84] + "...") if len(str(k)) > 87 else str(k)

        if isinstance(v, float):
            if math.isnan(v):
                vs = "nan"
            elif math.isinf(v):
                vs = "inf"
            else:
                vs = f"{v:.6e}" if abs(v) > 1e4 or 0 < abs(v) < 1e-3 else f"{v:.6f}"
        else:
            vs = str(v)

        vs = (vs[:32] + "...") if len(vs) > 35 else vs
        lines.append(f"| {ks:<87} | {vs:>{width - 94}} |")

    lines.append("-" * width)
    return "\n".join(lines)


def tracking_mean(agent) -> Dict[str, float]:
    out: Dict[str, float] = {}

    for k, v in getattr(agent, "tracking_data", {}).items():
        if v is None:
            continue

        try:
            if len(v) == 0:
                continue
        except Exception:
            pass

        try:
            arr = np.asarray(v, dtype=np.float64)
            if arr.size == 0:
                continue
            if k.endswith("(min)"):
                out[k] = float(np.min(arr))
            elif k.endswith("(max)"):
                out[k] = float(np.max(arr))
            else:
                out[k] = float(np.mean(arr))
        except Exception:
            val = to_float(v)
            if val is not None:
                out[k] = val

    return out


def current_lr(agent) -> float:
    for obj in [
        getattr(agent, "optimizer", None),
        getattr(getattr(agent, "scheduler", None), "optimizer", None),
    ]:
        try:
            if obj is not None:
                return float(obj.param_groups[0]["lr"])
        except Exception:
            pass
    return float("nan")


def resolve_checkpoint_path(path: str) -> str:
    if not path:
        return ""

    if os.path.isdir(path):
        for name in [
            "quadrotor_task4_skrl_model.pt",
            "quadrotor_task4_model.pt",
            "quadrotor_task3_skrl_model.pt",
            "quadrotor_task3_model.pt",
            "quadrotor_task2_skrl_model.pt",
            "quadrotor_task2_model.pt",
            "quadrotor_task1_skrl_model.pt",
            "quadrotor_task1_model.pt",
            "agent.pt",
            "model.pt",
            "checkpoint.pt",
        ]:
            p = os.path.join(path, name)
            if os.path.exists(p):
                return p

    return path


def try_load_agent(agent, path: str, label: str) -> bool:
    path = resolve_checkpoint_path(path)
    if not path:
        return False
    if not os.path.exists(path):
        print(f"[WARN] {label} checkpoint 不存在: {path}")
        return False

    print("\n" + "=" * 108)
    print(f"尝试加载 {label}: {path}")
    print("=" * 108)

    try:
        agent.load(path)
        print(f"[OK] 已通过 agent.load() 成功加载 {label}")
        return True
    except Exception as exc:
        print(f"[WARN] agent.load() 加载 {label} 失败: {type(exc).__name__}: {exc}")
        return False


def try_partial_load_policy(models: Dict[str, nn.Module], path: str, label: str) -> bool:
    path = resolve_checkpoint_path(path)
    if not path or not os.path.exists(path):
        return False

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"[WARN] partial {label} load failed to read checkpoint: {type(exc).__name__}: {exc}")
        return False

    src_state = None
    if isinstance(ckpt, dict):
        if "policy" in ckpt and isinstance(ckpt["policy"], dict):
            src_state = ckpt["policy"]
        elif "models" in ckpt and isinstance(ckpt["models"], dict):
            src_state = ckpt["models"].get("policy", None)

    if not isinstance(src_state, dict):
        print(f"[WARN] partial {label} load skipped: no policy state found")
        return False

    dst = models["policy"].state_dict()
    copied = 0
    total = 0

    for key, value in src_state.items():
        if key in dst and tuple(dst[key].shape) == tuple(value.shape):
            dst[key].copy_(value.to(dst[key].device))
            copied += 1
        total += 1

    models["policy"].load_state_dict(dst, strict=True)

    print(f"[INFO] {label} partial policy warm-start copied {copied}/{total} tensors")
    return copied > 0


def sanitize_tensor_inplace(x: torch.Tensor, nan=0.0, posinf=1.0, neginf=-1.0, clamp_abs=None) -> None:
    if x is None or not torch.is_tensor(x):
        return

    with torch.no_grad():
        x.data = torch.nan_to_num(x.data, nan=nan, posinf=posinf, neginf=neginf)
        if clamp_abs is not None:
            x.data.clamp_(-float(clamp_abs), float(clamp_abs))


def sanitize_agent_numerics(agent, models: Dict[str, nn.Module], min_log_std=-3.0, max_log_std=0.5) -> None:
    for _, model in models.items():
        for p in model.parameters():
            sanitize_tensor_inplace(p, nan=0.0, posinf=1.0, neginf=-1.0, clamp_abs=20.0)

        if hasattr(model, "log_std_parameter"):
            with torch.no_grad():
                model.log_std_parameter.data = torch.nan_to_num(
                    model.log_std_parameter.data,
                    nan=float(args_cli.init_log_std),
                    posinf=float(max_log_std),
                    neginf=float(min_log_std),
                )
                model.log_std_parameter.data.clamp_(float(min_log_std), float(max_log_std))

    opt = getattr(agent, "optimizer", None)
    if opt is not None:
        for state in opt.state.values():
            for _, v in state.items():
                if torch.is_tensor(v):
                    with torch.no_grad():
                        v.data = torch.nan_to_num(v.data, nan=0.0, posinf=1.0, neginf=-1.0)
                        v.data.clamp_(-100.0, 100.0)


def ppo_info_has_nan(ppo_info: Dict[str, Any]) -> tuple[bool, str]:
    keys_to_check = [
        "Loss / Entropy loss",
        "Loss / Policy loss",
        "Loss / Value loss",
        "Policy / Standard deviation",
        "Learning / Learning rate",
        "learning_rate",
    ]

    for k in keys_to_check:
        if k in ppo_info:
            val = to_float(ppo_info[k])
            if val is not None and not math.isfinite(val):
                return True, k

    for k, v in ppo_info.items():
        if "Loss" in k or "Standard deviation" in k or "Learning" in k:
            val = to_float(v)
            if val is not None and not math.isfinite(val):
                return True, k

    return False, ""


def save_normalizers(agent, save_dir: str) -> None:
    names = [
        "observation_preprocessor",
        "state_preprocessor",
        "value_preprocessor",
        "_observation_preprocessor",
        "_state_preprocessor",
        "_value_preprocessor",
    ]

    for name in names:
        obj = getattr(agent, name, None)
        if obj is not None:
            try:
                torch.save(obj.state_dict(), os.path.join(save_dir, f"{name}.pt"))
            except Exception:
                pass


def extract_preprocessor_state(agent, names):
    for name in names:
        obj = getattr(agent, name, None)
        if obj is not None:
            try:
                return obj.state_dict()
            except Exception:
                pass
    return None


def save_training_metadata(
    path: str,
    env_cfg: Task4Config,
    base_env: QuadrotorTask4Env,
    env,
    args,
    env_steps: int,
    extra: Dict[str, Any] | None = None,
) -> None:
    try:
        cfg_dict = dataclasses.asdict(env_cfg)
    except Exception:
        cfg_dict = {}

    metadata = {
        "stage": "quadrotor_task4_vision_gate_racing_skrl",
        "uses_skrl": True,
        "algorithm": "skrl PPO",
        "global_env_steps": int(env_steps),
        "num_envs": int(base_env.num_envs),
        "actor_obs_dim": int(env.observation_space.shape[0]),
        "critic_obs_dim": int(env.state_space.shape[0]),
        "action_dim": int(env.action_space.shape[0]),
        "depth_dim": int(base_env.cfg.depth_dim),
        "compact_state_dim": int(base_env.cfg.compact_state_dim),
        "depth_shape": [int(base_env.cfg.depth_channels), int(base_env.cfg.cam_res_h), int(base_env.cfg.cam_res_w)],
        "num_gates": int(base_env.cfg.num_gates),
        "max_episode_length_s": float(env_cfg.max_episode_length_s),
        "max_episode_length": int(env_cfg.max_episode_length),
        "policy_dt": float(env_cfg.policy_dt),
        "asset_source": str(getattr(base_env, "asset_source", "unknown")),
        "num_bodies": int(getattr(base_env.drone, "num_bodies", -1)),
        "num_joints": int(getattr(base_env.drone, "num_joints", -1)),
        "body_names": list(getattr(base_env.drone, "body_names", [])),
        "joint_names": list(getattr(base_env.drone, "joint_names", [])),
        "estimated_mass": float(base_env.estimated_mass.mean().detach().cpu().item()),
        "hover_thrust": float(base_env.hover_thrust.mean().detach().cpu().item()),
        "args": vars(args),
        "env_cfg": cfg_dict,
        "extra": extra or {},
    }

    torch.save(metadata, os.path.join(path, "train_metadata.pt"))


def save_eval_checkpoint(
    path: str,
    agent,
    models: Dict[str, nn.Module],
    env_cfg: Task4Config,
    base_env: QuadrotorTask4Env,
    env,
    args,
    env_steps: int,
    extra: Dict[str, Any] | None = None,
) -> None:
    obs_norm = extract_preprocessor_state(
        agent,
        ["observation_preprocessor", "_observation_preprocessor"],
    )
    state_norm = extract_preprocessor_state(
        agent,
        ["state_preprocessor", "_state_preprocessor"],
    )
    value_norm = extract_preprocessor_state(
        agent,
        ["value_preprocessor", "_value_preprocessor"],
    )

    ckpt = {
        "policy": models["policy"].state_dict(),
        "value": models["value"].state_dict(),
        "actor_obs_norm": obs_norm,
        "critic_obs_norm": state_norm,
        "value_norm": value_norm,
        "env_steps": int(env_steps),
        "args": vars(args),
        "metadata": {
            "uses_skrl": True,
            "algorithm": "skrl PPO",
            "task": "quadrotor_task4_vision_gate_racing",
            "num_envs": int(base_env.num_envs),
            "actor_obs_dim": int(env.observation_space.shape[0]),
            "critic_obs_dim": int(env.state_space.shape[0]),
            "action_dim": int(env.action_space.shape[0]),
            "depth_dim": int(base_env.cfg.depth_dim),
            "compact_state_dim": int(base_env.cfg.compact_state_dim),
            "depth_channels": int(base_env.cfg.depth_channels),
            "cam_res_h": int(base_env.cfg.cam_res_h),
            "cam_res_w": int(base_env.cfg.cam_res_w),
            "num_gates": int(base_env.cfg.num_gates),
            "policy_dt": float(env_cfg.policy_dt),
            "asset_source": str(getattr(base_env, "asset_source", "unknown")),
            "estimated_mass": float(base_env.estimated_mass.mean().detach().cpu().item()),
            "hover_thrust": float(base_env.hover_thrust.mean().detach().cpu().item()),
            "extra": extra or {},
        },
    }

    torch.save(ckpt, os.path.join(path, "quadrotor_task4_model.pt"))


def save_all_checkpoints(
    save_dir: str,
    agent,
    models: Dict[str, nn.Module],
    env_cfg: Task4Config,
    base_env: QuadrotorTask4Env,
    env,
    args,
    env_steps: int,
    extra: Dict[str, Any] | None = None,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    sanitize_agent_numerics(agent, models, args.min_log_std, args.max_log_std)

    agent.save(os.path.join(save_dir, "quadrotor_task4_skrl_model.pt"))
    save_normalizers(agent, save_dir)
    save_training_metadata(
        path=save_dir,
        env_cfg=env_cfg,
        base_env=base_env,
        env=env,
        args=args,
        env_steps=env_steps,
        extra=extra,
    )
    save_eval_checkpoint(
        path=save_dir,
        agent=agent,
        models=models,
        env_cfg=env_cfg,
        base_env=base_env,
        env=env,
        args=args,
        env_steps=env_steps,
        extra=extra,
    )
    save_policy_io(
        save_dir=save_dir,
        task_name="quadrotor_task4_vision_gate_racing",
        env_cfg=env_cfg,
        base_env=base_env,
        env=env,
        args=args,
        env_steps=env_steps,
        extra=extra,
    )


# ======================================================================
# skrl IsaacLab protocol adapter
# ======================================================================

class QuadrotorTask4SkrlWrapper(gym.Env):
    """Adapter from tested Tensor Gym API to skrl IsaacLab dict observations."""

    metadata = {"render_modes": []}

    def __init__(self, env: QuadrotorTask4Env):
        super().__init__()

        self.env = env
        self.num_envs = int(env.num_envs)
        self.device = env.device

        self.observation_space = env.observation_space
        self.state_space = env.state_space
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
        obs, info = self.env.reset(seed=seed, options=options)

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
            self.env.close()
        except Exception:
            pass


# ======================================================================
# Vision Actor-Critic
# ======================================================================

class Task4VisionEncoder(nn.Module):
    """CNN for 64x64 depth + MLP for compact state."""

    def __init__(
        self,
        depth_dim: int = 4096,
        compact_dim: int = 32,
        cnn_output_dim: int = 256,
        compact_output_dim: int = 128,
    ):
        super().__init__()

        self.depth_dim = int(depth_dim)
        self.compact_dim = int(compact_dim)
        self.cnn_output_dim = int(cnn_output_dim)
        self.compact_output_dim = int(compact_output_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, 64, 64)
            flat = int(self.cnn(dummy).shape[1])

        self.cnn_linear = nn.Sequential(
            nn.Linear(flat, self.cnn_output_dim),
            nn.ELU(),
        )

        self.compact_mlp = nn.Sequential(
            nn.Linear(self.compact_dim, 128),
            nn.ELU(),
            nn.Linear(128, self.compact_output_dim),
            nn.ELU(),
        )

        self.output_dim = self.cnn_output_dim + self.compact_output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        x = torch.clamp(x, -10.0, 10.0)

        depth = x[:, : self.depth_dim].reshape(-1, 1, 64, 64)
        compact = x[:, self.depth_dim : self.depth_dim + self.compact_dim]

        depth = torch.clamp(depth, 0.0, 1.0)
        compact = torch.clamp(compact, -10.0, 10.0)

        depth_feat = self.cnn_linear(self.cnn(depth))
        compact_feat = self.compact_mlp(compact)

        return torch.cat([depth_feat, compact_feat], dim=-1)


class QuadrotorTask4Actor(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        init_log_std: float = -0.55,
        min_log_std: float = -3.0,
        max_log_std: float = 0.5,
        cnn_output_dim: int = 256,
        compact_output_dim: int = 128,
        hidden_dim: int = 256,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=True,
            clip_log_std=True,
            min_log_std=float(min_log_std),
            max_log_std=float(max_log_std),
            reduction="sum",
        )

        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.init_log_std = float(init_log_std)

        act_dim = int(action_space.shape[0])

        self.encoder = Task4VisionEncoder(
            depth_dim=4096,
            compact_dim=32,
            cnn_output_dim=int(cnn_output_dim),
            compact_output_dim=int(compact_output_dim),
        )

        self.net = nn.Sequential(
            nn.Linear(self.encoder.output_dim, int(hidden_dim)),
            nn.ELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ELU(),
            nn.Linear(int(hidden_dim), 128),
            nn.ELU(),
            nn.Linear(128, act_dim),
        )

        self.log_std_parameter = nn.Parameter(
            torch.full((act_dim,), float(init_log_std), dtype=torch.float32)
        )

        self.apply(self._orthogonal_init)

        with torch.no_grad():
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                last.weight.mul_(0.03)
                last.bias.zero_()

    @staticmethod
    def _orthogonal_init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def compute(self, inputs, role):
        x = inputs.get("observations", inputs.get("states"))
        feat = self.encoder(x)

        mean = self.net(feat)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
        mean = torch.clamp(mean, -5.0, 5.0)

        log_std = torch.clamp(self.log_std_parameter, self.min_log_std, self.max_log_std)
        log_std = torch.nan_to_num(
            log_std,
            nan=self.init_log_std,
            posinf=self.max_log_std,
            neginf=self.min_log_std,
        )

        return mean, {"log_std": log_std}


class QuadrotorTask4Critic(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        cnn_output_dim: int = 256,
        compact_output_dim: int = 128,
        hidden_dim: int = 256,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False)

        self.encoder = Task4VisionEncoder(
            depth_dim=4096,
            compact_dim=32,
            cnn_output_dim=int(cnn_output_dim),
            compact_output_dim=int(compact_output_dim),
        )

        self.net = nn.Sequential(
            nn.Linear(self.encoder.output_dim, int(hidden_dim)),
            nn.ELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ELU(),
            nn.Linear(int(hidden_dim), 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

        self.apply(QuadrotorTask4Actor._orthogonal_init)

    def compute(self, inputs, role):
        x = inputs.get("states", inputs.get("observations"))
        feat = self.encoder(x)

        value = self.net(feat)
        value = torch.nan_to_num(value, nan=0.0, posinf=100.0, neginf=-100.0)
        value = torch.clamp(value, -500.0, 500.0)

        return value, {}


# ======================================================================
# PPO config / print
# ======================================================================

def build_skrl_cfg(env, log_dir: str) -> Dict[str, Any]:
    default_cfg = PPO_CFG()

    if dataclasses.is_dataclass(default_cfg):
        cfg = dataclasses.asdict(default_cfg)
    elif isinstance(default_cfg, dict):
        cfg = default_cfg.copy()
    else:
        cfg = dict(default_cfg.__dict__)

    cfg.update(
        {
            "rollouts": int(args_cli.rollouts),
            "learning_epochs": int(args_cli.learning_epochs),
            "mini_batches": int(args_cli.mini_batches),
            "discount_factor": float(args_cli.discount_factor),
            "gae_lambda": float(args_cli.gae_lambda),
            "learning_rate": float(args_cli.lr),
            "grad_norm_clip": float(args_cli.grad_norm_clip),
            "ratio_clip": float(args_cli.ratio_clip),
            "value_clip": float(args_cli.value_clip),
            "entropy_loss_scale": float(args_cli.entropy_loss_scale),
            "value_loss_scale": float(args_cli.value_loss_scale),
            "observation_preprocessor": RunningStandardScaler,
            "observation_preprocessor_kwargs": {
                "size": env.observation_space,
                "device": env.device,
            },
            "state_preprocessor": RunningStandardScaler,
            "state_preprocessor_kwargs": {
                "size": env.state_space,
                "device": env.device,
            },
            "value_preprocessor": RunningStandardScaler,
            "value_preprocessor_kwargs": {
                "size": 1,
                "device": env.device,
            },
        }
    )

    if KLAdaptiveLR is not None:
        cfg["learning_rate_scheduler"] = KLAdaptiveLR
        cfg["learning_rate_scheduler_kwargs"] = {
            "kl_threshold": float(args_cli.kl_threshold),
            "min_lr": float(args_cli.min_lr),
            "max_lr": float(args_cli.max_lr),
        }

    cfg.setdefault("experiment", {})
    cfg["experiment"].update(
        {
            "directory": log_dir,
            "experiment_name": "quadrotor_task4_vision_gate_racing",
            "write_interval": 1_000_000,
            "checkpoint_interval": 0,
        }
    )

    return cfg


def print_update(
    pbar,
    update_id: int,
    env_steps: int,
    total_steps: int,
    elapsed: float,
    num_envs: int,
    rollouts: int,
    info: Dict[str, Any],
    ppo: Dict[str, Any],
    lr: float,
):
    stat = {
        "update": float(update_id),
        "env_steps": float(env_steps),
        "target_env_steps": float(total_steps),
        "progress_percent": 100.0 * env_steps / max(total_steps, 1),
        "num_envs": float(num_envs),
        "rollouts_per_update": float(rollouts),
        "fps_env_steps": env_steps / max(elapsed, 1.0e-6),
        "learning_rate": lr,
    }

    tel = info.get("telemetry", {}) if isinstance(info, dict) else {}
    ev = info.get("events", {}) if isinstance(info, dict) else {}
    rew = info.get("reward_components", {}) if isinstance(info, dict) else {}

    pbar.write(
        "\n".join(
            [
                "\n" + "=" * 124,
                f"📊 [Quadrotor Task4 PPO 更新 {update_id}] "
                f"步数: {env_steps:,} / {total_steps:,} | "
                f"FPS: {stat['fps_env_steps']:,.0f} | LR: {lr:.3e} | "
                f"Gate: {tel.get('Target_Gate_Idx', 0):.2f} | "
                f"Passed_Gates: {tel.get('Passed_Gates', 0):.2f} | "
                f"Centerline: {tel.get('Centerline_Dist', 0):.3f} | "
                f"Depth_Min: {tel.get('Depth_Min', 0):.3f} | "
                f"Pose_Align: {tel.get('Pose_Align', 0):+.3f} | "
                f"Succ: {ev.get('Success_Rate', 0):.3f} | "
                f"Gate_Collision_Rate: {ev.get('Gate_Collision_Rate', 0):.3f} | "
                f"Missed_Gate_Rate: {ev.get('Missed_Gate_Rate', 0):.3f} | "
                f"Dev: {ev.get('Deviation_Rate', 0):.3f} | "
                f"R: {rew.get('Total', 0):+.3f}",
                "=" * 124,
                make_table("time / progress", stat),
                make_table("env info: rewards + events + telemetry + debug", flat_dict(info)),
                make_table("ppo update info", ppo),
                "=" * 124 + "\n",
            ]
        )
    )


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    set_seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    log_root = PROJECT_ROOT / "logs" / "task4"
    run_name = f"quadrotor_task4_skrl_ppo_vision_gate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = str(log_root / run_name)
    os.makedirs(log_dir, exist_ok=True)

    print("\n" + "=" * 124)
    print("🚁 Quadrotor / Crazyflie Task4: Vision Gate Racing TRUE skrl PPO Training")
    print("=" * 124)
    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] log_root     = {log_root}")
    print(f"[INFO] run_name     = {run_name}")
    print("[INFO] This version uses Isaac Lab + skrl PPO + CNN depth encoder.")

    env_cfg = Task4Config()
    env_cfg.num_envs = int(args_cli.num_envs)
    env_cfg.device = str(args_cli.test_device)
    env_cfg.seed = int(args_cli.seed)
    env_cfg.max_episode_length_s = float(args_cli.max_episode_length_s)
    env_cfg.action_scale = float(args_cli.action_scale)
    env_cfg.action_deadzone = float(args_cli.action_deadzone)
    env_cfg.action_ema_alpha = float(args_cli.action_ema_alpha)
    env_cfg.enable_sensor_noise = bool(args_cli.enable_sensor_noise)
    env_cfg.print_debug_info = bool(args_cli.print_debug_info)

    env_cfg.r_step = float(args_cli.r_step)
    env_cfg.r_track_v_scale = float(args_cli.r_track_v_scale)
    env_cfg.r_align_pose = float(args_cli.r_align_pose)
    env_cfg.r_gate_base = float(args_cli.r_gate_base)
    env_cfg.r_crash = float(args_cli.r_crash)
    env_cfg.r_success = float(args_cli.r_success)

    env_cfg.validate()

    base_env = QuadrotorTask4Env(env_cfg)
    skrl_base_env = QuadrotorTask4SkrlWrapper(base_env)
    env = wrap_env(skrl_base_env, wrapper="isaaclab")

    num_envs = int(getattr(env, "num_envs", skrl_base_env.num_envs))

    print("\n[DEBUG] Quadrotor Task4 skrl spaces")
    print(f"  num_envs              = {num_envs}")
    print(f"  env.observation_space = {env.observation_space}")
    print(f"  env.state_space       = {env.state_space}")
    print(f"  env.action_space      = {env.action_space}")
    print(f"  policy input dim      = {env.observation_space.shape[0]}")
    print(f"  critic input dim      = {env.state_space.shape[0]}")
    print(f"  action dim            = {env.action_space.shape[0]}")
    print(f"  depth_dim             = {env_cfg.depth_dim}")
    print(f"  compact_state_dim     = {env_cfg.compact_state_dim}")
    print(f"  asset source          = {base_env.asset_source}")
    print(f"  estimated mass        = {base_env.estimated_mass.mean().item():.6f} kg")

    assert int(env.observation_space.shape[0]) == int(env_cfg.actor_obs_dim)
    assert int(env.state_space.shape[0]) == int(env_cfg.critic_obs_dim)
    assert int(env.action_space.shape[0]) == int(env_cfg.action_dim)

    writer = SummaryWriter(log_dir)

    models = {
        "policy": QuadrotorTask4Actor(
            env.observation_space,
            env.state_space,
            env.action_space,
            env.device,
            init_log_std=args_cli.init_log_std,
            min_log_std=args_cli.min_log_std,
            max_log_std=args_cli.max_log_std,
            cnn_output_dim=args_cli.cnn_output_dim,
            compact_output_dim=args_cli.compact_output_dim,
            hidden_dim=args_cli.hidden_dim,
        ),
        "value": QuadrotorTask4Critic(
            env.observation_space,
            env.state_space,
            env.action_space,
            env.device,
            cnn_output_dim=args_cli.cnn_output_dim,
            compact_output_dim=args_cli.compact_output_dim,
            hidden_dim=args_cli.hidden_dim,
        ),
    }

    cfg = build_skrl_cfg(env, log_dir=log_dir)

    total_env_steps = int(args_cli.total_env_steps)
    start_env_steps = int(args_cli.start_env_steps)
    remaining_env_steps = max(total_env_steps - start_env_steps, 1)
    total_vector_steps = int(math.ceil(remaining_env_steps / max(num_envs, 1)))
    save_freq_env_steps = int(args_cli.save_freq_env_steps)
    update_env_steps = int(cfg["rollouts"] * num_envs)

    cfg.setdefault("experiment", {})
    cfg["experiment"]["write_interval"] = max(total_vector_steps + 1, 1_000_000)
    cfg["experiment"]["checkpoint_interval"] = 0

    print("\n[INFO] Quadrotor Task4 skrl PPO 训练配置")
    print(f"  - num_envs              : {num_envs:,}")
    print(f"  - total_env_steps       : {total_env_steps:,}")
    print(f"  - start_env_steps       : {start_env_steps:,}")
    print(f"  - remaining_env_steps   : {remaining_env_steps:,}")
    print(f"  - total_vector_steps    : {total_vector_steps:,}")
    print(f"  - rollouts              : {cfg['rollouts']}")
    print(f"  - update_env_steps      : {update_env_steps:,}")
    print(f"  - save_freq_env_steps   : {save_freq_env_steps:,}")
    print(f"  - actor_obs_dim         : {env.observation_space.shape[0]}")
    print(f"  - critic_obs_dim        : {env.state_space.shape[0]}")
    print(f"  - action_dim            : {env.action_space.shape[0]}")
    print(f"  - depth_dim             : {env_cfg.depth_dim}")
    print(f"  - compact_state_dim     : {env_cfg.compact_state_dim}")
    print(f"  - gates                 : {env_cfg.num_gates}")
    print(f"  - camera                : {env_cfg.cam_res_w} x {env_cfg.cam_res_h}, fov={env_cfg.cam_fov_deg}")
    print(f"  - max_episode_length_s  : {env_cfg.max_episode_length_s}")
    print(f"  - max_episode_length    : {env_cfg.max_episode_length}")
    print(f"  - action_scale          : {env_cfg.action_scale}")
    print(f"  - action_deadzone       : {env_cfg.action_deadzone}")
    print(f"  - action_ema_alpha      : {env_cfg.action_ema_alpha}")
    print(f"  - enable_sensor_noise   : {env_cfg.enable_sensor_noise}")
    print(f"  - lr/min/max            : {args_cli.lr} / {args_cli.min_lr} / {args_cli.max_lr}")
    print(f"  - gamma                 : {args_cli.discount_factor}")
    print(f"  - gae_lambda            : {args_cli.gae_lambda}")
    print(f"  - entropy_loss_scale    : {args_cli.entropy_loss_scale}")
    print(f"  - init_log_std          : {args_cli.init_log_std}")
    print(f"  - cnn_output_dim        : {args_cli.cnn_output_dim}")
    print(f"  - compact_output_dim    : {args_cli.compact_output_dim}")
    print(f"  - hidden_dim            : {args_cli.hidden_dim}")
    print(f"  - resume                : {args_cli.resume if args_cli.resume else '<none>'}")
    print(f"  - pretrained_task3      : {args_cli.pretrained_task3 if args_cli.pretrained_task3 else '<none>'}")
    print(f"  - tensorboard           : tensorboard --logdir={PROJECT_ROOT / 'logs'}")

    memory = RandomMemory(
        memory_size=int(cfg["rollouts"]),
        num_envs=num_envs,
        device=env.device,
    )

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=env.device,
    )

    pretrained_loaded = False
    resumed = False
    task1_partial_loaded = False
    task2_partial_loaded = False
    task3_partial_loaded = False

    if args_cli.resume:
        resumed = try_load_agent(agent, args_cli.resume, "resume checkpoint")
        sanitize_agent_numerics(agent, models, args_cli.min_log_std, args_cli.max_log_std)
    elif args_cli.pretrained:
        pretrained_loaded = try_load_agent(agent, args_cli.pretrained, "pretrained checkpoint")
        sanitize_agent_numerics(agent, models, args_cli.min_log_std, args_cli.max_log_std)
    elif args_cli.pretrained_task3:
        task3_partial_loaded = try_partial_load_policy(models, args_cli.pretrained_task3, "Task3")
        sanitize_agent_numerics(agent, models, args_cli.min_log_std, args_cli.max_log_std)
    elif args_cli.pretrained_task2:
        task2_partial_loaded = try_partial_load_policy(models, args_cli.pretrained_task2, "Task2")
        sanitize_agent_numerics(agent, models, args_cli.min_log_std, args_cli.max_log_std)
    elif args_cli.pretrained_task1:
        task1_partial_loaded = try_partial_load_policy(models, args_cli.pretrained_task1, "Task1")
        sanitize_agent_numerics(agent, models, args_cli.min_log_std, args_cli.max_log_std)

    trainer = StepTrainer(
        cfg={
            "timesteps": total_vector_steps,
            "headless": True,
            "disable_progressbar": True,
        },
        env=env,
        agents=agent,
    )

    print("\n🔥 [Quadrotor Task4 TRUE skrl PPO 已启动]")
    print("👉 重点观察：Passed_Gates / Centerline_Dist / Depth_Min / Pose_Align / Gate_Collision_Rate / Missed_Gate_Rate / Success_Rate")
    print(f"👉 TensorBoard: tensorboard --logdir={PROJECT_ROOT / 'logs'}\n")

    last_save = start_env_steps
    update_id = 0
    start_time = time.time()
    absolute_env_steps = start_env_steps

    try:
        trainer.reset()

        with tqdm(
            total=total_env_steps,
            initial=start_env_steps,
            desc="Quadrotor Task4 skrl PPO",
            unit="steps",
            dynamic_ncols=True,
            mininterval=0.5,
        ) as pbar:
            for t in range(total_vector_steps):
                absolute_env_steps = min(start_env_steps + (t + 1) * num_envs, total_env_steps)

                trainer.train(timestep=t, timesteps=total_vector_steps)

                prev_steps = min(start_env_steps + t * num_envs, total_env_steps)
                pbar.update(absolute_env_steps - prev_steps)

                info = getattr(skrl_base_env, "last_info", None)
                if info is None:
                    info = {}

                if not info and hasattr(base_env, "episode_steps"):
                    root_local = base_env._root_pos_w() - base_env.env_origins
                    info = {
                        "telemetry": {
                            "Episode_Length": base_env.episode_steps.float().mean().item(),
                            "Episode_Return": base_env.episode_return.float().mean().item(),
                            "Target_Gate_Idx": base_env.world.target_gate_idx.float().mean().item(),
                            "Depth_Min": base_env.current_depth.view(base_env.num_envs, -1).min(dim=-1).values.mean().item(),
                            "Depth_Mean": base_env.current_depth.mean().item(),
                            "Pos_Z": root_local[:, 2].mean().item(),
                            "Action_Abs": base_env.filtered_actions.abs().mean().item(),
                        },
                        "events": {},
                        "reward_components": {},
                    }

                write_scalars(writer, flat_dict(info), absolute_env_steps, "env_info")

                tel = info.get("telemetry", {})
                ev = info.get("events", {})
                rew = info.get("reward_components", {})

                pbar.set_postfix(
                    {
                        "steps": f"{absolute_env_steps:,}",
                        "fps": f"{(absolute_env_steps - start_env_steps) / max(time.time() - start_time, 1e-6):,.0f}",
                        "gate": f"{tel.get('Target_Gate_Idx', 0):.2f}",
                        "center": f"{tel.get('Centerline_Dist', 0):.2f}",
                        "depth": f"{tel.get('Depth_Min', 0):.2f}",
                        "succ": f"{ev.get('Success_Rate', 0):.3f}",
                        "hit": f"{ev.get('Gate_Collision_Rate', 0):.3f}",
                        "miss": f"{ev.get('Missed_Gate_Rate', 0):.3f}",
                        "rew": f"{rew.get('Total', 0):+.3f}",
                    }
                )

                if (t + 1) % 32 == 0:
                    sanitize_agent_numerics(agent, models, args_cli.min_log_std, args_cli.max_log_std)

                if (t + 1) % int(cfg["rollouts"]) == 0:
                    update_id += 1
                    sanitize_agent_numerics(agent, models, args_cli.min_log_std, args_cli.max_log_std)

                    ppo_info = tracking_mean(agent)
                    ppo_info["learning_rate"] = current_lr(agent)

                    write_scalars(writer, ppo_info, absolute_env_steps, "ppo")

                    bad, bad_key = ppo_info_has_nan(ppo_info)

                    if update_id % max(int(args_cli.log_interval_updates), 1) == 0:
                        print_update(
                            pbar=pbar,
                            update_id=update_id,
                            env_steps=absolute_env_steps,
                            total_steps=total_env_steps,
                            elapsed=time.time() - start_time,
                            num_envs=num_envs,
                            rollouts=int(cfg["rollouts"]),
                            info=info,
                            ppo=ppo_info,
                            lr=ppo_info["learning_rate"],
                        )

                    try:
                        agent.tracking_data.clear()
                    except Exception:
                        pass

                    if bad:
                        emergency_dir = os.path.join(log_dir, f"emergency_nan_checkpoint_{absolute_env_steps}")
                        save_all_checkpoints(
                            save_dir=emergency_dir,
                            agent=agent,
                            models=models,
                            env_cfg=env_cfg,
                            base_env=base_env,
                            env=env,
                            args=args_cli,
                            env_steps=absolute_env_steps,
                            extra={
                                "reason": f"ppo_nan_detected: {bad_key}",
                                "last_info": info,
                                "ppo_info": ppo_info,
                            },
                        )
                        raise RuntimeError(
                            f"PPO 数值异常：{bad_key}=NaN/Inf。已保存 emergency checkpoint: {emergency_dir}"
                        )

                if absolute_env_steps - last_save >= save_freq_env_steps:
                    last_save = absolute_env_steps
                    save_dir = os.path.join(log_dir, f"checkpoint_{absolute_env_steps}")
                    try:
                        save_all_checkpoints(
                            save_dir=save_dir,
                            agent=agent,
                            models=models,
                            env_cfg=env_cfg,
                            base_env=base_env,
                            env=env,
                            args=args_cli,
                            env_steps=absolute_env_steps,
                            extra={
                                "pretrained_loaded": pretrained_loaded,
                                "resumed": resumed,
                                "task1_partial_loaded": task1_partial_loaded,
                                "task2_partial_loaded": task2_partial_loaded,
                                "task3_partial_loaded": task3_partial_loaded,
                                "last_info": info,
                            },
                        )
                        pbar.write(
                            f"\n💾 [Quadrotor Task4 skrl 备份] "
                            f"总步数: {absolute_env_steps:,} | 已保存至: {save_dir}\n"
                        )
                    except Exception as exc:
                        pbar.write(f"\n[WARN] checkpoint 保存失败: {type(exc).__name__}: {exc}\n")

    except KeyboardInterrupt:
        print("\n[WARN] 接收到手动中断信号，正在安全保存...")
    except Exception:
        print("\n[ERROR] Quadrotor Task4 skrl PPO 训练过程中发生真实异常：")
        traceback.print_exc()
    finally:
        final_dir = os.path.join(log_dir, "final_checkpoint")
        os.makedirs(final_dir, exist_ok=True)

        try:
            info = getattr(skrl_base_env, "last_info", {})
            save_all_checkpoints(
                save_dir=final_dir,
                agent=agent,
                models=models,
                env_cfg=env_cfg,
                base_env=base_env,
                env=env,
                args=args_cli,
                env_steps=absolute_env_steps,
                extra={
                    "final": True,
                    "pretrained_loaded": pretrained_loaded,
                    "resumed": resumed,
                    "task1_partial_loaded": task1_partial_loaded,
                    "task2_partial_loaded": task2_partial_loaded,
                    "task3_partial_loaded": task3_partial_loaded,
                    "last_info": info,
                },
            )
            print(f"✅ Quadrotor Task4 skrl 模型与归一化统计已保存至 {final_dir}")
        except Exception as exc:
            print(f"[WARN] 保存最终模型失败: {type(exc).__name__}: {exc}")

        try:
            writer.flush()
            writer.close()
        except Exception:
            pass

        try:
            env.close()
        except Exception:
            pass

        try:
            simulation_app.close()
        except Exception:
            pass

        print("✅ Quadrotor Task4 skrl PPO 训练管线安全退出")


if __name__ == "__main__":
    main()
