from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Evaluate Quadrotor / Crazyflie Task2 TRUE skrl PPO model"
)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=800)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test-device", type=str, default="cuda:0")
parser.add_argument("--print-interval", type=int, default=20)
parser.add_argument("--max-episode-length-s", type=float, default=16.667)
parser.add_argument("--slow-action-scale", type=float, default=1.0)
parser.add_argument("--hold-success", action="store_true")
parser.add_argument("--visualize", action="store_true")
parser.add_argument("--print-names", action="store_true")
parser.add_argument("--save-plot", type=str, default="")
parser.add_argument("--save-npz", type=str, default="")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.device = args_cli.test_device
args_cli.headless = not bool(args_cli.visualize)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from skrl.models.torch import GaussianMixin, Model

from quadrotor_rl.tasks.task2.task2_config import Task2Config
from quadrotor_rl.tasks.task2.task2_env import QuadrotorTask2Env


# ======================================================================
# Actor
# ======================================================================

class QuadrotorTask2Actor(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        init_log_std: float = -0.65,
        min_log_std: float = -3.0,
        max_log_std: float = 0.5,
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

        obs_dim = int(observation_space.shape[0])
        act_dim = int(action_space.shape[0])

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, act_dim),
        )
        self.log_std_parameter = nn.Parameter(
            torch.full((act_dim,), float(init_log_std), dtype=torch.float32)
        )

    def compute(self, inputs, role):
        states = inputs.get("observations", inputs.get("states"))
        states = torch.nan_to_num(states, nan=0.0, posinf=10.0, neginf=-10.0)
        states = torch.clamp(states, -10.0, 10.0)

        mean = self.net(states)
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

    @torch.no_grad()
    def act_deterministic_direct(self, states: torch.Tensor) -> torch.Tensor:
        actions, _ = self.compute({"states": states}, role="policy")
        return torch.clamp(actions, -1.0, 1.0)


# ======================================================================
# Checkpoint helpers
# ======================================================================

def torch_load_checkpoint(path: Path, device: str):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def resolve_checkpoint(path: str) -> Path:
    p = Path(path).expanduser().resolve()

    if p.is_file():
        return p

    if p.is_dir():
        candidates = [
            p / "quadrotor_task2_model.pt",
            p / "final_checkpoint" / "quadrotor_task2_model.pt",
            p / "checkpoint_5000" / "quadrotor_task2_model.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        pt_files = sorted(p.glob("*.pt"))
        for pt in pt_files:
            if pt.name.endswith("_preprocessor.pt"):
                continue
            if pt.name in {"train_metadata.pt", "quadrotor_task2_skrl_model.pt"}:
                continue
            return pt

    return p


def _find_norm_tensors(norm_state: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not isinstance(norm_state, dict):
        return None, None

    mean = None
    var = None

    mean_keys = [
        "running_mean",
        "_running_mean",
        "mean",
        "_mean",
        "obs_mean",
    ]
    var_keys = [
        "running_variance",
        "_running_variance",
        "variance",
        "_variance",
        "var",
        "_var",
        "obs_var",
    ]

    for key in mean_keys:
        if key in norm_state:
            mean = norm_state[key]
            break

    for key in var_keys:
        if key in norm_state:
            var = norm_state[key]
            break

    return mean, var


def normalize_with_saved_obs_norm(obs: torch.Tensor, obs_norm: Optional[Dict[str, Any]]) -> torch.Tensor:
    if not obs_norm:
        return obs

    mean, var = _find_norm_tensors(obs_norm)
    if mean is None or var is None:
        return obs

    mean = torch.as_tensor(mean, device=obs.device, dtype=torch.float32).view(-1)
    var = torch.as_tensor(var, device=obs.device, dtype=torch.float32).view(-1)

    if mean.numel() != obs.shape[-1] or var.numel() != obs.shape[-1]:
        return obs

    return torch.clamp((obs - mean) / torch.sqrt(var + 1.0e-8), -10.0, 10.0)


def load_policy_checkpoint(ckpt_path: Path, env: QuadrotorTask2Env):
    ckpt = torch_load_checkpoint(ckpt_path, env.device)

    if not isinstance(ckpt, dict) or "policy" not in ckpt:
        raise RuntimeError(
            f"当前测试脚本需要 task2_train.py 保存的 eval checkpoint: quadrotor_task2_model.pt\n"
            f"当前文件不是 eval checkpoint: {ckpt_path}"
        )

    metadata = ckpt.get("metadata", {})
    train_args = ckpt.get("args", {})

    if not bool(metadata.get("uses_skrl", False)):
        raise RuntimeError("checkpoint metadata 缺少 uses_skrl=True，请使用 TRUE skrl 版本重新训练。")

    task_name = str(metadata.get("task", ""))
    if "task2" not in task_name and "trajectory" not in task_name:
        raise RuntimeError(f"checkpoint task metadata 不是 Task2 trajectory tracking: {task_name}")

    expected_actor_dim = int(metadata.get("actor_obs_dim", env.observation_space.shape[0]))
    expected_critic_dim = int(metadata.get("critic_obs_dim", env.state_space.shape[0]))
    expected_action_dim = int(metadata.get("action_dim", env.action_space.shape[0]))
    expected_single_dim = int(metadata.get("single_actor_obs_dim", env.cfg.single_actor_obs_dim))
    expected_stack = int(metadata.get("frame_stack", env.cfg.frame_stack))
    expected_lookahead = int(metadata.get("lookahead_steps", env.cfg.lookahead_steps))

    if expected_actor_dim != env.observation_space.shape[0]:
        raise RuntimeError(f"actor obs dim mismatch: checkpoint={expected_actor_dim}, env={env.observation_space.shape[0]}")

    if expected_critic_dim != env.state_space.shape[0]:
        raise RuntimeError(f"critic obs dim mismatch: checkpoint={expected_critic_dim}, env={env.state_space.shape[0]}")

    if expected_action_dim != env.action_space.shape[0]:
        raise RuntimeError(f"action dim mismatch: checkpoint={expected_action_dim}, env={env.action_space.shape[0]}")

    if expected_single_dim != env.cfg.single_actor_obs_dim:
        raise RuntimeError(f"single obs dim mismatch: checkpoint={expected_single_dim}, env={env.cfg.single_actor_obs_dim}")

    if expected_stack != env.cfg.frame_stack:
        raise RuntimeError(f"frame stack mismatch: checkpoint={expected_stack}, env={env.cfg.frame_stack}")

    if expected_lookahead != env.cfg.lookahead_steps:
        raise RuntimeError(f"lookahead mismatch: checkpoint={expected_lookahead}, env={env.cfg.lookahead_steps}")

    policy = QuadrotorTask2Actor(
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=env.device,
        init_log_std=float(train_args.get("init_log_std", -0.65)),
        min_log_std=float(train_args.get("min_log_std", -3.0)),
        max_log_std=float(train_args.get("max_log_std", 0.5)),
    ).to(env.device)

    policy.load_state_dict(ckpt["policy"], strict=True)
    policy.eval()

    actor_obs_norm = ckpt.get("actor_obs_norm", None)
    trained_env_steps = int(ckpt.get("env_steps", 0))

    return policy, actor_obs_norm, trained_env_steps, metadata


# ======================================================================
# Reporting helpers
# ======================================================================

def to_float(x: Any):
    try:
        if torch.is_tensor(x):
            return float(x.detach().float().mean().cpu().item())
        if isinstance(x, np.ndarray):
            return float(np.mean(x))
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
    except Exception:
        return None
    return None


def flat_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}

    for key, value in (data or {}).items():
        name = f"{prefix}/{key}" if prefix else str(key)

        if isinstance(value, dict):
            out.update(flat_dict(value, name))
        else:
            val = to_float(value)
            if val is not None and np.isfinite(val):
                out[name] = val

    return out


def summarize(records: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not records:
        return {}

    keys = sorted({key for row in records for key in row.keys()})
    out: Dict[str, Dict[str, float]] = {}

    for key in keys:
        vals = np.asarray([row[key] for row in records if key in row], dtype=np.float64)
        if vals.size == 0:
            continue

        out[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "p25": float(np.percentile(vals, 25)),
            "p50": float(np.percentile(vals, 50)),
            "p75": float(np.percentile(vals, 75)),
            "max": float(np.max(vals)),
        }

    return out


def print_summary_table(summary: Dict[str, Dict[str, float]]) -> None:
    print("\n" + "=" * 188)
    print("Quadrotor / Crazyflie Task2 TRUE skrl PPO Model Test Summary")
    print("=" * 188)
    print(
        f"{'metric':<86} | {'mean':>12} | {'std':>12} | {'min':>12} | "
        f"{'p25':>12} | {'p50':>12} | {'p75':>12} | {'max':>12}"
    )
    print("-" * 188)

    for key in sorted(summary.keys()):
        row = summary[key]
        print(
            f"{key:<86} | "
            f"{row['mean']:>12.6f} | "
            f"{row['std']:>12.6f} | "
            f"{row['min']:>12.6f} | "
            f"{row['p25']:>12.6f} | "
            f"{row['p50']:>12.6f} | "
            f"{row['p75']:>12.6f} | "
            f"{row['max']:>12.6f}"
        )

    print("=" * 188 + "\n")


def save_eval_npz(path: str, records: List[Dict[str, float]]) -> None:
    if not path or not records:
        return

    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    keys = sorted({key for row in records for key in row.keys()})
    arrays = {}
    for key in keys:
        safe_key = key.replace("/", "__")
        arrays[safe_key] = np.asarray([row.get(key, np.nan) for row in records], dtype=np.float32)

    np.savez_compressed(str(p), **arrays)
    print(f"[OK] eval npz saved to: {p}")


def save_tracking_plot(path: str, records: List[Dict[str, float]]) -> None:
    if not path or not records:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable, skip plot: {type(exc).__name__}: {exc}")
        return

    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    steps = np.arange(len(records))
    dist = np.asarray([row.get("telemetry/Dist_Error", np.nan) for row in records], dtype=np.float32)
    completion = np.asarray([row.get("telemetry/Completion_Rate", np.nan) for row in records], dtype=np.float32)
    reward = np.asarray([row.get("test/reward_mean", np.nan) for row in records], dtype=np.float32)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(steps, dist, label="Tracking distance error")
    plt.xlabel("Logged evaluation step")
    plt.ylabel("Distance error (m)")
    plt.title("Task2 Tracking Error")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(steps, completion, label="Completion rate")
    plt.plot(steps, reward, label="Reward mean", alpha=0.65)
    plt.xlabel("Logged evaluation step")
    plt.ylabel("Value")
    plt.title("Task2 Completion / Reward")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(str(p), dpi=150)
    plt.close()

    print(f"[OK] tracking plot saved to: {p}")


# ======================================================================
# Main
# ======================================================================

def build_env() -> QuadrotorTask2Env:
    cfg = Task2Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(args_cli.test_device)
    cfg.seed = int(args_cli.seed)
    cfg.max_episode_length_s = float(args_cli.max_episode_length_s)
    cfg.print_debug_info = bool(args_cli.print_names)

    if bool(args_cli.hold_success):
        cfg.success_end_margin = max(int(cfg.trajectory_num_points) + 100, 10000)

    cfg.validate()
    return QuadrotorTask2Env(cfg)


def main() -> None:
    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    env = build_env()
    obs, info = env.reset(seed=int(args_cli.seed))

    ckpt_path = resolve_checkpoint(args_cli.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    policy, actor_obs_norm, trained_env_steps, metadata = load_policy_checkpoint(ckpt_path, env)

    print("\n" + "=" * 150)
    print("Quadrotor / Crazyflie Task2 TRUE skrl PPO model test started")
    print("=" * 150)
    print(f"checkpoint            : {ckpt_path}")
    print(f"trained_env_steps     : {trained_env_steps:,}")
    print(f"num_envs              : {env.num_envs}")
    print(f"steps                 : {args_cli.steps}")
    print(f"actor_obs_dim         : {env.observation_space.shape[0]}")
    print(f"critic_obs_dim        : {env.state_space.shape[0]}")
    print(f"single_obs_dim        : {env.cfg.single_actor_obs_dim}")
    print(f"frame_stack           : {env.cfg.frame_stack}")
    print(f"lookahead             : {env.cfg.lookahead_steps} x {env.cfg.lookahead_interval}")
    print(f"trajectory_num_points : {env.num_points}")
    print(f"action_dim            : {env.action_space.shape[0]}")
    print(f"asset_source          : {env.asset_source}")
    print(f"estimated_mass        : {env.estimated_mass.mean().item():.6f} kg")
    print(f"hover_thrust          : {env.hover_thrust.mean().item():.6f} N")
    print(f"slow_action_scale     : {args_cli.slow_action_scale}")
    print(f"hold_success          : {bool(args_cli.hold_success)}")
    print(f"device                : {env.device}")
    print(f"visualize             : {bool(args_cli.visualize)}")
    print("algorithm             : skrl PPO")
    print("checkpoint metadata   : uses_skrl=True")
    print("metadata              :", metadata)
    print("=" * 150 + "\n")

    records: List[Dict[str, float]] = []
    total_terminated = 0
    total_truncated = 0
    start_time = time.time()

    try:
        with tqdm(
            total=int(args_cli.steps),
            desc="Quadrotor Task2 skrl Model Test",
            dynamic_ncols=True,
            mininterval=0.5,
        ) as pbar:
            for step in range(int(args_cli.steps)):
                with torch.no_grad():
                    actor_obs = obs
                    actor_obs_n = normalize_with_saved_obs_norm(actor_obs, actor_obs_norm)
                    actions = policy.act_deterministic_direct(actor_obs_n)
                    actions = torch.clamp(actions * float(args_cli.slow_action_scale), -1.0, 1.0)

                if step < 3:
                    print(
                        f"[DEBUG][eval step {step}] action_mean={actions.mean().item():+.6f}, "
                        f"action_abs_max={actions.abs().max().item():.6f}",
                        flush=True,
                    )

                obs, rewards, terminated, truncated, info = env.step(actions)

                total_terminated += int(terminated.sum().item())
                total_truncated += int(truncated.sum().item())

                if bool(args_cli.visualize):
                    try:
                        time.sleep(float(env.cfg.policy_dt))
                    except Exception:
                        pass

                if step % max(int(args_cli.print_interval), 1) == 0 or step == int(args_cli.steps) - 1:
                    flat = flat_dict(info)
                    row = {
                        "test/reward_mean": float(rewards.detach().float().mean().cpu().item()),
                        "test/reward_min": float(rewards.detach().float().min().cpu().item()),
                        "test/reward_max": float(rewards.detach().float().max().cpu().item()),
                        "test/terminated_rate": float(terminated.float().mean().cpu().item()),
                        "test/truncated_rate": float(truncated.float().mean().cpu().item()),
                    }
                    row.update(flat)
                    records.append(row)

                    pbar.set_postfix(
                        {
                            "rew": f"{row['test/reward_mean']:+.3f}",
                            "dist": f"{flat.get('telemetry/Dist_Error', 0.0):.3f}",
                            "comp": f"{flat.get('telemetry/Completion_Rate', 0.0):.3f}",
                            "idx": f"{flat.get('telemetry/Target_Idx', 0.0):.1f}",
                            "z": f"{flat.get('telemetry/Z', 0.0):.3f}",
                            "rp": f"{flat.get('telemetry/RollPitchAbs', 0.0):.3f}",
                            "succ": f"{flat.get('events/Success_Rate', 0.0):.3f}",
                            "dev": f"{flat.get('events/Deviation_Rate', 0.0):.3f}",
                        }
                    )

                    if bool(args_cli.visualize):
                        sys.stdout.write(
                            f"\r🚁 "
                            f"Dist={flat.get('telemetry/Dist_Error', 0.0):.3f} | "
                            f"Comp={flat.get('telemetry/Completion_Rate', 0.0):.3f} | "
                            f"Idx={flat.get('telemetry/Target_Idx', 0.0):.1f} | "
                            f"Z={flat.get('telemetry/Z', 0.0):.3f} | "
                            f"RP={flat.get('telemetry/RollPitchAbs', 0.0):.3f} | "
                            f"VelAlign={flat.get('telemetry/Vel_Align', 0.0):+.3f} | "
                            f"Heading_Align={flat.get('telemetry/Heading_Align', 0.0):+.3f} | "
                            f"ActionAbs={flat.get('telemetry/Action_Abs', 0.0):.3f} | "
                            f"R={row['test/reward_mean']:+.3f} | "
                            f"Succ={flat.get('events/Success_Rate', 0.0):.3f} | "
                            f"Crash={flat.get('events/Crash_Rate', 0.0):.3f} | "
                            f"Dev={flat.get('events/Deviation_Rate', 0.0):.3f} | "
                            f"Timeout={flat.get('events/Timeout_Rate', 0.0):.3f}"
                        )
                        sys.stdout.flush()

                pbar.update(1)

                if bool(args_cli.visualize) and not simulation_app.is_running():
                    print("\n[INFO] Isaac Sim window closed.")
                    break

        elapsed = time.time() - start_time
        env_steps = int(args_cli.steps) * int(env.num_envs)
        fps = env_steps / max(elapsed, 1.0e-6)

        print("\n✅ Quadrotor Task2 TRUE skrl PPO model test rollout finished")
        print(f"  env steps        : {env_steps:,}")
        print(f"  fps env steps    : {fps:,.2f}")
        print(f"  total terminated : {total_terminated:,}")
        print(f"  total truncated  : {total_truncated:,}")

        print_summary_table(summarize(records))

        if args_cli.save_npz:
            save_eval_npz(args_cli.save_npz, records)

        if args_cli.save_plot:
            save_tracking_plot(args_cli.save_plot, records)

        print("Quadrotor Task2 model test checklist:")
        print("1. checkpoint metadata 必须标记 uses_skrl=True。")
        print("2. actor obs 必须为 100 维。")
        print("3. action 必须为 4 维。")
        print("4. smoke checkpoint 效果差是正常的，先看加载、rollout 和无 NaN/Inf。")
        print("5. 正式效果重点看 Dist_Error、Completion_Rate、Target_Idx、Vel_Align、Heading_Align、Crash_Rate、Deviation_Rate。")

    finally:
        try:
            env.close()
        except Exception:
            pass

        try:
            simulation_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
