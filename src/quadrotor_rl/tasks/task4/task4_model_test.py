from __future__ import annotations

import argparse
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
    description="Evaluate Quadrotor / Crazyflie Task4 TRUE skrl PPO vision gate-racing model"
)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num-envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test-device", type=str, default="cuda:0")
parser.add_argument("--print-interval", type=int, default=20)
parser.add_argument("--max-episode-length-s", type=float, default=12.0)
parser.add_argument("--slow-action-scale", type=float, default=1.0)
parser.add_argument("--visualize", action="store_true")
parser.add_argument("--print-names", action="store_true")
parser.add_argument("--enable-sensor-noise", action="store_true")
parser.add_argument("--save-plot", type=str, default="")
parser.add_argument("--save-npz", type=str, default="")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.device = str(args_cli.test_device)
args_cli.headless = not bool(args_cli.visualize)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from skrl.models.torch import GaussianMixin, Model

from quadrotor_rl.tasks.task4.task4_config import Task4Config
from quadrotor_rl.tasks.task4.task4_env import QuadrotorTask4Env


# ======================================================================
# Vision policy
# ======================================================================

class Task4VisionEncoder(nn.Module):
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
            p / "quadrotor_task4_model.pt",
            p / "final_checkpoint" / "quadrotor_task4_model.pt",
            p / "checkpoint_5000" / "quadrotor_task4_model.pt",
            p / "checkpoint_3000" / "quadrotor_task4_model.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        pt_files = sorted(p.glob("*.pt"))
        for pt in pt_files:
            if pt.name.endswith("_preprocessor.pt"):
                continue
            if pt.name in {"train_metadata.pt", "quadrotor_task4_skrl_model.pt"}:
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

    if mean is None or var is None:
        for _, value in norm_state.items():
            if isinstance(value, dict):
                m, v = _find_norm_tensors(value)
                if m is not None and v is not None:
                    return m, v

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


def infer_network_dims_from_checkpoint(ckpt: Dict[str, Any], train_args: Dict[str, Any]) -> Dict[str, int]:
    policy_state = ckpt.get("policy", {})
    dims = {
        "cnn_output_dim": int(train_args.get("cnn_output_dim", 256)),
        "compact_output_dim": int(train_args.get("compact_output_dim", 128)),
        "hidden_dim": int(train_args.get("hidden_dim", 256)),
    }

    if isinstance(policy_state, dict):
        w = policy_state.get("encoder.cnn_linear.0.weight", None)
        if torch.is_tensor(w):
            dims["cnn_output_dim"] = int(w.shape[0])

        w = policy_state.get("encoder.compact_mlp.2.weight", None)
        if torch.is_tensor(w):
            dims["compact_output_dim"] = int(w.shape[0])

        w = policy_state.get("net.0.weight", None)
        if torch.is_tensor(w):
            dims["hidden_dim"] = int(w.shape[0])

    return dims


def load_policy_checkpoint(ckpt_path: Path, env: QuadrotorTask4Env):
    ckpt = torch_load_checkpoint(ckpt_path, env.device)

    if not isinstance(ckpt, dict) or "policy" not in ckpt:
        raise RuntimeError(
            f"当前测试脚本需要 task4_train.py 保存的 eval checkpoint: quadrotor_task4_model.pt\n"
            f"当前文件不是 eval checkpoint: {ckpt_path}"
        )

    metadata = ckpt.get("metadata", {})
    train_args = ckpt.get("args", {})

    if not bool(metadata.get("uses_skrl", False)):
        raise RuntimeError("checkpoint metadata 缺少 uses_skrl=True，请使用 TRUE skrl 版本重新训练。")

    task_name = str(metadata.get("task", ""))
    if "task4" not in task_name and "gate" not in task_name:
        raise RuntimeError(f"checkpoint task metadata 不是 Task4 vision gate racing: {task_name}")

    expected_actor_dim = int(metadata.get("actor_obs_dim", env.observation_space.shape[0]))
    expected_critic_dim = int(metadata.get("critic_obs_dim", env.state_space.shape[0]))
    expected_action_dim = int(metadata.get("action_dim", env.action_space.shape[0]))
    expected_depth_dim = int(metadata.get("depth_dim", env.cfg.depth_dim))
    expected_compact_dim = int(metadata.get("compact_state_dim", env.cfg.compact_state_dim))
    expected_num_gates = int(metadata.get("num_gates", env.cfg.num_gates))

    if expected_actor_dim != env.observation_space.shape[0]:
        raise RuntimeError(f"actor obs dim mismatch: checkpoint={expected_actor_dim}, env={env.observation_space.shape[0]}")

    if expected_critic_dim != env.state_space.shape[0]:
        raise RuntimeError(f"critic obs dim mismatch: checkpoint={expected_critic_dim}, env={env.state_space.shape[0]}")

    if expected_action_dim != env.action_space.shape[0]:
        raise RuntimeError(f"action dim mismatch: checkpoint={expected_action_dim}, env={env.action_space.shape[0]}")

    if expected_depth_dim != env.cfg.depth_dim:
        raise RuntimeError(f"depth dim mismatch: checkpoint={expected_depth_dim}, env={env.cfg.depth_dim}")

    if expected_compact_dim != env.cfg.compact_state_dim:
        raise RuntimeError(f"compact state dim mismatch: checkpoint={expected_compact_dim}, env={env.cfg.compact_state_dim}")

    if expected_num_gates != env.cfg.num_gates:
        raise RuntimeError(f"num_gates mismatch: checkpoint={expected_num_gates}, env={env.cfg.num_gates}")

    dims = infer_network_dims_from_checkpoint(ckpt, train_args)

    policy = QuadrotorTask4Actor(
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=env.device,
        init_log_std=float(train_args.get("init_log_std", -0.55)),
        min_log_std=float(train_args.get("min_log_std", -3.0)),
        max_log_std=float(train_args.get("max_log_std", 0.5)),
        cnn_output_dim=int(dims["cnn_output_dim"]),
        compact_output_dim=int(dims["compact_output_dim"]),
        hidden_dim=int(dims["hidden_dim"]),
    ).to(env.device)

    policy.load_state_dict(ckpt["policy"], strict=True)
    policy.eval()

    actor_obs_norm = ckpt.get("actor_obs_norm", None)
    trained_env_steps = int(ckpt.get("env_steps", 0))

    return policy, actor_obs_norm, trained_env_steps, metadata, dims


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
    print("Quadrotor / Crazyflie Task4 TRUE skrl PPO Model Test Summary")
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
    passed = np.asarray([row.get("telemetry/Passed_Gates", np.nan) for row in records], dtype=np.float32)
    center = np.asarray([row.get("telemetry/Centerline_Dist", np.nan) for row in records], dtype=np.float32)
    depth_min = np.asarray([row.get("telemetry/Depth_Min", np.nan) for row in records], dtype=np.float32)
    reward = np.asarray([row.get("test/reward_mean", np.nan) for row in records], dtype=np.float32)

    plt.figure(figsize=(12, 5))
    plt.plot(steps, passed, label="Passed gates")
    plt.plot(steps, center, label="Centerline distance")
    plt.plot(steps, depth_min, label="Depth min")
    plt.plot(steps, reward, label="Reward mean", alpha=0.7)
    plt.xlabel("Logged evaluation step")
    plt.ylabel("Value")
    plt.title("Task4 Vision Gate-Racing Evaluation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(p), dpi=150)
    plt.close()

    print(f"[OK] gate-racing plot saved to: {p}")


# ======================================================================
# Main
# ======================================================================

def build_env() -> QuadrotorTask4Env:
    cfg = Task4Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(args_cli.test_device)
    cfg.seed = int(args_cli.seed)
    cfg.max_episode_length_s = float(args_cli.max_episode_length_s)
    cfg.enable_sensor_noise = bool(args_cli.enable_sensor_noise)
    cfg.print_debug_info = bool(args_cli.print_names)

    cfg.validate()
    return QuadrotorTask4Env(cfg)


def main() -> None:
    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    env = build_env()
    obs, info = env.reset(seed=int(args_cli.seed))

    ckpt_path = resolve_checkpoint(args_cli.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    policy, actor_obs_norm, trained_env_steps, metadata, dims = load_policy_checkpoint(ckpt_path, env)

    print("\n" + "=" * 150)
    print("Quadrotor / Crazyflie Task4 TRUE skrl PPO model test started")
    print("=" * 150)
    print(f"checkpoint              : {ckpt_path}")
    print(f"trained_env_steps       : {trained_env_steps:,}")
    print(f"num_envs                : {env.num_envs}")
    print(f"steps                   : {args_cli.steps}")
    print(f"actor_obs_dim           : {env.observation_space.shape[0]}")
    print(f"critic_obs_dim          : {env.state_space.shape[0]}")
    print(f"depth_dim               : {env.cfg.depth_dim}")
    print(f"compact_state_dim       : {env.cfg.compact_state_dim}")
    print(f"camera                  : {env.cfg.cam_res_w} x {env.cfg.cam_res_h}, fov={env.cfg.cam_fov_deg}")
    print(f"num_gates               : {env.cfg.num_gates}")
    print(f"action_dim              : {env.action_space.shape[0]}")
    print(f"asset_source            : {env.asset_source}")
    print(f"estimated_mass          : {env.estimated_mass.mean().item():.6f} kg")
    print(f"hover_thrust            : {env.hover_thrust.mean().item():.6f} N")
    print(f"slow_action_scale       : {args_cli.slow_action_scale}")
    print(f"device                  : {env.device}")
    print(f"visualize               : {bool(args_cli.visualize)}")
    print(f"enable_sensor_noise     : {bool(args_cli.enable_sensor_noise)}")
    print("algorithm               : skrl PPO")
    print("checkpoint metadata     : uses_skrl=True")
    print(f"network_dims            : {dims}")
    print("metadata                :", metadata)
    print("=" * 150 + "\n")

    records: List[Dict[str, float]] = []
    total_terminated = 0
    total_truncated = 0
    start_time = time.time()

    try:
        with tqdm(
            total=int(args_cli.steps),
            desc="Quadrotor Task4 skrl Model Test",
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
                            "gate": f"{flat.get('telemetry/Target_Gate_Idx', 0.0):.2f}",
                            "passed": f"{flat.get('telemetry/Passed_Gates', 0.0):.2f}",
                            "center": f"{flat.get('telemetry/Centerline_Dist', 0.0):.2f}",
                            "depth": f"{flat.get('telemetry/Depth_Min', 0.0):.2f}",
                            "align": f"{flat.get('telemetry/Pose_Align', 0.0):+.2f}",
                            "hit": f"{flat.get('events/Gate_Collision_Rate', 0.0):.3f}",
                            "miss": f"{flat.get('events/Missed_Gate_Rate', 0.0):.3f}",
                        }
                    )

                    if bool(args_cli.visualize):
                        sys.stdout.write(
                            f"\r🚁 "
                            f"Gate={flat.get('telemetry/Target_Gate_Idx', 0.0):.3f} | "
                            f"Passed_Gates={flat.get('telemetry/Passed_Gates', 0.0):.3f} | "
                            f"Centerline_Dist={flat.get('telemetry/Centerline_Dist', 0.0):.3f} | "
                            f"Progress={flat.get('telemetry/Progress', 0.0):+.3f} | "
                            f"Depth_Min={flat.get('telemetry/Depth_Min', 0.0):.3f} | "
                            f"Depth_Mean={flat.get('telemetry/Depth_Mean', 0.0):.3f} | "
                            f"Pose_Align={flat.get('telemetry/Pose_Align', 0.0):+.3f} | "
                            f"V_Tangent={flat.get('telemetry/V_Tangent', 0.0):+.3f} | "
                            f"ActionAbs={flat.get('telemetry/Action_Abs', 0.0):.3f} | "
                            f"R={row['test/reward_mean']:+.3f} | "
                            f"Succ={flat.get('events/Success_Rate', 0.0):.3f} | "
                            f"Crash={flat.get('events/Crash_Rate', 0.0):.3f} | "
                            f"Gate_Collision_Rate={flat.get('events/Gate_Collision_Rate', 0.0):.3f} | "
                            f"Missed_Gate_Rate={flat.get('events/Missed_Gate_Rate', 0.0):.3f} | "
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

        print("\n✅ Quadrotor Task4 TRUE skrl PPO model test rollout finished")
        print(f"  env steps        : {env_steps:,}")
        print(f"  fps env steps    : {fps:,.2f}")
        print(f"  total terminated : {total_terminated:,}")
        print(f"  total truncated  : {total_truncated:,}")

        print_summary_table(summarize(records))

        if args_cli.save_npz:
            save_eval_npz(args_cli.save_npz, records)

        if args_cli.save_plot:
            save_tracking_plot(args_cli.save_plot, records)

        print("Quadrotor Task4 model test checklist:")
        print("1. checkpoint metadata 必须标记 uses_skrl=True。")
        print("2. actor obs 必须为 4128 维。")
        print("3. depth 必须为 1 x 64 x 64，compact state 必须为 32 维。")
        print("4. action 必须为 4 维。")
        print("5. smoke checkpoint 效果差是正常的，先看加载、rollout 和无 NaN/Inf。")
        print("6. 正式效果重点看 Passed_Gates、Centerline_Dist、Depth_Min、Pose_Align、Gate_Collision_Rate、Missed_Gate_Rate、Success_Rate。")

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
