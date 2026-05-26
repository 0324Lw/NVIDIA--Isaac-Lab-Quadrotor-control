from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Quadrotor / Crazyflie Task4 vision gate racing environment test")
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test-device", type=str, default="cuda:0")
parser.add_argument("--quick", action="store_true")
parser.add_argument("--print-names", action="store_true")
parser.add_argument("--collect-interval", type=int, default=20)

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from quadrotor_rl.tasks.task4.task4_config import Task4Config
from quadrotor_rl.tasks.task4.task4_env import QuadrotorTask4Env


def heading(title: str) -> None:
    print("\n" + "=" * 140)
    print(title)
    print("=" * 140, flush=True)


def print_ok(msg: str) -> None:
    print(f" ✅ {msg}", flush=True)


def assert_finite_tensor(name: str, x: torch.Tensor) -> None:
    assert torch.is_tensor(x), f"{name} must be torch.Tensor"
    assert torch.isfinite(x).all().item(), f"{name} has NaN/Inf"


def check_shape(name: str, x: torch.Tensor, expected) -> None:
    assert tuple(x.shape) == tuple(expected), f"{name} shape mismatch: {tuple(x.shape)} != {tuple(expected)}"


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


def flat_info(info: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    out = {}
    for k, v in (info or {}).items():
        name = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flat_info(v, name))
        else:
            val = to_float(v)
            if val is not None and np.isfinite(val):
                out[name] = val
    return out


def summarize(records: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not records:
        return {}
    keys = sorted({k for row in records for k in row.keys()})
    out = {}
    for key in keys:
        vals = np.asarray([row[key] for row in records if key in row], dtype=np.float64)
        if vals.size == 0:
            continue
        out[key] = {
            "mean": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "p50": float(np.percentile(vals, 50)),
            "max": float(np.max(vals)),
        }
    return out


def print_summary(summary: Dict[str, Dict[str, float]]) -> None:
    print("\n" + "=" * 130)
    print("Quadrotor Task4 Env Test Summary")
    print("=" * 130)
    print(f"{'metric':<72} | {'mean':>12} | {'min':>12} | {'p50':>12} | {'max':>12}")
    print("-" * 130)
    for key in sorted(summary.keys()):
        r = summary[key]
        print(f"{key:<72} | {r['mean']:>12.6f} | {r['min']:>12.6f} | {r['p50']:>12.6f} | {r['max']:>12.6f}")
    print("=" * 130 + "\n")


def check_obs(env: QuadrotorTask4Env, obs: torch.Tensor) -> None:
    check_shape("obs", obs, (env.num_envs, env.num_observations))
    assert_finite_tensor("obs", obs)
    assert obs.abs().max().item() <= float(env.cfg.obs_clip) + 1e-5


def check_state(env: QuadrotorTask4Env, state: torch.Tensor) -> None:
    check_shape("state", state, (env.num_envs, env.num_privileged_obs))
    assert_finite_tensor("state", state)


def depth_slice(env: QuadrotorTask4Env, obs: torch.Tensor) -> torch.Tensor:
    return obs[:, : int(env.cfg.depth_dim)].view(
        env.num_envs,
        int(env.cfg.depth_channels),
        int(env.cfg.cam_res_h),
        int(env.cfg.cam_res_w),
    )


def compact_slice(env: QuadrotorTask4Env, obs: torch.Tensor) -> torch.Tensor:
    return obs[:, int(env.cfg.depth_dim):]


def set_deterministic_gate_track(env: QuadrotorTask4Env, first_x: float = -10.0, spacing: float = 2.0) -> None:
    """Create a straight gate track. Gate normal points along +X."""

    world = env.world
    cfg = env.cfg
    n = env.num_envs
    g = int(cfg.num_gates)

    normal = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=env.device)
    rot = world._rotation_from_local_z_to_vector(normal)
    quat = world._matrix_to_quat(rot)

    for i in range(g):
        world.gate_pos[:, i, 0] = float(first_x + spacing * i)
        world.gate_pos[:, i, 1] = 0.0
        world.gate_pos[:, i, 2] = float(cfg.start_pos[2])
        world.gate_rot[:, i, :, :] = rot.view(1, 3, 3).repeat(n, 1, 1)
        world.gate_quat[:, i, :] = quat.view(1, 4).repeat(n, 1)
        world.gate_tangent[:, i, :] = normal.view(1, 3).repeat(n, 1)
        world.gate_valid[:, i] = True

    world.target_gate_idx.zero_()
    for env_id in range(n):
        world.centerline[env_id] = world._build_centerline_from_gates(env_id)


def build_env() -> QuadrotorTask4Env:
    cfg = Task4Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(args_cli.test_device)
    cfg.seed = int(args_cli.seed)
    cfg.print_debug_info = bool(args_cli.print_names)
    cfg.enable_sensor_noise = False

    if bool(args_cli.quick):
        cfg.num_envs = min(cfg.num_envs, 2)
        args_cli.steps = min(int(args_cli.steps), 60)

    cfg.validate()
    return QuadrotorTask4Env(cfg)


def test_config() -> None:
    heading("[测试 1] Task4Config 基础配置检测")

    cfg = Task4Config()
    cfg.validate()

    assert cfg.action_dim == 4
    assert cfg.depth_dim == 4096
    assert cfg.compact_state_dim == 32
    assert cfg.single_actor_obs_dim == 4128
    assert cfg.actor_obs_dim == 4128
    assert cfg.critic_obs_dim == 4128
    assert cfg.num_gates == 5
    assert cfg.cam_res_w == 64
    assert cfg.cam_res_h == 64
    assert cfg.start_pos == (-12.0, 0.0, 1.5)

    print_ok(f"action_dim = {cfg.action_dim}")
    print_ok(f"depth_dim = {cfg.depth_dim}")
    print_ok(f"compact_state_dim = {cfg.compact_state_dim}")
    print_ok(f"actor_obs_dim = {cfg.actor_obs_dim}")
    print_ok(f"critic_obs_dim = {cfg.critic_obs_dim}")
    print_ok(f"num_gates = {cfg.num_gates}")
    print_ok(f"camera = {cfg.cam_res_w} x {cfg.cam_res_h}, fov = {cfg.cam_fov_deg}")
    print_ok(f"hover_thrust = {cfg.hover_thrust:.6f} N")


def test_init_reset(env: QuadrotorTask4Env) -> None:
    heading("[测试 2] 环境初始化 / reset / obs / start 对齐检测")

    obs, info = env.reset()

    check_obs(env, obs)
    assert "state" in info
    check_state(env, info["state"])

    assert env.observation_space.shape == (4128,)
    assert env.state_space.shape == (4128,)
    assert env.action_space.shape == (4,)

    root_local = env._root_pos_w() - env.env_origins
    err = torch.norm(root_local - env.world.start_pos, dim=-1)

    assert err.max().item() < 5e-4, f"reset root pos not aligned: {err.max().item():.8f}"
    assert env.world.target_gate_idx.float().mean().item() == 0.0

    print_ok(f"asset_source = {env.asset_source}")
    print_ok(f"num_bodies = {getattr(env.drone, 'num_bodies', '<unknown>')}")
    print_ok(f"num_joints = {getattr(env.drone, 'num_joints', '<unknown>')}")
    print_ok(f"obs shape = {tuple(obs.shape)}")
    print_ok(f"state shape = {tuple(info['state'].shape)}")
    print_ok(f"reset position max error = {err.max().item():.8f}")

    if args_cli.print_names:
        print("body_names:")
        for i, name in enumerate(list(getattr(env.drone, "body_names", []))):
            print(f"  body[{i:02d}] = {name}")
        print("joint_names:")
        for i, name in enumerate(list(getattr(env.drone, "joint_names", []))):
            print(f"  joint[{i:02d}] = {name}")


def test_obs_slices(env: QuadrotorTask4Env) -> None:
    heading("[测试 3] obs 切片范围检测：depth / compact")

    obs, info = env.reset()
    check_obs(env, obs)

    depth = depth_slice(env, obs)
    compact = compact_slice(env, obs)

    check_shape("depth", depth, (env.num_envs, 1, env.cfg.cam_res_h, env.cfg.cam_res_w))
    check_shape("compact", compact, (env.num_envs, env.cfg.compact_state_dim))
    assert_finite_tensor("depth", depth)
    assert_finite_tensor("compact", compact)

    assert depth.min().item() >= -1.0e-6
    assert depth.max().item() <= 1.0 + 1.0e-6

    # Compact layout:
    # target_features 14 + lin_vel 3 + ang_vel 3 + projected_gravity 3
    # filtered_action 4 + rpy 3 + gate_progress 1 + speed 1 = 32
    target_features = compact[:, :14]
    lin_vel = compact[:, 14:17]
    ang_vel = compact[:, 17:20]
    gravity = compact[:, 20:23]
    action = compact[:, 23:27]
    rpy = compact[:, 27:30]
    gate_progress = compact[:, 30:31]
    speed = compact[:, 31:32]

    for name, tensor, shape in [
        ("target_features", target_features, (env.num_envs, 14)),
        ("lin_vel", lin_vel, (env.num_envs, 3)),
        ("ang_vel", ang_vel, (env.num_envs, 3)),
        ("gravity", gravity, (env.num_envs, 3)),
        ("action", action, (env.num_envs, 4)),
        ("rpy", rpy, (env.num_envs, 3)),
        ("gate_progress", gate_progress, (env.num_envs, 1)),
        ("speed", speed, (env.num_envs, 1)),
    ]:
        check_shape(name, tensor, shape)
        assert_finite_tensor(name, tensor)

    assert target_features[:, -3:].min().item() >= -1.0e-6
    assert target_features[:, -3:].max().item() <= 1.0 + 1.0e-6
    assert gate_progress.min().item() >= -1.0e-6
    assert gate_progress.max().item() <= 1.0 + 1.0e-6
    assert speed.min().item() >= -1.0e-6

    print_ok(f"depth shape = {tuple(depth.shape)}")
    print_ok(f"compact shape = {tuple(compact.shape)}")
    print_ok(f"depth min/mean/max = {depth.min().item():.3f} / {depth.mean().item():.3f} / {depth.max().item():.3f}")
    print_ok(f"compact mean/max_abs = {compact.mean().item():+.3f} / {compact.abs().max().item():.3f}")


def test_step_structure(env: QuadrotorTask4Env) -> None:
    heading("[测试 4] step 返回结构 / info 字段检测")

    env.reset()
    action = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.device)
    obs, reward, terminated, truncated, info = env.step(action)

    check_obs(env, obs)
    check_state(env, info["state"])
    check_shape("reward", reward, (env.num_envs,))
    check_shape("terminated", terminated, (env.num_envs,))
    check_shape("truncated", truncated, (env.num_envs,))
    assert_finite_tensor("reward", reward)

    for group in ["reward_components", "events", "telemetry", "debug", "task4_stats"]:
        assert group in info, f"info missing group: {group}"

    for key in ["R_Track", "R_Align", "R_Smooth", "R_Depth", "Total"]:
        assert key in info["reward_components"], f"reward_components missing {key}"

    for key in ["Success_Rate", "Crash_Rate", "Gate_Collision_Rate", "Missed_Gate_Rate", "Deviation_Rate", "Timeout_Rate", "Gate_Pass_Rate"]:
        assert key in info["events"], f"events missing {key}"

    for key in ["Target_Gate_Idx", "Passed_Gates", "Centerline_Dist", "Depth_Min", "Pose_Align", "V_Tangent", "Pos_Z"]:
        assert key in info["telemetry"], f"telemetry missing {key}"

    print_ok(f"reward mean = {reward.mean().item():+.6f}")
    print_ok(f"terminated count = {terminated.sum().item()}")
    print_ok(f"truncated count = {truncated.sum().item()}")


def test_wrench_mapping(env: QuadrotorTask4Env) -> None:
    heading("[测试 5] 四旋翼动作到 root wrench 映射白盒检测")

    env.reset()

    zero = torch.zeros((env.num_envs, 4), dtype=torch.float32, device=env.device)
    force_w, torque_w, torque_b, motor_mult = env._actions_to_wrench(zero)

    assert_finite_tensor("force_w", force_w)
    assert_finite_tensor("torque_w", torque_w)
    assert_finite_tensor("torque_b", torque_b)
    assert_finite_tensor("motor_mult", motor_mult)

    assert force_w[:, 2].mean().item() > 0.0
    assert abs(torque_b[:, 0].mean().item()) < 1.0e-3
    assert abs(torque_b[:, 1].mean().item()) < 1.0e-3

    yaw_cmd = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float32, device=env.device).view(1, 4).repeat(env.num_envs, 1)
    _, _, torque_b_yaw, _ = env._actions_to_wrench(yaw_cmd * env.cfg.action_scale)

    assert torque_b_yaw[:, 2].abs().mean().item() > 1.0e-5

    print_ok(f"hover force z mean = {force_w[:, 2].mean().item():.6f}")
    print_ok(f"zero torque_b mean = {torque_b.mean(dim=0).detach().cpu().numpy()}")
    print_ok(f"yaw torque_b z mean = {torque_b_yaw[:, 2].mean().item():+.6f}")


def run_action_response(env: QuadrotorTask4Env, action_value: torch.Tensor, steps: int = 30):
    env.reset()
    action = action_value.to(env.device).view(1, 4).repeat(env.num_envs, 1)

    z0 = (env._root_pos_w() - env.env_origins)[:, 2].clone()
    yaw0 = env._quat_to_euler_wxyz(env._root_quat_wxyz())[2].clone()

    obs = None
    reward = None
    terminated = None
    truncated = None
    info = {}

    for _ in range(int(steps)):
        obs, reward, terminated, truncated, info = env.step(action)

    z1 = (env._root_pos_w() - env.env_origins)[:, 2].clone()
    yaw1 = env._quat_to_euler_wxyz(env._root_quat_wxyz())[2].clone()
    dz = z1 - z0
    dyaw = torch.atan2(torch.sin(yaw1 - yaw0), torch.cos(yaw1 - yaw0))

    return dz, dyaw, obs, reward, terminated, truncated, info


def test_control_response(env: QuadrotorTask4Env) -> None:
    heading("[测试 6] 动作控制响应检测：上升 / 下降 / 偏航")

    dz_up, _, *_ = run_action_response(env, torch.ones(4, dtype=torch.float32), steps=30)
    dz_down, _, *_ = run_action_response(env, -torch.ones(4, dtype=torch.float32), steps=30)

    yaw_action = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float32)
    _, dyaw, *_ = run_action_response(env, yaw_action, steps=30)

    print_ok(f"up action delta z mean = {dz_up.mean().item():+.6f}")
    print_ok(f"down action delta z mean = {dz_down.mean().item():+.6f}")
    print_ok(f"yaw action delta yaw mean = {dyaw.mean().item():+.6f}")

    assert torch.isfinite(dz_up).all().item()
    assert torch.isfinite(dz_down).all().item()
    assert torch.isfinite(dyaw).all().item()
    assert dz_up.mean().item() > dz_down.mean().item() - 0.05, "up action should not be weaker than down action"


def test_gate_progress_and_success(env: QuadrotorTask4Env) -> None:
    heading("[测试 7] 手动触发穿门进度 / 成功事件检测")

    env.reset()
    set_deterministic_gate_track(env, first_x=-10.0, spacing=2.0)

    prev = env.world.start_pos.clone()
    prev[:, 0] = -11.0
    prev[:, 1] = 0.0
    prev[:, 2] = float(env.cfg.start_pos[2])

    curr = prev.clone()
    curr[:, 0] = -9.5

    env.prev_pos_local = prev.clone()
    env.set_root_pose_for_test(curr)

    reward, terminated, truncated, info = env.check_events_for_test()

    assert info["events"]["Gate_Pass_Rate"] > 0.99, f"Gate_Pass_Rate={info['events']['Gate_Pass_Rate']}"
    assert info["telemetry"]["Passed_Gates"] >= 1.0 - 1e-5

    print_ok(f"gate pass triggered, Gate_Pass_Rate={info['events']['Gate_Pass_Rate']:.6f}")
    print_ok(f"passed gates after first pass = {info['telemetry']['Passed_Gates']:.3f}")

    env.reset()
    set_deterministic_gate_track(env, first_x=-10.0, spacing=2.0)
    env.world.target_gate_idx[:] = int(env.cfg.num_gates)
    env.prev_pos_local = env.world.start_pos.clone()
    env.set_root_pose_for_test(env.world.start_pos.clone())

    reward, terminated, truncated, info = env.check_events_for_test()

    assert truncated.float().mean().item() > 0.99
    assert info["events"]["Success_Rate"] > 0.99
    assert info["task4_stats"]["reason"] == "SUCCESS_ALL_GATES"

    print_ok(f"success triggered, Success_Rate={info['events']['Success_Rate']:.6f}")


def test_terminal_events(env: QuadrotorTask4Env) -> None:
    heading("[测试 8] 手动触发 gate collision / missed gate / floor / flip / deviation / timeout 检测")

    env.reset()
    set_deterministic_gate_track(env, first_x=-10.0, spacing=2.0)

    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    gp = env.world.gate_pos[env_ids, torch.zeros(env.num_envs, dtype=torch.long, device=env.device)]
    gr = env.world.gate_rot[env_ids, torch.zeros(env.num_envs, dtype=torch.long, device=env.device)]

    local_frame = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)
    local_frame[:, 0] = float(env.cfg.gate_inner_half + env.cfg.gate_thickness * 0.5)
    frame_pos = gp + torch.bmm(gr, local_frame.unsqueeze(-1)).squeeze(-1)

    env.prev_pos_local = frame_pos.clone()
    env.set_root_pose_for_test(frame_pos)
    reward, terminated, truncated, info = env.check_events_for_test()

    assert terminated.float().mean().item() > 0.99
    assert info["events"]["Gate_Collision_Rate"] > 0.99
    print_ok(f"gate collision triggered, Gate_Collision_Rate={info['events']['Gate_Collision_Rate']:.6f}")

    env.reset()
    set_deterministic_gate_track(env, first_x=-10.0, spacing=2.0)

    prev = env.world.start_pos.clone()
    prev[:, 0] = -11.0
    prev[:, 1] = 1.50
    prev[:, 2] = float(env.cfg.start_pos[2])

    curr = prev.clone()
    curr[:, 0] = -9.5

    env.prev_pos_local = prev.clone()
    env.set_root_pose_for_test(curr)
    reward, terminated, truncated, info = env.check_events_for_test()

    assert terminated.float().mean().item() > 0.99
    assert info["events"]["Missed_Gate_Rate"] > 0.99
    print_ok(f"missed gate triggered, Missed_Gate_Rate={info['events']['Missed_Gate_Rate']:.6f}")

    env.reset()
    low = env.world.start_pos.clone()
    low[:, 2] = float(env.cfg.crash_z_min - 0.05)
    env.prev_pos_local = low.clone()
    env.set_root_pose_for_test(low)
    reward, terminated, truncated, info = env.check_events_for_test()

    assert terminated.float().mean().item() > 0.99
    assert info["events"]["Floor_Crash_Rate"] > 0.99
    print_ok(f"floor crash triggered, Floor_Crash_Rate={info['events']['Floor_Crash_Rate']:.6f}")

    env.reset()
    start = env.world.start_pos.clone()
    env.prev_pos_local = start.clone()
    env.set_root_pose_for_test(start, roll=float(env.cfg.crash_roll_pitch_max + 0.20))
    reward, terminated, truncated, info = env.check_events_for_test()

    assert terminated.float().mean().item() > 0.99
    assert info["events"]["Flip_Crash_Rate"] > 0.99
    print_ok(f"flip crash triggered, Flip_Crash_Rate={info['events']['Flip_Crash_Rate']:.6f}")

    env.reset()
    far = env.world.start_pos.clone()
    far[:, 1] = float(env.cfg.arena_half_width + 1.0)
    far[:, 2] = float(env.cfg.start_pos[2])
    env.prev_pos_local = far.clone()
    env.set_root_pose_for_test(far)
    reward, terminated, truncated, info = env.check_events_for_test()

    assert terminated.float().mean().item() > 0.99
    assert info["events"]["Deviation_Rate"] > 0.99
    print_ok(f"out-of-bounds deviation triggered, Deviation_Rate={info['events']['Deviation_Rate']:.6f}")

    env.reset()
    env.episode_steps[:] = int(env.cfg.max_episode_length)
    env.prev_pos_local = env.world.start_pos.clone()
    reward, terminated, truncated, info = env.check_events_for_test()

    assert truncated.float().mean().item() > 0.99
    assert info["events"]["Timeout_Rate"] > 0.99
    print_ok(f"timeout triggered, Timeout_Rate={info['events']['Timeout_Rate']:.6f}")


def random_rollout(env: QuadrotorTask4Env) -> None:
    heading(f"[测试 9] 随机策略 rollout {args_cli.steps} 步，无 NaN / Inf 检测")

    obs, info = env.reset()
    check_obs(env, obs)
    check_state(env, info["state"])

    records = []
    t0 = time.time()

    for step in range(int(args_cli.steps)):
        action = torch.rand((env.num_envs, env.num_actions), dtype=torch.float32, device=env.device) * 2.0 - 1.0
        obs, reward, terminated, truncated, info = env.step(action)

        check_obs(env, obs)
        check_state(env, info["state"])
        assert_finite_tensor("reward rollout", reward)
        assert_finite_tensor("root_pos", env._root_pos_w())
        assert_finite_tensor("root_quat", env._root_quat_wxyz())
        assert_finite_tensor("last_force_w", env.last_force_w)
        assert_finite_tensor("last_torque_w", env.last_torque_w)
        assert_finite_tensor("current_depth", env.current_depth)

        depth = depth_slice(env, obs)
        assert depth.min().item() >= -1.0e-6
        assert depth.max().item() <= 1.0 + 1.0e-6

        if (step + 1) % max(int(args_cli.collect_interval), 1) == 0 or (step + 1) == int(args_cli.steps):
            flat = flat_info(info)
            flat["Reward_Mean_Step"] = reward.mean().item()
            records.append(flat)

            print(
                f" -> Step {step + 1:05d} | "
                f"Reward={reward.mean().item():+.4f} | "
                f"Gate={flat.get('telemetry/Target_Gate_Idx', 0.0):.2f} | "
                f"Centerline={flat.get('telemetry/Centerline_Dist', 0.0):.3f} | "
                f"Prog={flat.get('telemetry/Progress', 0.0):+.3f} | "
                f"DepthMin={flat.get('telemetry/Depth_Min', 0.0):.3f} | "
                f"Align={flat.get('telemetry/Pose_Align', 0.0):+.3f} | "
                f"Succ={flat.get('events/Success_Rate', 0.0):.3f} | "
                f"Crash={flat.get('events/Crash_Rate', 0.0):.3f} | "
                f"Miss={flat.get('events/Missed_Gate_Rate', 0.0):.3f} | "
                f"Dev={flat.get('events/Deviation_Rate', 0.0):.3f}",
                flush=True,
            )

    elapsed = time.time() - t0
    fps = int(args_cli.steps) * env.num_envs / max(elapsed, 1.0e-6)

    print_ok(f"random rollout finished: {args_cli.steps} steps, fps={fps:,.2f} env steps/s")
    print_summary(summarize(records))


def run_tests() -> None:
    heading("🚁 Quadrotor / Crazyflie Task4 Vision Gate Racing Env 全量测试启动")

    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    test_config()

    env = None
    try:
        env = build_env()
        test_init_reset(env)
        test_obs_slices(env)
        test_step_structure(env)
        test_wrench_mapping(env)
        test_control_response(env)
        test_gate_progress_and_success(env)
        test_terminal_events(env)
        random_rollout(env)

        heading("✅ Quadrotor / Crazyflie Task4 环境测试全部通过")

    except Exception as exc:
        print("\n❌ Quadrotor / Crazyflie Task4 环境测试失败：")
        print(type(exc).__name__, ":", exc)
        raise

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    try:
        run_tests()
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
