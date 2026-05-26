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

parser = argparse.ArgumentParser(description="Quadrotor / Crazyflie Task3 dynamic obstacle navigation environment test")
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=200)
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

from quadrotor_rl.tasks.task3.task3_config import Task3Config
from quadrotor_rl.tasks.task3.task3_env import QuadrotorTask3Env


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
    print("Quadrotor Task3 Env Test Summary")
    print("=" * 130)
    print(f"{'metric':<72} | {'mean':>12} | {'min':>12} | {'p50':>12} | {'max':>12}")
    print("-" * 130)
    for key in sorted(summary.keys()):
        r = summary[key]
        print(f"{key:<72} | {r['mean']:>12.6f} | {r['min']:>12.6f} | {r['p50']:>12.6f} | {r['max']:>12.6f}")
    print("=" * 130 + "\n")


def check_obs(env: QuadrotorTask3Env, obs: torch.Tensor) -> None:
    check_shape("obs", obs, (env.num_envs, env.num_observations))
    assert_finite_tensor("obs", obs)
    assert obs.abs().max().item() <= float(env.cfg.obs_clip) + 1e-5


def check_state(env: QuadrotorTask3Env, state: torch.Tensor) -> None:
    check_shape("state", state, (env.num_envs, env.num_privileged_obs))
    assert_finite_tensor("state", state)


def last_frame(env: QuadrotorTask3Env, obs: torch.Tensor) -> torch.Tensor:
    start = (int(env.cfg.frame_stack) - 1) * int(env.cfg.single_actor_obs_dim)
    end = start + int(env.cfg.single_actor_obs_dim)
    return obs[:, start:end]


def build_env() -> QuadrotorTask3Env:
    cfg = Task3Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(args_cli.test_device)
    cfg.seed = int(args_cli.seed)
    cfg.print_debug_info = bool(args_cli.print_names)

    if bool(args_cli.quick):
        cfg.num_envs = min(cfg.num_envs, 2)
        args_cli.steps = min(int(args_cli.steps), 60)

    cfg.validate()
    return QuadrotorTask3Env(cfg)


def test_config() -> None:
    heading("[测试 1] Task3Config 基础配置检测")

    cfg = Task3Config()
    cfg.validate()

    assert cfg.action_dim == 4
    assert cfg.single_actor_obs_dim == 75
    assert cfg.frame_stack == 4
    assert cfg.actor_obs_dim == 300
    assert cfg.critic_obs_dim == 300
    assert cfg.lidar_num_rays == 24
    assert cfg.num_static_obs == 30
    assert cfg.num_dynamic_obs == 4

    print_ok(f"action_dim = {cfg.action_dim}")
    print_ok(f"single_actor_obs_dim = {cfg.single_actor_obs_dim}")
    print_ok(f"actor_obs_dim = {cfg.actor_obs_dim}")
    print_ok(f"critic_obs_dim = {cfg.critic_obs_dim}")
    print_ok(f"lidar = {cfg.lidar_num_rays} rays, range = {cfg.lidar_max_range} m")
    print_ok(f"static / dynamic obs = {cfg.num_static_obs} / {cfg.num_dynamic_obs}")
    print_ok(f"hover_thrust = {cfg.hover_thrust:.6f} N")


def test_init_reset(env: QuadrotorTask3Env) -> None:
    heading("[测试 2] 环境初始化 / reset / obs / world 对齐检测")

    obs, info = env.reset()

    check_obs(env, obs)
    assert "state" in info
    check_state(env, info["state"])

    assert env.observation_space.shape == (300,)
    assert env.state_space.shape == (300,)
    assert env.action_space.shape == (4,)

    root_local = env._root_pos_w() - env.env_origins
    err = torch.norm(root_local - env.world.start_pos, dim=-1)

    assert err.max().item() < 5e-4, f"reset root pos not aligned: {err.max().item():.8f}"

    goal_dist = env.world.distance_to_goal(root_local)
    assert goal_dist.min().item() >= env.cfg.min_start_goal_dist - 1.0e-4
    assert goal_dist.max().item() <= env.cfg.max_start_goal_dist + 1.0e-4

    print_ok(f"asset_source = {env.asset_source}")
    print_ok(f"num_bodies = {getattr(env.drone, 'num_bodies', '<unknown>')}")
    print_ok(f"num_joints = {getattr(env.drone, 'num_joints', '<unknown>')}")
    print_ok(f"obs shape = {tuple(obs.shape)}")
    print_ok(f"state shape = {tuple(info['state'].shape)}")
    print_ok(f"reset position max error = {err.max().item():.8f}")
    print_ok(f"start-goal distance mean = {goal_dist.mean().item():.3f}")

    if args_cli.print_names:
        print("body_names:")
        for i, name in enumerate(list(getattr(env.drone, "body_names", []))):
            print(f"  body[{i:02d}] = {name}")
        print("joint_names:")
        for i, name in enumerate(list(getattr(env.drone, "joint_names", []))):
            print(f"  joint[{i:02d}] = {name}")


def test_obs_slices(env: QuadrotorTask3Env) -> None:
    heading("[测试 3] obs 切片范围检测：goal / lidar / lidar_delta / risk")

    obs, info = env.reset()
    check_obs(env, obs)

    frame = last_frame(env, obs)

    goal_vec = frame[:, 0:3]
    goal_dist = frame[:, 3:4]
    heading_sincos = frame[:, 4:6]
    lin_vel = frame[:, 6:9]
    ang_vel = frame[:, 9:12]
    gravity = frame[:, 12:15]
    action = frame[:, 15:19]
    lidar = frame[:, 19:43]
    lidar_delta = frame[:, 43:67]
    risk = frame[:, 67:75]

    for name, tensor, shape in [
        ("goal_vec", goal_vec, (env.num_envs, 3)),
        ("goal_dist", goal_dist, (env.num_envs, 1)),
        ("heading_sincos", heading_sincos, (env.num_envs, 2)),
        ("lin_vel", lin_vel, (env.num_envs, 3)),
        ("ang_vel", ang_vel, (env.num_envs, 3)),
        ("gravity", gravity, (env.num_envs, 3)),
        ("action", action, (env.num_envs, 4)),
        ("lidar", lidar, (env.num_envs, env.cfg.lidar_num_rays)),
        ("lidar_delta", lidar_delta, (env.num_envs, env.cfg.lidar_num_rays)),
        ("risk", risk, (env.num_envs, 8)),
    ]:
        check_shape(name, tensor, shape)
        assert_finite_tensor(name, tensor)

    assert lidar.min().item() >= -1.0e-6
    assert lidar.max().item() <= 1.0 + 1.0e-6
    assert lidar_delta.min().item() >= -1.0 - 1.0e-6
    assert lidar_delta.max().item() <= 1.0 + 1.0e-6
    assert risk[:, :7].min().item() >= -1.0e-6
    assert risk[:, :7].max().item() <= 1.0 + 1.0e-6
    assert risk[:, 7].min().item() >= -1.0e-6

    sincos_norm = torch.norm(heading_sincos, dim=-1)
    assert torch.allclose(sincos_norm, torch.ones_like(sincos_norm), atol=2.0e-3)

    print_ok(f"last frame shape = {tuple(frame.shape)}")
    print_ok(f"lidar min/mean/max = {lidar.min().item():.3f} / {lidar.mean().item():.3f} / {lidar.max().item():.3f}")
    print_ok(f"risk mean = {risk.mean().item():.3f}")
    print_ok("obs slice range check passed")


def test_step_structure(env: QuadrotorTask3Env) -> None:
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

    for group in ["reward_components", "events", "telemetry", "debug", "task3_stats"]:
        assert group in info, f"info missing group: {group}"

    for key in ["R_Height", "R_Approach", "R_Dir", "R_Repulsion", "Total"]:
        assert key in info["reward_components"], f"reward_components missing {key}"

    for key in ["Success_Rate", "Crash_Rate", "Deviation_Rate", "Obstacle_Collision_Rate", "Timeout_Rate", "Done_Rate"]:
        assert key in info["events"], f"events missing {key}"

    for key in ["Goal_Dist", "Goal_Dist_XY", "Min_Lidar", "Vel_Toward", "Heading_Align", "Pos_Z"]:
        assert key in info["telemetry"], f"telemetry missing {key}"

    print_ok(f"reward mean = {reward.mean().item():+.6f}")
    print_ok(f"terminated count = {terminated.sum().item()}")
    print_ok(f"truncated count = {truncated.sum().item()}")


def test_curriculum_hook(env: QuadrotorTask3Env) -> None:
    heading("[测试 5] set_curriculum 课程接口检测")

    assert hasattr(env, "set_curriculum"), "Task3 env missing set_curriculum()"

    env.set_curriculum(num_static=0, num_dynamic=0, max_sg_dist=10.0)
    env.reset()

    assert env.cfg.num_static_obs == 0
    assert env.cfg.num_dynamic_obs == 0
    assert env.world.cfg.num_static_obs == 0
    assert env.world.cfg.num_dynamic_obs == 0
    assert env.world.static_valid.float().sum().item() == 0.0
    assert env.world.dynamic_valid.float().sum().item() == 0.0

    root_local = env._root_pos_w() - env.env_origins
    goal_dist = env.world.distance_to_goal(root_local)
    assert goal_dist.max().item() <= 10.0 + 1.0e-4

    env.set_curriculum(num_static=15, num_dynamic=0, max_sg_dist=30.0)
    env.reset()
    assert env.cfg.num_static_obs == 15
    assert env.cfg.num_dynamic_obs == 0
    assert env.world.static_valid.float().sum().item() > 0.0
    assert env.world.dynamic_valid.float().sum().item() == 0.0

    env.set_curriculum(num_static=30, num_dynamic=4, max_sg_dist=45.0)
    env.reset()
    assert env.cfg.num_static_obs == 30
    assert env.cfg.num_dynamic_obs == 4
    assert env.world.static_valid.float().sum().item() > 0.0
    assert env.world.dynamic_valid.float().sum().item() > 0.0

    print_ok("stage 1 no-obstacle curriculum passed")
    print_ok("stage 2 static-obstacle curriculum passed")
    print_ok("stage 3 dynamic-obstacle curriculum passed")


def test_world_dynamics_from_env(env: QuadrotorTask3Env) -> None:
    heading("[测试 6] env.step 驱动动态障碍物更新检测")

    env.reset()

    before = env.world.dynamic_pos.clone()
    action = torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32, device=env.device)

    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(action)

    after = env.world.dynamic_pos.clone()
    valid = env.world.dynamic_valid
    moved = torch.norm(after - before, dim=-1)

    assert_finite_tensor("dynamic_pos", env.world.dynamic_pos)
    assert_finite_tensor("dynamic_vel", env.world.dynamic_vel)

    if valid.any().item():
        assert moved[valid].mean().item() > 0.001, "dynamic obstacles did not move through env.step"

    print_ok(f"dynamic movement mean = {moved[valid].mean().item() if valid.any().item() else 0.0:.6f}")


def run_action_response(env: QuadrotorTask3Env, action_value: torch.Tensor, steps: int = 40):
    env.set_curriculum(num_static=0, num_dynamic=0, max_sg_dist=10.0)
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


def test_control_response(env: QuadrotorTask3Env) -> None:
    heading("[测试 7] 动作控制响应检测：上升 / 下降 / 偏航")

    dz_up, _, *_ = run_action_response(env, torch.ones(4, dtype=torch.float32), steps=40)
    dz_down, _, *_ = run_action_response(env, -torch.ones(4, dtype=torch.float32), steps=40)

    yaw_action = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float32)
    _, dyaw, *_ = run_action_response(env, yaw_action, steps=40)

    print_ok(f"up action delta z mean = {dz_up.mean().item():+.6f}")
    print_ok(f"down action delta z mean = {dz_down.mean().item():+.6f}")
    print_ok(f"yaw action delta yaw mean = {dyaw.mean().item():+.6f}")

    assert dz_up.mean().item() > -0.02, "up action did not produce valid lift response"
    assert dz_up.mean().item() > dz_down.mean().item(), "up action should produce more lift than down action"
    assert torch.isfinite(dyaw).all().item()


def test_wrench_mapping(env: QuadrotorTask3Env) -> None:
    heading("[测试 8] 四旋翼动作到 root wrench 映射白盒检测")

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


def test_terminal_events(env: QuadrotorTask3Env) -> None:
    heading("[测试 9] 手动触发 collision / success / out-of-bounds / flip / timeout 事件检测")

    # Previous control-response tests intentionally switch to no-obstacle curriculum.
    # Restore active obstacle curriculum before terminal-event tests.
    env.set_curriculum(num_static=30, num_dynamic=4, max_sg_dist=10.0)

    env.reset()
    low = env.world.start_pos.clone()
    low[:, 2] = float(env.cfg.min_flight_z - 0.05)
    env.set_root_pose_for_test(low)
    reward, terminated, truncated, info = env.check_events_for_test()
    assert terminated.float().mean().item() > 0.99
    assert info["events"]["Crash_Rate"] > 0.99
    print_ok(f"floor crash triggered, Crash_Rate={info['events']['Crash_Rate']:.6f}")

    env.reset()
    root = env._root_pos_w() - env.env_origins
    env.world.static_valid[:] = False
    env.world.dynamic_valid[:] = False
    env.world.static_pos[:, 0, :] = root[:, :2]
    env.world.static_radius[:, 0] = 1.0
    env.world.static_valid[:, 0] = True
    env.set_root_pose_for_test(root)
    reward, terminated, truncated, info = env.check_events_for_test()
    assert terminated.float().mean().item() > 0.99
    assert info["events"]["Obstacle_Collision_Rate"] > 0.99
    print_ok(f"obstacle collision triggered, rate={info['events']['Obstacle_Collision_Rate']:.6f}")

    env.reset()
    goal = env.world.goal_pos.clone()
    env.set_root_pose_for_test(goal)
    reward, terminated, truncated, info = env.check_events_for_test()
    assert truncated.float().mean().item() > 0.99
    assert info["events"]["Success_Rate"] > 0.99
    print_ok(f"success triggered, Success_Rate={info['events']['Success_Rate']:.6f}")

    env.reset()
    far = env.world.start_pos.clone()
    far[:, 0] = float(env.cfg.arena_half + 2.0)
    far[:, 2] = float(env.cfg.start_goal_z)
    env.set_root_pose_for_test(far)
    reward, terminated, truncated, info = env.check_events_for_test()
    assert terminated.float().mean().item() > 0.99
    assert info["events"]["Deviation_Rate"] > 0.99
    print_ok(f"out-of-bounds deviation triggered, Deviation_Rate={info['events']['Deviation_Rate']:.6f}")

    env.reset()
    start = env.world.start_pos.clone()
    env.set_root_pose_for_test(start, roll=float(env.cfg.max_roll_pitch + 0.20))
    reward, terminated, truncated, info = env.check_events_for_test()
    assert terminated.float().mean().item() > 0.99
    assert info["events"]["Flip_Crash_Rate"] > 0.99
    print_ok(f"flip crash triggered, Flip_Crash_Rate={info['events']['Flip_Crash_Rate']:.6f}")

    env.reset()
    env.episode_steps[:] = int(env.cfg.max_episode_length)
    reward, terminated, truncated, info = env.check_events_for_test()
    assert truncated.float().mean().item() > 0.99
    print_ok(f"timeout triggered, Timeout_Rate={info['events']['Timeout_Rate']:.6f}")


def random_rollout(env: QuadrotorTask3Env) -> None:
    heading(f"[测试 10] 随机策略 rollout {args_cli.steps} 步，无 NaN / Inf 检测")

    env.set_curriculum(num_static=30, num_dynamic=4, max_sg_dist=45.0)
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
        assert_finite_tensor("current_lidar", env.current_lidar)

        assert env.current_lidar.min().item() >= -1.0e-6
        assert env.current_lidar.max().item() <= 1.0 + 1.0e-6

        if (step + 1) % max(int(args_cli.collect_interval), 1) == 0 or (step + 1) == int(args_cli.steps):
            flat = flat_info(info)
            flat["Reward_Mean_Step"] = reward.mean().item()
            records.append(flat)

            print(
                f" -> Step {step + 1:05d} | "
                f"Reward={reward.mean().item():+.4f} | "
                f"Goal={flat.get('telemetry/Goal_Dist', 0.0):.3f} | "
                f"Prog={flat.get('telemetry/Progress', 0.0):+.3f} | "
                f"Z={flat.get('telemetry/Pos_Z', 0.0):.3f} | "
                f"MinLidar={flat.get('telemetry/Min_Lidar', 0.0):.3f} | "
                f"RP={flat.get('telemetry/RollPitchAbs', 0.0):.3f} | "
                f"Succ={flat.get('events/Success_Rate', 0.0):.3f} | "
                f"Crash={flat.get('events/Crash_Rate', 0.0):.3f} | "
                f"Dev={flat.get('events/Deviation_Rate', 0.0):.3f}",
                flush=True,
            )

    elapsed = time.time() - t0
    fps = int(args_cli.steps) * env.num_envs / max(elapsed, 1.0e-6)

    print_ok(f"random rollout finished: {args_cli.steps} steps, fps={fps:,.2f} env steps/s")
    print_summary(summarize(records))


def run_tests() -> None:
    heading("🚁 Quadrotor / Crazyflie Task3 Dynamic Obstacle Navigation Env 全量测试启动")

    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    test_config()

    env = None
    try:
        env = build_env()
        test_init_reset(env)
        test_obs_slices(env)
        test_step_structure(env)
        test_curriculum_hook(env)
        test_world_dynamics_from_env(env)
        test_control_response(env)
        test_wrench_mapping(env)
        test_terminal_events(env)
        random_rollout(env)

        heading("✅ Quadrotor / Crazyflie Task3 环境测试全部通过")

    except Exception as exc:
        print("\n❌ Quadrotor / Crazyflie Task3 环境测试失败：")
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
