from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quadrotor_rl.tasks.task3.task3_config import Task3Config
from quadrotor_rl.tasks.task3.task3_world import QuadrotorTask3World


parser = argparse.ArgumentParser(
    description="Quadrotor / Crazyflie Task3 analytic dynamic-obstacle world test"
)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test-device", type=str, default="cuda:0")
parser.add_argument("--quick", action="store_true")
parser.add_argument("--print-every", type=int, default=50)

args_cli = parser.parse_args()


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


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().float()
    return {
        "mean": float(x.mean().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def build_world() -> QuadrotorTask3World:
    cfg = Task3Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(args_cli.test_device)
    cfg.seed = int(args_cli.seed)

    if bool(args_cli.quick):
        cfg.num_envs = min(cfg.num_envs, 16)
        args_cli.steps = min(int(args_cli.steps), 80)

    cfg.validate()
    world = QuadrotorTask3World(cfg=cfg, num_envs=cfg.num_envs, device=cfg.device)
    return world


def test_config() -> None:
    heading("[测试 1] Task3Config 基础参数检测")

    cfg = Task3Config()
    cfg.validate()

    assert cfg.arena_size == 50.0
    assert cfg.min_start_goal_dist == 7.0
    assert cfg.max_start_goal_dist == 10.0
    assert cfg.safe_zone_radius == 5.0
    assert cfg.num_static_obs == 30
    assert cfg.num_dynamic_obs == 4
    assert cfg.lidar_num_rays == 24
    assert cfg.lidar_max_range == 10.0
    assert cfg.dynamic_speed == 1.5
    assert cfg.single_actor_obs_dim == 75
    assert cfg.actor_obs_dim == 300
    assert cfg.critic_obs_dim == 300

    print_ok(f"arena_size = {cfg.arena_size}")
    print_ok(f"start-goal distance range = [{cfg.min_start_goal_dist}, {cfg.max_start_goal_dist}]")
    print_ok(f"safe_zone_radius = {cfg.safe_zone_radius}")
    print_ok(f"static / dynamic obstacles = {cfg.num_static_obs} / {cfg.num_dynamic_obs}")
    print_ok(f"lidar = {cfg.lidar_num_rays} rays, range = {cfg.lidar_max_range} m")
    print_ok(f"planned actor_obs_dim = {cfg.actor_obs_dim}")


def test_reset_shapes_and_ranges(world: QuadrotorTask3World) -> None:
    heading("[测试 2] reset / tensor shape / finite / 起终点距离检测")

    start, goal = world.reset()

    n = world.num_envs
    cfg = world.cfg

    check_shape("start", start, (n, 3))
    check_shape("goal", goal, (n, 3))
    check_shape("world.start_pos", world.start_pos, (n, 3))
    check_shape("world.goal_pos", world.goal_pos, (n, 3))
    check_shape("static_pos", world.static_pos, (n, cfg.num_static_obs, 2))
    check_shape("static_radius", world.static_radius, (n, cfg.num_static_obs))
    check_shape("dynamic_pos", world.dynamic_pos, (n, cfg.num_dynamic_obs, 2))
    check_shape("dynamic_vel", world.dynamic_vel, (n, cfg.num_dynamic_obs, 2))
    check_shape("last_lidar", world.last_lidar, (n, cfg.lidar_num_rays))

    for name, tensor in world.get_world_tensors().items():
        if tensor.dtype == torch.bool:
            continue
        assert_finite_tensor(name, tensor)

    dist_xy = torch.norm(world.goal_pos[:, :2] - world.start_pos[:, :2], dim=-1)

    assert dist_xy.min().item() >= cfg.min_start_goal_dist - 1.0e-4
    assert dist_xy.max().item() <= cfg.max_start_goal_dist + 1.0e-4
    assert torch.allclose(world.start_pos[:, 2], torch.full_like(world.start_pos[:, 2], cfg.start_goal_z))
    assert torch.allclose(world.goal_pos[:, 2], torch.full_like(world.goal_pos[:, 2], cfg.start_goal_z))

    bound = float(cfg.start_goal_bound)
    assert world.start_pos[:, :2].abs().max().item() <= bound + 1.0e-4
    assert world.goal_pos[:, :2].abs().max().item() <= bound + 1.0e-4

    stats = tensor_stats(dist_xy)
    print_ok(f"num_envs = {n}")
    print_ok(f"start-goal dist mean/min/max = {stats['mean']:.3f} / {stats['min']:.3f} / {stats['max']:.3f}")
    print_ok(f"static_valid mean = {world.static_valid.float().mean().item():.3f}")
    print_ok(f"dynamic_valid mean = {world.dynamic_valid.float().mean().item():.3f}")


def test_obstacle_geometry(world: QuadrotorTask3World) -> None:
    heading("[测试 3] 障碍物安全区 / 边界 / 间距几何检测")

    world.reset()
    cfg = world.cfg
    half = float(cfg.arena_half)

    start_xy = world.start_pos[:, :2]
    goal_xy = world.goal_pos[:, :2]

    centers = []
    radii = []
    valid = []

    if cfg.num_static_obs > 0:
        centers.append(world.static_pos)
        radii.append(world.static_radius)
        valid.append(world.static_valid)

    if cfg.num_dynamic_obs > 0:
        centers.append(world.dynamic_pos)
        radii.append(world.dynamic_radius)
        valid.append(world.dynamic_valid)

    all_centers = torch.cat(centers, dim=1)
    all_radii = torch.cat(radii, dim=1)
    all_valid = torch.cat(valid, dim=1)

    n, m, _ = all_centers.shape

    # Safe-zone check.
    d_start = torch.norm(all_centers - start_xy[:, None, :], dim=-1)
    d_goal = torch.norm(all_centers - goal_xy[:, None, :], dim=-1)

    safe_required = float(cfg.safe_zone_radius) + all_radii

    assert ((d_start >= safe_required - 1.0e-4) | (~all_valid)).all().item()
    assert ((d_goal >= safe_required - 1.0e-4) | (~all_valid)).all().item()

    # Boundary check.
    assert ((all_centers[..., 0].abs() + all_radii <= half + 1.0e-4) | (~all_valid)).all().item()
    assert ((all_centers[..., 1].abs() + all_radii <= half + 1.0e-4) | (~all_valid)).all().item()

    # Pairwise spacing check.
    for env_id in range(n):
        ids = torch.nonzero(all_valid[env_id], as_tuple=False).flatten()
        if ids.numel() <= 1:
            continue

        c = all_centers[env_id, ids]
        r = all_radii[env_id, ids]

        diff = c[:, None, :] - c[None, :, :]
        d = torch.norm(diff, dim=-1)

        required = r[:, None] + r[None, :] + float(cfg.min_obs_gap)

        eye = torch.eye(ids.numel(), dtype=torch.bool, device=world.device)
        ok = (d >= required - 1.0e-4) | eye

        assert ok.all().item(), f"obstacle spacing failed at env {env_id}"

    print_ok(f"checked {n} envs, up to {m} obstacles each")
    print_ok("start/goal safe-zone check passed")
    print_ok("boundary check passed")
    print_ok("pairwise spacing check passed")


def test_dynamic_motion(world: QuadrotorTask3World) -> None:
    heading("[测试 4] 动态障碍物游走 / 速度维持 / 边界反弹检测")

    world.reset()

    cfg = world.cfg
    before_pos = world.dynamic_pos.clone()

    for _ in range(int(args_cli.steps)):
        world.step_dynamics(dt=cfg.policy_dt)

    after_pos = world.dynamic_pos.clone()
    speed = torch.norm(world.dynamic_vel, dim=-1)

    assert_finite_tensor("dynamic_pos after step", world.dynamic_pos)
    assert_finite_tensor("dynamic_vel after step", world.dynamic_vel)

    valid = world.dynamic_valid
    moved = torch.norm(after_pos - before_pos, dim=-1)

    if valid.any().item():
        assert moved[valid].mean().item() > 0.01, "dynamic obstacles did not move"
        assert speed[valid].mean().item() > cfg.dynamic_speed * 0.70, "dynamic speed not maintained"

    half = float(cfg.arena_half)
    bound = half - world.dynamic_radius
    assert ((world.dynamic_pos[..., 0].abs() <= bound + 1.0e-4) | (~valid)).all().item()
    assert ((world.dynamic_pos[..., 1].abs() <= bound + 1.0e-4) | (~valid)).all().item()

    print_ok(f"dynamic movement mean = {moved[valid].mean().item():.4f} m")
    print_ok(f"dynamic speed mean = {speed[valid].mean().item():.4f} m/s")
    print_ok("dynamic boundary clamp / reflection check passed")


def test_lidar_basic(world: QuadrotorTask3World) -> None:
    heading("[测试 5] LiDAR shape / value range / finite 检测")

    world.reset()

    yaw = torch.zeros(world.num_envs, dtype=torch.float32, device=world.device)
    lidar = world.get_lidar_scan(world.start_pos, yaw)

    check_shape("lidar", lidar, (world.num_envs, world.cfg.lidar_num_rays))
    assert_finite_tensor("lidar", lidar)
    assert lidar.min().item() >= -1.0e-6
    assert lidar.max().item() <= 1.0 + 1.0e-6

    print_ok(f"lidar shape = {tuple(lidar.shape)}")
    print_ok(f"lidar min/mean/max = {lidar.min().item():.3f} / {lidar.mean().item():.3f} / {lidar.max().item():.3f}")


def test_lidar_known_obstacle_and_wall(world: QuadrotorTask3World) -> None:
    heading("[测试 6] LiDAR 已知障碍物命中 / 边界墙命中检测")

    world.reset()

    cfg = world.cfg
    n = world.num_envs
    half = float(cfg.arena_half)

    # Make envs deterministic for white-box LiDAR tests.
    world.static_valid[:] = False
    world.dynamic_valid[:] = False

    # Place a circular obstacle 2m in front of the drone along +X.
    world.static_pos[:, 0, :] = torch.tensor([2.0, 0.0], dtype=torch.float32, device=world.device).view(1, 2)
    world.static_radius[:, 0] = 0.5
    world.static_valid[:, 0] = True

    pos = torch.zeros((n, 3), dtype=torch.float32, device=world.device)
    pos[:, 2] = float(cfg.start_goal_z)
    yaw = torch.zeros(n, dtype=torch.float32, device=world.device)

    lidar = world.get_lidar_scan(pos, yaw)

    front = lidar[:, 0]
    assert front.mean().item() < 0.30, f"front ray should hit known obstacle, got {front.mean().item():.3f}"

    # Disable obstacles and test wall hit near +X boundary.
    world.static_valid[:] = False
    world.dynamic_valid[:] = False

    wall_pos = torch.zeros((n, 3), dtype=torch.float32, device=world.device)
    wall_pos[:, 0] = half - 1.0
    wall_pos[:, 1] = 0.0
    wall_pos[:, 2] = float(cfg.start_goal_z)

    wall_lidar = world.get_lidar_scan(wall_pos, yaw)
    wall_front = wall_lidar[:, 0]

    assert wall_front.mean().item() < 0.20, f"front ray should hit wall, got {wall_front.mean().item():.3f}"

    print_ok(f"known obstacle front lidar mean = {front.mean().item():.4f}")
    print_ok(f"wall front lidar mean = {wall_front.mean().item():.4f}")


def test_events(world: QuadrotorTask3World) -> None:
    heading("[测试 7] collision / success / out-of-bounds / goal distance 检测")

    world.reset()

    n = world.num_envs
    cfg = world.cfg

    # Collision with known obstacle.
    world.static_valid[:] = False
    world.dynamic_valid[:] = False

    center = torch.zeros((n, 2), dtype=torch.float32, device=world.device)
    center[:, 0] = 1.0
    center[:, 1] = 0.0

    world.static_pos[:, 0, :] = center
    world.static_radius[:, 0] = 0.8
    world.static_valid[:, 0] = True

    drone_pos = torch.zeros((n, 3), dtype=torch.float32, device=world.device)
    drone_pos[:, 0] = 1.0
    drone_pos[:, 1] = 0.0
    drone_pos[:, 2] = float(cfg.start_goal_z)

    collision = world.check_obstacle_collision(drone_pos)
    assert collision.float().mean().item() > 0.99

    # Success at goal.
    success = world.check_success(world.goal_pos)
    assert success.float().mean().item() > 0.99

    # Out of bounds: XY.
    oob_xy = world.goal_pos.clone()
    oob_xy[:, 0] = float(cfg.arena_half + 1.0)
    out_xy = world.check_out_of_bounds(oob_xy)
    assert out_xy.float().mean().item() > 0.99

    # Out of bounds: Z too low.
    oob_z = world.goal_pos.clone()
    oob_z[:, 2] = float(cfg.min_flight_z - 0.05)
    out_z = world.check_out_of_bounds(oob_z)
    assert out_z.float().mean().item() > 0.99

    goal_vec = world.goal_vector(world.start_pos)
    goal_dist = world.distance_to_goal(world.start_pos)

    check_shape("goal_vec", goal_vec, (n, 3))
    check_shape("goal_dist", goal_dist, (n,))
    assert_finite_tensor("goal_vec", goal_vec)
    assert_finite_tensor("goal_dist", goal_dist)
    assert goal_dist.min().item() >= cfg.min_start_goal_dist - 1.0e-4
    assert goal_dist.max().item() <= math.sqrt(cfg.max_start_goal_dist**2 + 1.0e-6) + 1.0e-3

    print_ok(f"collision rate = {collision.float().mean().item():.3f}")
    print_ok(f"success rate at goal = {success.float().mean().item():.3f}")
    print_ok(f"out-of-bounds xy rate = {out_xy.float().mean().item():.3f}")
    print_ok(f"out-of-bounds z rate = {out_z.float().mean().item():.3f}")
    print_ok(f"goal distance mean = {goal_dist.mean().item():.3f}")


def test_risk_features(world: QuadrotorTask3World) -> None:
    heading("[测试 8] risk features 检测")

    world.reset()

    yaw = torch.zeros(world.num_envs, dtype=torch.float32, device=world.device)
    risk = world.risk_features(world.start_pos, yaw)

    check_shape("risk", risk, (world.num_envs, 8))
    assert_finite_tensor("risk", risk)

    assert risk[:, :7].min().item() >= -1.0e-6
    assert risk[:, :7].max().item() <= 1.0 + 1.0e-6
    assert risk[:, 7].min().item() >= 0.0

    print_ok(f"risk shape = {tuple(risk.shape)}")
    print_ok(f"risk mean = {risk.mean().item():.4f}")
    print_ok(f"risk first env = {risk[0].detach().cpu().numpy()}")


def random_world_rollout(world: QuadrotorTask3World) -> None:
    heading(f"[测试 9] 随机位置 LiDAR / 动态世界 rollout {args_cli.steps} 步，无 NaN / Inf")

    world.reset()

    cfg = world.cfg
    n = world.num_envs
    half = float(cfg.arena_half)

    records: List[Dict[str, float]] = []
    t0 = time.time()

    for step in range(int(args_cli.steps)):
        world.step_dynamics(dt=cfg.policy_dt)

        pos = torch.empty((n, 3), dtype=torch.float32, device=world.device)
        pos[:, 0].uniform_(-half + 1.0, half - 1.0)
        pos[:, 1].uniform_(-half + 1.0, half - 1.0)
        pos[:, 2].uniform_(float(cfg.min_flight_z + 0.2), float(cfg.max_flight_z - 0.2))

        yaw = torch.empty((n,), dtype=torch.float32, device=world.device).uniform_(-math.pi, math.pi)

        lidar = world.get_lidar_scan(pos, yaw)
        collision = world.check_obstacle_collision(pos)
        success = world.check_success(pos)
        out = world.check_out_of_bounds(pos)
        risk = world.risk_features(pos, yaw)
        nearest = world.nearest_obstacle_distance(pos)
        goal_dist = world.distance_to_goal(pos)

        for name, tensor in [
            ("lidar", lidar),
            ("risk", risk),
            ("nearest", nearest),
            ("goal_dist", goal_dist),
            ("dynamic_pos", world.dynamic_pos),
            ("dynamic_vel", world.dynamic_vel),
        ]:
            assert_finite_tensor(name, tensor)

        assert lidar.min().item() >= -1.0e-6
        assert lidar.max().item() <= 1.0 + 1.0e-6

        if (step + 1) % max(int(args_cli.print_every), 1) == 0 or (step + 1) == int(args_cli.steps):
            row = {
                "step": float(step + 1),
                "lidar_min": float(lidar.min().item()),
                "lidar_mean": float(lidar.mean().item()),
                "collision_rate": float(collision.float().mean().item()),
                "success_rate": float(success.float().mean().item()),
                "out_of_bounds_rate": float(out.float().mean().item()),
                "risk_mean": float(risk.mean().item()),
                "nearest_mean": float(nearest.mean().item()),
                "goal_dist_mean": float(goal_dist.mean().item()),
            }
            records.append(row)

            print(
                f" -> Step {step + 1:05d} | "
                f"lidar_min={row['lidar_min']:.3f} | "
                f"lidar_mean={row['lidar_mean']:.3f} | "
                f"collision={row['collision_rate']:.3f} | "
                f"success={row['success_rate']:.3f} | "
                f"risk={row['risk_mean']:.3f} | "
                f"nearest={row['nearest_mean']:.3f} | "
                f"goal_dist={row['goal_dist_mean']:.3f}",
                flush=True,
            )

    elapsed = time.time() - t0
    fps = int(args_cli.steps) * n / max(elapsed, 1.0e-6)

    print_ok(f"random world rollout finished: {args_cli.steps} steps, fps={fps:,.2f} env world-steps/s")

    if records:
        last = records[-1]
        print_ok(
            "last stats: "
            f"lidar_min={last['lidar_min']:.3f}, "
            f"collision={last['collision_rate']:.3f}, "
            f"risk={last['risk_mean']:.3f}"
        )


def run_tests() -> None:
    heading("🚁 Quadrotor / Crazyflie Task3 Analytic Dynamic-Obstacle World 全量测试启动")

    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    test_config()

    world = build_world()

    try:
        test_reset_shapes_and_ranges(world)
        test_obstacle_geometry(world)
        test_dynamic_motion(world)
        test_lidar_basic(world)
        test_lidar_known_obstacle_and_wall(world)
        test_events(world)
        test_risk_features(world)
        random_world_rollout(world)

        heading("✅ Quadrotor / Crazyflie Task3 世界场景测试全部通过")

        print("World summary:")
        for k, v in world.summary().items():
            print(f"  {k}: {v}")

    except Exception as exc:
        print("\n❌ Quadrotor / Crazyflie Task3 世界场景测试失败：")
        print(type(exc).__name__, ":", exc)
        raise


if __name__ == "__main__":
    run_tests()
