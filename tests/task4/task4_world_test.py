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

from quadrotor_rl.tasks.task4.task4_config import Task4Config
from quadrotor_rl.tasks.task4.task4_world import QuadrotorTask4World


parser = argparse.ArgumentParser(
    description="Quadrotor / Crazyflie Task4 analytic vision gate-racing world test"
)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=200)
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


def build_world() -> QuadrotorTask4World:
    cfg = Task4Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(args_cli.test_device)
    cfg.seed = int(args_cli.seed)

    if bool(args_cli.quick):
        cfg.num_envs = min(cfg.num_envs, 16)
        args_cli.steps = min(int(args_cli.steps), 80)

    cfg.validate()
    return QuadrotorTask4World(cfg=cfg, num_envs=cfg.num_envs, device=cfg.device)


def test_config() -> None:
    heading("[测试 1] Task4Config 基础参数检测")

    cfg = Task4Config()
    cfg.validate()

    assert cfg.arena_length == 30.0
    assert cfg.arena_width == 10.0
    assert cfg.arena_height == 5.0
    assert cfg.start_pos == (-12.0, 0.0, 1.5)
    assert cfg.num_gates == 5
    assert cfg.gate_size == 0.5
    assert cfg.gate_thickness == 0.05
    assert cfg.max_roll_pitch_deg == 45.0
    assert cfg.cam_res_w == 64
    assert cfg.cam_res_h == 64
    assert cfg.cam_fov_deg == 110.0
    assert cfg.cam_far == 10.0
    assert cfg.depth_dim == 4096
    assert cfg.single_actor_obs_dim == 4128
    assert cfg.actor_obs_dim == 4128
    assert cfg.critic_obs_dim == 4128

    print_ok(f"arena = {cfg.arena_length} x {cfg.arena_width} x {cfg.arena_height} m")
    print_ok(f"start_pos = {cfg.start_pos}")
    print_ok(f"num_gates = {cfg.num_gates}")
    print_ok(f"gate inner size = {cfg.gate_size} m, thickness = {cfg.gate_thickness} m")
    print_ok(f"camera = {cfg.cam_res_w} x {cfg.cam_res_h}, fov = {cfg.cam_fov_deg}, far = {cfg.cam_far}")
    print_ok(f"planned actor_obs_dim = {cfg.actor_obs_dim}")


def test_reset_shapes_and_ranges(world: QuadrotorTask4World) -> None:
    heading("[测试 2] reset / tensor shape / finite / gate range 检测")

    start, gate_poses = world.reset()

    n = world.num_envs
    cfg = world.cfg
    g = cfg.num_gates

    check_shape("start", start, (n, 3))
    check_shape("gate_pos", world.gate_pos, (n, g, 3))
    check_shape("gate_quat", world.gate_quat, (n, g, 4))
    check_shape("gate_rot", world.gate_rot, (n, g, 3, 3))
    check_shape("gate_tangent", world.gate_tangent, (n, g, 3))
    check_shape("gate_valid", world.gate_valid, (n, g))
    check_shape("target_gate_idx", world.target_gate_idx, (n,))
    check_shape("centerline", world.centerline, (n, cfg.centerline_samples, 3))
    check_shape("last_depth", world.last_depth, (n, 1, cfg.cam_res_h, cfg.cam_res_w))

    assert isinstance(gate_poses, dict)
    assert set(["pos", "quat_wxyz", "rot", "tangent", "valid"]).issubset(set(gate_poses.keys()))

    for name, tensor in world.get_world_tensors().items():
        if tensor.dtype == torch.bool:
            continue
        assert_finite_tensor(name, tensor)

    expected_start = torch.tensor(cfg.start_pos, dtype=torch.float32, device=world.device).view(1, 3)
    assert torch.allclose(world.start_pos, expected_start.repeat(n, 1), atol=1e-6)

    x = world.gate_pos[..., 0]
    y = world.gate_pos[..., 1]
    z = world.gate_pos[..., 2]

    assert x.min().item() >= cfg.gate_start_x - 1.0e-5
    assert x.max().item() <= cfg.gate_end_x + 1.0e-5

    assert y.min().item() >= cfg.gate_y_range[0] - 1.0e-5
    assert y.max().item() <= cfg.gate_y_range[1] + 1.0e-5

    assert z.min().item() >= cfg.gate_z_range[0] - 1.0e-5
    assert z.max().item() <= cfg.gate_z_range[1] + 1.0e-5

    # X positions should be sorted along the racing direction.
    dx = world.gate_pos[:, 1:, 0] - world.gate_pos[:, :-1, 0]
    assert (dx > 0.0).all().item()

    quat_norm = torch.norm(world.gate_quat, dim=-1)
    tangent_norm = torch.norm(world.gate_tangent, dim=-1)

    assert torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=2.0e-4)
    assert torch.allclose(tangent_norm, torch.ones_like(tangent_norm), atol=2.0e-4)

    print_ok(f"num_envs = {n}")
    print_ok(f"gate_pos shape = {tuple(world.gate_pos.shape)}")
    print_ok(f"gate x range = [{x.min().item():.3f}, {x.max().item():.3f}]")
    print_ok(f"gate y range = [{y.min().item():.3f}, {y.max().item():.3f}]")
    print_ok(f"gate z range = [{z.min().item():.3f}, {z.max().item():.3f}]")
    print_ok(f"quat norm mean = {quat_norm.mean().item():.6f}")
    print_ok(f"tangent norm mean = {tangent_norm.mean().item():.6f}")


def test_gate_orientation_and_centerline(world: QuadrotorTask4World) -> None:
    heading("[测试 3] gate orientation / centerline 几何检测")

    world.reset()

    cfg = world.cfg

    # Gate tangent must equal gate local +Z axis in world coordinates.
    local_z = world.gate_rot[..., :, 2]
    align = torch.sum(local_z * world.gate_tangent, dim=-1)

    assert align.min().item() > 0.999, f"gate tangent not aligned with rot local-z: min={align.min().item():.6f}"

    # Rotation matrices should be orthonormal.
    eye = torch.eye(3, dtype=torch.float32, device=world.device).view(1, 1, 3, 3)
    gram = torch.matmul(world.gate_rot.transpose(-1, -2), world.gate_rot)
    rot_err = torch.abs(gram - eye).max()

    assert rot_err.item() < 5.0e-4, f"gate_rot not orthonormal: max err={rot_err.item():.6e}"

    # Centerline begins at start and ends near final gate.
    start_err = torch.norm(world.centerline[:, 0, :] - world.start_pos, dim=-1)
    end_err = torch.norm(world.centerline[:, -1, :] - world.gate_pos[:, -1, :], dim=-1)

    assert start_err.max().item() < 1.0e-5
    assert end_err.max().item() < 1.0e-4

    # Centerline should mostly move forward in X.
    cx = world.centerline[..., 0]
    dx = cx[:, 1:] - cx[:, :-1]
    assert (dx >= -1.0e-5).float().mean().item() > 0.98

    print_ok(f"gate tangent alignment min = {align.min().item():.6f}")
    print_ok(f"gate rotation orthonormal max error = {rot_err.item():.6e}")
    print_ok(f"centerline start error max = {start_err.max().item():.8f}")
    print_ok(f"centerline end error max = {end_err.max().item():.8f}")


def make_identity_quat(world: QuadrotorTask4World) -> torch.Tensor:
    q = torch.zeros((world.num_envs, 4), dtype=torch.float32, device=world.device)
    q[:, 0] = 1.0
    return q


def set_deterministic_gate_track(world: QuadrotorTask4World, first_x: float = 1.0, spacing: float = 2.0) -> None:
    """Create a simple straight gate track whose gate normal points along +X."""

    cfg = world.cfg
    n = world.num_envs
    g = cfg.num_gates

    normal = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=world.device)
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
    world.centerline[:] = 0.0

    pts = torch.cat([world.start_pos[:, None, :], world.gate_pos], dim=1)
    for env_id in range(n):
        # Reuse the world interpolation routine after the deterministic gates are written.
        world.centerline[env_id] = world._build_centerline_from_gates(env_id)


def test_depth_basic(world: QuadrotorTask4World) -> None:
    heading("[测试 4] analytic depth vision shape / dtype / value range 检测")

    world.reset()

    pos = world.start_pos.clone()
    quat = make_identity_quat(world)

    depth = world.get_depth_vision(pos, quat)

    check_shape("depth", depth, (world.num_envs, 1, world.cfg.cam_res_h, world.cfg.cam_res_w))
    assert_finite_tensor("depth", depth)

    assert depth.min().item() >= -1.0e-6
    assert depth.max().item() <= 1.0 + 1.0e-6
    assert depth.dtype == torch.float32

    print_ok(f"depth shape = {tuple(depth.shape)}")
    print_ok(f"depth dtype = {depth.dtype}")
    print_ok(f"depth min/mean/max = {depth.min().item():.3f} / {depth.mean().item():.3f} / {depth.max().item():.3f}")


def test_depth_known_gate_and_wall(world: QuadrotorTask4World) -> None:
    heading("[测试 5] depth 已知门框命中 / 墙体命中检测")

    set_deterministic_gate_track(world, first_x=1.0, spacing=2.0)

    cfg = world.cfg
    n = world.num_envs

    pos = torch.zeros((n, 3), dtype=torch.float32, device=world.device)
    pos[:, 0] = 0.0
    pos[:, 1] = 0.0
    pos[:, 2] = float(cfg.start_pos[2])

    quat = make_identity_quat(world)

    depth = world.get_depth_vision(pos, quat)
    dmin = depth.view(n, -1).min(dim=-1).values

    assert dmin.mean().item() < 0.25, f"known gate frame should be visible, depth min={dmin.mean().item():.3f}"

    # Disable all gates and put the camera one meter before the +X wall.
    world.gate_valid[:] = False

    wall_pos = pos.clone()
    wall_pos[:, 0] = float(cfg.arena_half_length - 1.0)

    wall_depth = world.get_depth_vision(wall_pos, quat)
    center = wall_depth[:, 0, cfg.cam_res_h // 2, cfg.cam_res_w // 2]
    wall_min = wall_depth.view(n, -1).min(dim=-1).values

    assert center.mean().item() < 0.25, f"front wall should be close, center depth={center.mean().item():.3f}"
    assert wall_min.mean().item() < 0.25, f"front wall should be close, min depth={wall_min.mean().item():.3f}"

    print_ok(f"known gate depth min mean = {dmin.mean().item():.4f}")
    print_ok(f"front wall center depth mean = {center.mean().item():.4f}")
    print_ok(f"front wall min depth mean = {wall_min.mean().item():.4f}")


def test_gate_pass_and_progress(world: QuadrotorTask4World) -> None:
    heading("[测试 6] check_gate_pass / update_gate_progress / success 检测")

    set_deterministic_gate_track(world, first_x=0.0, spacing=2.0)

    n = world.num_envs
    cfg = world.cfg

    prev = torch.zeros((n, 3), dtype=torch.float32, device=world.device)
    curr = torch.zeros_like(prev)

    prev[:, 0] = -1.0
    curr[:, 0] = 1.0
    prev[:, 1] = 0.0
    curr[:, 1] = 0.0
    prev[:, 2] = float(cfg.start_pos[2])
    curr[:, 2] = float(cfg.start_pos[2])

    passed, alpha = world.check_gate_pass(prev, curr)

    assert passed.float().mean().item() > 0.99, "segment should pass through first gate opening"
    assert alpha.min().item() >= 0.0
    assert alpha.max().item() <= 1.0

    advanced = world.update_gate_progress(prev, curr)

    assert advanced.float().mean().item() > 0.99
    assert world.target_gate_idx.float().mean().item() == 1.0
    assert world.check_success().float().mean().item() == 0.0

    world.target_gate_idx[:] = int(cfg.num_gates)
    success = world.check_success()

    assert success.float().mean().item() > 0.99

    print_ok(f"gate pass rate = {passed.float().mean().item():.3f}")
    print_ok(f"cross alpha mean = {alpha.mean().item():.3f}")
    print_ok(f"target_gate_idx after progress = {world.target_gate_idx.float().mean().item():.1f}")
    print_ok(f"success rate after final gate = {success.float().mean().item():.3f}")


def test_gate_collision_and_bounds(world: QuadrotorTask4World) -> None:
    heading("[测试 7] gate frame collision / opening safe / out-of-bounds 检测")

    set_deterministic_gate_track(world, first_x=0.0, spacing=2.0)

    n = world.num_envs
    cfg = world.cfg

    env_ids = torch.arange(n, dtype=torch.long, device=world.device)
    gp = world.gate_pos[env_ids, torch.zeros(n, dtype=torch.long, device=world.device)]
    gr = world.gate_rot[env_ids, torch.zeros(n, dtype=torch.long, device=world.device)]

    local_frame = torch.zeros((n, 3), dtype=torch.float32, device=world.device)
    local_frame[:, 0] = float(cfg.gate_inner_half + cfg.gate_thickness * 0.5)
    frame_pos = gp + torch.bmm(gr, local_frame.unsqueeze(-1)).squeeze(-1)

    collision = world.check_gate_collision(frame_pos)
    assert collision.float().mean().item() > 0.99, "frame position should collide with gate"

    opening_pos = gp.clone()
    opening_collision = world.check_gate_collision(opening_pos)
    assert opening_collision.float().mean().item() < 1.0e-6, "center opening should be safe"

    oob = gp.clone()
    oob[:, 0] = float(cfg.arena_half_length + 1.0)
    out = world.check_out_of_bounds(oob)
    assert out.float().mean().item() > 0.99

    low = gp.clone()
    low[:, 2] = float(cfg.min_flight_z - 0.05)
    out_low = world.check_out_of_bounds(low)
    assert out_low.float().mean().item() > 0.99

    print_ok(f"frame collision rate = {collision.float().mean().item():.3f}")
    print_ok(f"opening collision rate = {opening_collision.float().mean().item():.3f}")
    print_ok(f"xy out-of-bounds rate = {out.float().mean().item():.3f}")
    print_ok(f"z out-of-bounds rate = {out_low.float().mean().item():.3f}")


def test_current_target_gate_features(world: QuadrotorTask4World) -> None:
    heading("[测试 8] current_target_gate_features 检测")

    world.reset()

    pos = world.start_pos.clone()
    quat = make_identity_quat(world)

    depth = world.get_depth_vision(pos, quat)
    features = world.current_target_gate_features(pos, quat)

    check_shape("depth", depth, (world.num_envs, 1, world.cfg.cam_res_h, world.cfg.cam_res_w))
    check_shape("features", features, (world.num_envs, 14))
    assert_finite_tensor("features", features)

    assert features[:, -3:].min().item() >= -1.0e-6
    assert features[:, -3:].max().item() <= 1.0 + 1.0e-6

    print_ok(f"target gate feature shape = {tuple(features.shape)}")
    print_ok(f"feature mean/min/max = {features.mean().item():.4f} / {features.min().item():.4f} / {features.max().item():.4f}")


def random_world_rollout(world: QuadrotorTask4World) -> None:
    heading(f"[测试 9] 随机位置 depth / pass / collision rollout {args_cli.steps} 步，无 NaN / Inf")

    world.reset()

    cfg = world.cfg
    n = world.num_envs

    records: List[Dict[str, float]] = []
    t0 = time.time()

    for step in range(int(args_cli.steps)):
        pos = torch.empty((n, 3), dtype=torch.float32, device=world.device)
        pos[:, 0].uniform_(-float(cfg.arena_half_length) + 0.5, float(cfg.arena_half_length) - 0.5)
        pos[:, 1].uniform_(-float(cfg.arena_half_width) + 0.5, float(cfg.arena_half_width) - 0.5)
        pos[:, 2].uniform_(float(cfg.min_flight_z + 0.2), float(cfg.max_flight_z - 0.2))

        prev = pos.clone()
        curr = pos.clone()
        curr[:, 0] += 0.05

        yaw = torch.empty((n,), dtype=torch.float32, device=world.device).uniform_(-math.pi, math.pi)
        quat = torch.zeros((n, 4), dtype=torch.float32, device=world.device)
        quat[:, 0] = torch.cos(0.5 * yaw)
        quat[:, 3] = torch.sin(0.5 * yaw)

        depth = world.get_depth_vision(pos, quat)
        passed, alpha = world.check_gate_pass(prev, curr)
        advanced = world.update_gate_progress(prev, curr)
        collision = world.check_gate_collision(pos)
        out = world.check_out_of_bounds(pos)
        success = world.check_success()
        features = world.current_target_gate_features(pos, quat)

        for name, tensor in [
            ("depth", depth),
            ("alpha", alpha),
            ("features", features),
            ("gate_pos", world.gate_pos),
            ("gate_quat", world.gate_quat),
            ("gate_tangent", world.gate_tangent),
            ("centerline", world.centerline),
        ]:
            assert_finite_tensor(name, tensor)

        assert depth.min().item() >= -1.0e-6
        assert depth.max().item() <= 1.0 + 1.0e-6

        if (step + 1) % max(int(args_cli.print_every), 1) == 0 or (step + 1) == int(args_cli.steps):
            row = {
                "step": float(step + 1),
                "depth_min": float(depth.min().item()),
                "depth_mean": float(depth.mean().item()),
                "depth_max": float(depth.max().item()),
                "pass_rate": float(passed.float().mean().item()),
                "advanced_rate": float(advanced.float().mean().item()),
                "collision_rate": float(collision.float().mean().item()),
                "out_of_bounds_rate": float(out.float().mean().item()),
                "success_rate": float(success.float().mean().item()),
                "feature_mean": float(features.mean().item()),
                "target_gate_idx_mean": float(world.target_gate_idx.float().mean().item()),
            }
            records.append(row)

            print(
                f" -> Step {step + 1:05d} | "
                f"depth_min={row['depth_min']:.3f} | "
                f"depth_mean={row['depth_mean']:.3f} | "
                f"pass={row['pass_rate']:.3f} | "
                f"advance={row['advanced_rate']:.3f} | "
                f"collision={row['collision_rate']:.3f} | "
                f"oob={row['out_of_bounds_rate']:.3f} | "
                f"success={row['success_rate']:.3f} | "
                f"target_idx={row['target_gate_idx_mean']:.2f}",
                flush=True,
            )

    elapsed = time.time() - t0
    fps = int(args_cli.steps) * n / max(elapsed, 1.0e-6)

    print_ok(f"random world rollout finished: {args_cli.steps} steps, fps={fps:,.2f} env world-steps/s")

    if records:
        last = records[-1]
        print_ok(
            "last stats: "
            f"depth_min={last['depth_min']:.3f}, "
            f"depth_mean={last['depth_mean']:.3f}, "
            f"collision={last['collision_rate']:.3f}, "
            f"target_idx={last['target_gate_idx_mean']:.2f}"
        )


def run_tests() -> None:
    heading("🚁 Quadrotor / Crazyflie Task4 Analytic Vision Gate-Racing World 全量测试启动")

    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    test_config()

    world = build_world()

    try:
        test_reset_shapes_and_ranges(world)
        test_gate_orientation_and_centerline(world)
        test_depth_basic(world)
        test_depth_known_gate_and_wall(world)
        test_gate_pass_and_progress(world)
        test_gate_collision_and_bounds(world)
        test_current_target_gate_features(world)
        random_world_rollout(world)

        heading("✅ Quadrotor / Crazyflie Task4 世界场景测试全部通过")

        print("World summary:")
        for k, v in world.summary().items():
            print(f"  {k}: {v}")

    except Exception as exc:
        print("\n❌ Quadrotor / Crazyflie Task4 世界场景测试失败：")
        print(type(exc).__name__, ":", exc)
        raise


if __name__ == "__main__":
    run_tests()
