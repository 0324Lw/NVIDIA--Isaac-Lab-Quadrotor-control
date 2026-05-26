from __future__ import annotations

import argparse
import importlib
import math
import sys
from pathlib import Path
from typing import Any, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Quadrotor Crazyflie asset interface and external-wrench control smoke test"
)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--settle-steps", type=int, default=30)
parser.add_argument("--test-device", type=str, default="cuda:0")
parser.add_argument("--spawn-height", type=float, default=1.0)
parser.add_argument("--force-scale", type=float, default=1.35)
parser.add_argument("--yaw-torque", type=float, default=0.02)
parser.add_argument("--print-names", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

args_cli.device = args_cli.test_device
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

try:
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
except Exception:
    ISAAC_NUCLEUS_DIR = ""


def heading(title: str) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120, flush=True)


def print_ok(msg: str) -> None:
    print(f" ✅ {msg}", flush=True)


def print_warn(msg: str) -> None:
    print(f" ⚠️ {msg}", flush=True)


def quat_to_yaw_wxyz(q: torch.Tensor) -> torch.Tensor:
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def try_import_crazyflie_cfg() -> Tuple[Any | None, str]:
    candidates = [
        ("isaaclab_assets.robots.crazyflie", "CRAZYFLIE_CFG"),
        ("isaaclab_assets.robots.quadrotor", "CRAZYFLIE_CFG"),
        ("isaaclab_assets.robots.quadrotor", "QUADROTOR_CFG"),
    ]

    for module_name, attr_name in candidates:
        try:
            module = importlib.import_module(module_name)
            cfg = getattr(module, attr_name)
            cfg = cfg.replace(prim_path="{ENV_REGEX_NS}/Drone")

            try:
                cfg.init_state.pos = (0.0, 0.0, float(args_cli.spawn_height))
                cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
            except Exception:
                pass

            return cfg, f"{module_name}.{attr_name}"

        except Exception as exc:
            print_warn(f"cannot import {module_name}.{attr_name}: {type(exc).__name__}: {exc}")

    return None, ""


def fallback_usd_cfg() -> Tuple[ArticulationCfg, str]:
    if ISAAC_NUCLEUS_DIR:
        usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd"
    else:
        usd_path = "omniverse://localhost/NVIDIA/Assets/Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd"

    cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Drone",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, float(args_cli.spawn_height)),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "passive_rotors": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=1.0e6,
                velocity_limit_sim=1.0e6,
                stiffness=0.0,
                damping=0.0,
            )
        },
    )

    return cfg, f"fallback_usd:{usd_path}"


def build_drone_cfg() -> Tuple[Any, str]:
    cfg, source = try_import_crazyflie_cfg()
    if cfg is not None:
        return cfg, source

    print_warn("Built-in CRAZYFLIE_CFG not found. Falling back to USD path.")
    return fallback_usd_cfg()


DRONE_CFG, DRONE_SOURCE = build_drone_cfg()


@configclass
class QuadrotorSmokeSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )

    drone = DRONE_CFG


def get_drone(scene: InteractiveScene):
    try:
        return scene["drone"]
    except Exception:
        return scene.articulations["drone"]


def estimate_mass(drone, device: str) -> torch.Tensor:
    num_envs = int(args_cli.num_envs)

    try:
        masses = drone.root_physx_view.get_masses()
        masses = torch.as_tensor(masses, dtype=torch.float32, device=device)

        if masses.ndim == 2:
            total = masses.sum(dim=-1)
        else:
            total = masses.reshape(num_envs, -1).sum(dim=-1)

        if torch.isfinite(total).all().item() and total.mean().item() > 1.0e-5:
            return total

    except Exception as exc:
        print_warn(f"mass from root_physx_view failed: {type(exc).__name__}: {exc}")

    for attr in ["default_mass", "body_mass"]:
        try:
            masses = getattr(drone.data, attr)
            masses = torch.as_tensor(masses, dtype=torch.float32, device=device)

            if masses.ndim == 2:
                total = masses.sum(dim=-1)
            else:
                total = masses.reshape(num_envs, -1).sum(dim=-1)

            if torch.isfinite(total).all().item() and total.mean().item() > 1.0e-5:
                return total

        except Exception:
            pass

    print_warn("cannot read mass, fallback to Crazyflie-like 0.030 kg")
    return torch.full((num_envs,), 0.030, dtype=torch.float32, device=device)


def yaw_to_quat_wxyz(yaw: torch.Tensor) -> torch.Tensor:
    quat = torch.zeros((yaw.shape[0], 4), dtype=torch.float32, device=yaw.device)
    quat[:, 0] = torch.cos(0.5 * yaw)
    quat[:, 3] = torch.sin(0.5 * yaw)
    return quat


def write_root_state(drone, scene: InteractiveScene, device: str) -> None:
    env_ids = torch.arange(args_cli.num_envs, dtype=torch.long, device=device)

    root_state = drone.data.default_root_state[env_ids].clone()
    root_state[:, :3] = scene.env_origins[env_ids]
    root_state[:, 2] += float(args_cli.spawn_height)
    root_state[:, 3:7] = yaw_to_quat_wxyz(torch.zeros(args_cli.num_envs, dtype=torch.float32, device=device))
    root_state[:, 7:13] = 0.0

    drone.write_root_state_to_sim(root_state, env_ids=env_ids)

    try:
        joint_pos = drone.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(drone.data.default_joint_vel[env_ids])
        drone.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    except Exception as exc:
        print_warn(f"write_joint_state_to_sim skipped: {type(exc).__name__}: {exc}")

    try:
        drone.reset(env_ids)
    except Exception:
        pass


def set_external_wrench(
    drone,
    device: str,
    force_z: torch.Tensor | float,
    torque_z: torch.Tensor | float = 0.0,
) -> None:
    num_envs = int(args_cli.num_envs)

    force_z_t = torch.as_tensor(force_z, dtype=torch.float32, device=device).reshape(num_envs)

    torque_z_t = torch.as_tensor(torque_z, dtype=torch.float32, device=device)
    if torque_z_t.numel() == 1:
        torque_z_t = torque_z_t.repeat(num_envs)
    torque_z_t = torque_z_t.reshape(num_envs)

    forces = torch.zeros((num_envs, 1, 3), dtype=torch.float32, device=device)
    torques = torch.zeros((num_envs, 1, 3), dtype=torch.float32, device=device)
    forces[:, 0, 2] = force_z_t
    torques[:, 0, 2] = torque_z_t

    try:
        drone.set_external_force_and_torque(
            forces=forces,
            torques=torques,
            body_ids=[0],
            is_global=True,
        )
        return
    except TypeError:
        pass
    except Exception as exc:
        print_warn(f"root-body wrench with is_global failed: {type(exc).__name__}: {exc}")

    try:
        drone.set_external_force_and_torque(
            forces=forces,
            torques=torques,
            body_ids=[0],
        )
        return
    except Exception as exc:
        print_warn(f"root-body wrench failed: {type(exc).__name__}: {exc}")

    num_bodies = int(getattr(drone, "num_bodies", 1))
    forces_full = torch.zeros((num_envs, num_bodies, 3), dtype=torch.float32, device=device)
    torques_full = torch.zeros((num_envs, num_bodies, 3), dtype=torch.float32, device=device)
    forces_full[:, 0, 2] = force_z_t
    torques_full[:, 0, 2] = torque_z_t

    drone.set_external_force_and_torque(forces_full, torques_full)


def root_pos(drone) -> torch.Tensor:
    return drone.data.root_pos_w.clone()


def root_quat(drone) -> torch.Tensor:
    return drone.data.root_quat_w.clone()


def root_lin_vel(drone) -> torch.Tensor:
    if hasattr(drone.data, "root_lin_vel_w"):
        return drone.data.root_lin_vel_w.clone()
    return drone.data.root_lin_vel_b.clone()


def root_ang_vel(drone) -> torch.Tensor:
    if hasattr(drone.data, "root_ang_vel_w"):
        return drone.data.root_ang_vel_w.clone()
    return drone.data.root_ang_vel_b.clone()


def step_sim(
    sim,
    scene: InteractiveScene,
    drone,
    device: str,
    force_z: torch.Tensor | float,
    torque_z: torch.Tensor | float = 0.0,
    steps: int = 1,
) -> None:
    for _ in range(int(steps)):
        set_external_wrench(drone, device, force_z=force_z, torque_z=torque_z)

        try:
            drone.write_data_to_sim()
        except Exception:
            pass

        scene.write_data_to_sim()
        sim.step()
        scene.update(float(sim.get_physics_dt()))


def main() -> None:
    heading("Quadrotor Crazyflie asset interface and external-wrench control smoke test")

    device = str(args_cli.test_device)

    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,
        device=device,
        physx=sim_utils.PhysxCfg(
            enable_external_forces_every_iteration=True,
            min_position_iteration_count=4,
            max_position_iteration_count=8,
            min_velocity_iteration_count=1,
            max_velocity_iteration_count=2,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = QuadrotorSmokeSceneCfg(
        num_envs=int(args_cli.num_envs),
        env_spacing=2.0,
    )
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.update(0.0)

    drone = get_drone(scene)

    heading("[测试 1] 模型资产 / articulation 接口信息")

    print_ok(f"asset source = {DRONE_SOURCE}")
    print_ok(f"num_envs = {args_cli.num_envs}")
    print_ok(f"device = {device}")
    print_ok(f"num_bodies = {getattr(drone, 'num_bodies', '<unknown>')}")
    print_ok(f"num_joints = {getattr(drone, 'num_joints', '<unknown>')}")

    body_names = list(getattr(drone, "body_names", []))
    joint_names = list(getattr(drone, "joint_names", []))

    if args_cli.print_names:
        print("\nbody_names:")
        for i, name in enumerate(body_names):
            print(f"  body[{i:02d}] = {name}")

        print("\njoint_names:")
        for i, name in enumerate(joint_names):
            print(f"  joint[{i:02d}] = {name}")

    assert getattr(drone, "num_bodies", 0) >= 1, "Drone must have at least one body"
    assert torch.isfinite(drone.data.root_pos_w).all().item(), "root_pos_w has NaN/Inf"
    assert torch.isfinite(drone.data.root_quat_w).all().item(), "root_quat_w has NaN/Inf"

    mass = estimate_mass(drone, device)
    weight = mass * 9.81

    print_ok(f"estimated mass mean = {mass.mean().item():.6f} kg")
    print_ok(f"estimated weight mean = {weight.mean().item():.6f} N")

    heading("[测试 2] reset root state / settle finite 检测")

    write_root_state(drone, scene, device)
    scene.write_data_to_sim()
    scene.update(0.0)

    z0 = root_pos(drone)[:, 2].clone()

    step_sim(
        sim=sim,
        scene=scene,
        drone=drone,
        device=device,
        force_z=torch.zeros_like(weight),
        torque_z=0.0,
        steps=int(args_cli.settle_steps),
    )

    z_settle = root_pos(drone)[:, 2].clone()

    assert torch.isfinite(root_pos(drone)).all().item()
    assert torch.isfinite(root_quat(drone)).all().item()
    assert torch.isfinite(root_lin_vel(drone)).all().item()
    assert torch.isfinite(root_ang_vel(drone)).all().item()

    print_ok(f"z initial mean = {z0.mean().item():.6f}")
    print_ok(f"z after settle mean = {z_settle.mean().item():.6f}")
    print_ok("settle finite check passed")

    heading("[测试 3] upward force control 检测")

    write_root_state(drone, scene, device)
    scene.write_data_to_sim()
    scene.update(0.0)

    z_before = root_pos(drone)[:, 2].clone()
    vz_before = root_lin_vel(drone)[:, 2].clone()

    up_force = weight * float(args_cli.force_scale)

    step_sim(
        sim=sim,
        scene=scene,
        drone=drone,
        device=device,
        force_z=up_force,
        torque_z=0.0,
        steps=int(args_cli.steps),
    )

    z_after = root_pos(drone)[:, 2].clone()
    vz_after = root_lin_vel(drone)[:, 2].clone()

    dz = z_after - z_before
    dvz = vz_after - vz_before

    print_ok(f"up force mean = {up_force.mean().item():.6f} N")
    print_ok(f"z before mean = {z_before.mean().item():.6f}")
    print_ok(f"z after mean = {z_after.mean().item():.6f}")
    print_ok(f"delta z mean = {dz.mean().item():+.6f}")
    print_ok(f"delta vz mean = {dvz.mean().item():+.6f}")

    assert torch.isfinite(z_after).all().item()
    assert torch.isfinite(vz_after).all().item()
    assert dz.mean().item() > -0.02 or dvz.mean().item() > 0.02, (
        "upward force did not produce valid lift response. "
        "Check asset mass / external force API."
    )

    heading("[测试 4] yaw torque control 检测")

    write_root_state(drone, scene, device)
    scene.write_data_to_sim()
    scene.update(0.0)

    yaw_before = quat_to_yaw_wxyz(root_quat(drone)).clone()
    yaw_rate_before = root_ang_vel(drone)[:, 2].clone()

    hover_force = weight
    yaw_torque = torch.full_like(weight, float(args_cli.yaw_torque))

    step_sim(
        sim=sim,
        scene=scene,
        drone=drone,
        device=device,
        force_z=hover_force,
        torque_z=yaw_torque,
        steps=int(args_cli.steps),
    )

    yaw_after = quat_to_yaw_wxyz(root_quat(drone)).clone()
    yaw_rate_after = root_ang_vel(drone)[:, 2].clone()

    dyaw = torch.atan2(
        torch.sin(yaw_after - yaw_before),
        torch.cos(yaw_after - yaw_before),
    )
    dyaw_rate = yaw_rate_after - yaw_rate_before

    print_ok(f"yaw torque mean = {yaw_torque.mean().item():.6f} N*m")
    print_ok(f"delta yaw mean = {dyaw.mean().item():+.6f} rad")
    print_ok(f"delta yaw rate mean = {dyaw_rate.mean().item():+.6f} rad/s")

    assert torch.isfinite(yaw_after).all().item()
    assert torch.isfinite(yaw_rate_after).all().item()
    assert abs(dyaw.mean().item()) > 1.0e-4 or abs(dyaw_rate.mean().item()) > 1.0e-4, (
        "yaw torque did not produce yaw response. "
        "Check external torque API / root body id."
    )

    heading("✅ Quadrotor Crazyflie asset interface and control smoke test passed")

    print("下一步可以创建 Ubuntu 启动脚本，再运行本测试。")


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
