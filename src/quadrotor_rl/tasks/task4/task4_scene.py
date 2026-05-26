from __future__ import annotations

import importlib
from typing import Any, Tuple

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from quadrotor_rl.tasks.task4.task4_config import Task4Config


_LAST_ASSET_SOURCE = "unknown"


def get_quadrotor_task4_asset_source() -> str:
    return _LAST_ASSET_SOURCE


def _try_import_crazyflie_cfg(cfg: Task4Config) -> Tuple[Any | None, str]:
    candidates = [
        ("isaaclab_assets.robots.crazyflie", "CRAZYFLIE_CFG"),
        ("isaaclab_assets.robots.quadrotor", "CRAZYFLIE_CFG"),
        ("isaaclab_assets.robots.quadrotor", "QUADROTOR_CFG"),
    ]

    for module_name, attr_name in candidates:
        try:
            module = importlib.import_module(module_name)
            drone_cfg = getattr(module, attr_name)
            drone_cfg = drone_cfg.replace(prim_path="{ENV_REGEX_NS}/Drone")

            try:
                drone_cfg.init_state.pos = tuple(float(x) for x in cfg.start_pos)
                drone_cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
            except Exception:
                pass

            return drone_cfg, f"{module_name}.{attr_name}"

        except Exception:
            continue

    return None, ""


def make_fallback_crazyflie_cfg(cfg: Task4Config) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Drone",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(cfg.crazyflie_usd_url),
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
            pos=tuple(float(x) for x in cfg.start_pos),
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


def build_quadrotor_asset_cfg(cfg: Task4Config) -> Tuple[Any, str]:
    drone_cfg, source = _try_import_crazyflie_cfg(cfg)
    if drone_cfg is not None:
        return drone_cfg, source

    return make_fallback_crazyflie_cfg(cfg), f"fallback_usd:{cfg.crazyflie_usd_url}"


def make_quadrotor_task4_scene_cfg(task_cfg: Task4Config | None = None):
    """Build Task4 InteractiveSceneCfg without launching Isaac Sim."""

    global _LAST_ASSET_SOURCE

    if task_cfg is None:
        task_cfg = Task4Config()

    task_cfg.validate()

    drone_cfg, source = build_quadrotor_asset_cfg(task_cfg)
    _LAST_ASSET_SOURCE = source

    @configclass
    class QuadrotorTask4SceneCfg(InteractiveSceneCfg):
        num_envs: int = int(task_cfg.num_envs)
        env_spacing: float = float(task_cfg.env_spacing)

        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
        )

        light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0),
        )

        drone = drone_cfg

    return QuadrotorTask4SceneCfg


QuadrotorTask4SceneCfgFactory = make_quadrotor_task4_scene_cfg
CrazyflieTask4SceneCfgFactory = make_quadrotor_task4_scene_cfg
