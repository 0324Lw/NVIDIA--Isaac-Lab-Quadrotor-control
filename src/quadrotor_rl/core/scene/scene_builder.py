from __future__ import annotations

from typing import Any, Callable

from isaaclab.scene import InteractiveScene


def build_interactive_scene(scene_cfg_factory: Callable[[Any], Any], cfg: Any) -> InteractiveScene:
    SceneCfg = scene_cfg_factory(cfg)
    scene_cfg = SceneCfg(num_envs=int(cfg.num_envs), env_spacing=float(cfg.env_spacing))
    return InteractiveScene(scene_cfg)
