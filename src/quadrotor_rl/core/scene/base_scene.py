from __future__ import annotations

from typing import Any


def get_articulation_from_scene(scene: Any, name: str = "drone") -> Any:
    try:
        return scene[name]
    except Exception:
        return scene.articulations[name]
