from __future__ import annotations

from typing import Any, Callable


def create_task_env(env_cls: Callable[[Any], Any], cfg: Any):
    cfg.validate()
    return env_cls(cfg)
