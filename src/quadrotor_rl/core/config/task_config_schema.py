from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class TaskObservationSchema:
    task_name: str
    actor_obs_dim: int
    critic_obs_dim: int
    action_dim: int
    frame_stack: int
    single_obs_dim: int | None = None
    depth_dim: int | None = None
    compact_state_dim: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "actor_obs_dim": int(self.actor_obs_dim),
            "critic_obs_dim": int(self.critic_obs_dim),
            "action_dim": int(self.action_dim),
            "frame_stack": int(self.frame_stack),
            "single_obs_dim": None if self.single_obs_dim is None else int(self.single_obs_dim),
            "depth_dim": None if self.depth_dim is None else int(self.depth_dim),
            "compact_state_dim": None if self.compact_state_dim is None else int(self.compact_state_dim),
        }


def build_task_observation_schema(task_name: str, cfg: Any) -> TaskObservationSchema:
    return TaskObservationSchema(
        task_name=str(task_name),
        actor_obs_dim=int(cfg.actor_obs_dim),
        critic_obs_dim=int(cfg.critic_obs_dim),
        action_dim=int(cfg.action_dim),
        frame_stack=int(getattr(cfg, "frame_stack", 1)),
        single_obs_dim=getattr(cfg, "single_actor_obs_dim", getattr(cfg, "obs_dim_per_frame", None)),
        depth_dim=getattr(cfg, "depth_dim", None),
        compact_state_dim=getattr(cfg, "compact_state_dim", None),
    )
