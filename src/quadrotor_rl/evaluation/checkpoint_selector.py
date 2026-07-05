from __future__ import annotations

from pathlib import Path
from typing import Iterable

DEFAULT_CHECKPOINT_NAMES = (
    "quadrotor_task4_skrl_model.pt",
    "quadrotor_task4_model.pt",
    "quadrotor_task3_skrl_model.pt",
    "quadrotor_task3_model.pt",
    "quadrotor_task2_skrl_model.pt",
    "quadrotor_task2_model.pt",
    "quadrotor_task1_skrl_model.pt",
    "quadrotor_task1_model.pt",
    "agent.pt",
    "model.pt",
    "checkpoint.pt",
)


def resolve_checkpoint_path(path: str | Path, preferred_names: Iterable[str] = ()) -> Path:
    p = Path(path).expanduser().resolve()
    if p.is_file():
        return p
    if p.is_dir():
        for name in tuple(preferred_names) + DEFAULT_CHECKPOINT_NAMES:
            candidate = p / name
            if candidate.exists():
                return candidate
        final_dir = p / "final_checkpoint"
        if final_dir.exists():
            return resolve_checkpoint_path(final_dir, preferred_names=preferred_names)
    return p
