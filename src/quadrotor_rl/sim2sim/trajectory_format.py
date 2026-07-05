from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


REQUIRED_ROLLOUT_KEYS = ("obs", "action")


def save_rollout_npz(path: str | Path, **arrays) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_rollout_npz(path: str | Path) -> Dict[str, np.ndarray]:
    data = np.load(Path(path), allow_pickle=False)
    out = {key: data[key] for key in data.files}
    for key in REQUIRED_ROLLOUT_KEYS:
        if key not in out:
            raise RuntimeError(f"rollout is missing required key: {key}")
    return out
