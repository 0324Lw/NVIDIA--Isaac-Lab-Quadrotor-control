from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


class RolloutRecorder:
    def __init__(self):
        self.frames: List[Dict[str, np.ndarray]] = []

    def append(self, **kwargs) -> None:
        frame: Dict[str, np.ndarray] = {}
        for key, value in kwargs.items():
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            frame[key] = np.asarray(value)
        self.frames.append(frame)

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.frames:
            np.savez_compressed(path, empty=np.array([1], dtype=np.int32))
            return
        keys = sorted(self.frames[0].keys())
        data = {key: np.stack([frame[key] for frame in self.frames], axis=0) for key in keys}
        np.savez_compressed(path, **data)
