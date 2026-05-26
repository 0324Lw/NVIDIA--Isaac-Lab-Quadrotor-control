from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ProgressMeter:
    total_steps: int
    start_steps: int = 0
    start_time: float = field(default_factory=time.time)

    def elapsed(self) -> float:
        return max(time.time() - self.start_time, 1.0e-9)

    def fraction(self, current_steps: int) -> float:
        denom = max(int(self.total_steps) - int(self.start_steps), 1)
        return max(0.0, min(1.0, (int(current_steps) - int(self.start_steps)) / denom))

    def percent(self, current_steps: int) -> float:
        return 100.0 * self.fraction(current_steps)

    def fps(self, current_steps: int) -> float:
        done = max(int(current_steps) - int(self.start_steps), 0)
        return float(done) / self.elapsed()

    def eta_seconds(self, current_steps: int) -> float:
        fps = self.fps(current_steps)
        if fps <= 1.0e-9:
            return float("inf")
        remain = max(int(self.total_steps) - int(current_steps), 0)
        return float(remain) / fps

    def summary(self, current_steps: int) -> dict[str, float]:
        return {
            "current_steps": float(current_steps),
            "total_steps": float(self.total_steps),
            "progress_percent": self.percent(current_steps),
            "fps": self.fps(current_steps),
            "eta_seconds": self.eta_seconds(current_steps),
            "elapsed_seconds": self.elapsed(),
        }
