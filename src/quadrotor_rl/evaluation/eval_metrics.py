from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class QuadrotorEvalMetrics:
    success_rate: float = 0.0
    crash_rate: float = 0.0
    timeout_rate: float = 0.0
    mean_episode_length: float = 0.0
    mean_episode_return: float = 0.0
    mean_action_abs: float = 0.0
    motor_saturation_rate: float = 0.0
    max_roll_pitch: float = 0.0
    max_angular_velocity: float = 0.0
    position_error: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}
