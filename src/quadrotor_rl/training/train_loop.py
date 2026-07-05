from __future__ import annotations

from typing import Callable


def run_fixed_step_loop(num_updates: int, update_fn: Callable[[int], None]) -> None:
    for update_id in range(int(num_updates)):
        update_fn(update_id)
