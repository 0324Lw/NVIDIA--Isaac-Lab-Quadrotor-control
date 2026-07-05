from __future__ import annotations

import torch


def make_single_body_wrench_tensors(env_ids: torch.Tensor, force_w: torch.Tensor, torque_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    forces = torch.zeros((env_ids.numel(), 1, 3), dtype=torch.float32, device=force_w.device)
    torques = torch.zeros_like(forces)
    forces[:, 0, :] = force_w
    torques[:, 0, :] = torque_w
    return forces, torques
