from __future__ import annotations

import torch
import torch.nn as nn


def orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class MLPBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(256, 256, 128), activation=nn.ELU):
        super().__init__()

        layers = []
        last = int(in_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(last, int(h)))
            layers.append(activation())
            last = int(h)

        self.net = nn.Sequential(*layers)
        self.output_dim = last
        self.apply(orthogonal_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        x = torch.clamp(x, -10.0, 10.0)
        return self.net(x)


class Depth64Encoder(nn.Module):
    """Small CNN encoder for 1x64x64 depth observations."""

    def __init__(self, output_dim: int = 256):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            flat = int(self.cnn(torch.zeros(1, 1, 64, 64)).shape[1])

        self.head = nn.Sequential(
            nn.Linear(flat, int(output_dim)),
            nn.ELU(),
        )
        self.output_dim = int(output_dim)
        self.apply(orthogonal_init)

    def forward(self, depth_flat: torch.Tensor) -> torch.Tensor:
        depth = depth_flat.reshape(-1, 1, 64, 64)
        depth = torch.clamp(torch.nan_to_num(depth, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        return self.head(self.cnn(depth))


def clamp_log_std(log_std: torch.Tensor, min_log_std: float = -3.0, max_log_std: float = 0.5) -> torch.Tensor:
    return torch.clamp(
        torch.nan_to_num(log_std, nan=float(min_log_std), posinf=float(max_log_std), neginf=float(min_log_std)),
        float(min_log_std),
        float(max_log_std),
    )


# Go2-style generic alias for downstream scripts.
SkrlMLPBackbone = MLPBackbone
SkrlDepth64Encoder = Depth64Encoder
