from __future__ import annotations

import torch


class RunningMeanStd:
    """Simple torch running mean/std utility.

    This is intentionally lightweight. It is useful for diagnostics and
    compatible with tensor batches shaped [N, D] or [N, ...].
    """

    def __init__(self, shape, device: str = "cpu", epsilon: float = 1.0e-4):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(float(epsilon), dtype=torch.float32, device=device)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = torch.as_tensor(x, dtype=torch.float32, device=self.mean.device)
        if x.numel() == 0:
            return

        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(x.shape[0]), dtype=torch.float32, device=self.mean.device)

        self.update_from_moments(batch_mean, batch_var, batch_count)

    @torch.no_grad()
    def update_from_moments(self, batch_mean: torch.Tensor, batch_var: torch.Tensor, batch_count: torch.Tensor) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / torch.clamp(total_count, min=1.0e-8)

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        correction = torch.square(delta) * self.count * batch_count / torch.clamp(total_count, min=1.0e-8)

        new_var = (m_a + m_b + correction) / torch.clamp(total_count, min=1.0e-8)

        self.mean = new_mean
        self.var = torch.clamp(new_var, min=1.0e-8)
        self.count = total_count

    def normalize(self, x: torch.Tensor, clip: float | None = None) -> torch.Tensor:
        y = (x - self.mean) / torch.sqrt(self.var + 1.0e-8)
        if clip is not None:
            y = torch.clamp(y, -float(clip), float(clip))
        return y

    def state_dict(self) -> dict:
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.mean = torch.as_tensor(state["mean"], dtype=torch.float32, device=self.mean.device)
        self.var = torch.as_tensor(state["var"], dtype=torch.float32, device=self.mean.device)
        self.count = torch.as_tensor(state["count"], dtype=torch.float32, device=self.mean.device)
