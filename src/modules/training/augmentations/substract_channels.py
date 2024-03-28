"""Substracts other channels to the current one."""
from dataclasses import dataclass

import torch


@dataclass
class SubstractChannels(torch.nn.Module):
    """Randomly substract other channels to the current one."""

    p: float = 0.5

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply substracting other channels to the input signal."""
        if torch.rand(1) < self.p:
            length = x.shape[1] - 1
            total = x.sum(dim=1) / length
            x = x - total + (x / length)
        return x
