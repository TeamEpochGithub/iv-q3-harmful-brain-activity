"""Mixup augmentation for 1d signals."""
from dataclasses import dataclass

import torch


@dataclass
class NoOp(torch.nn.Module):
    """CutMix augmentation for 1D signals."""

    p: float = 0.5

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the augmentation to the input signal."""
        return x