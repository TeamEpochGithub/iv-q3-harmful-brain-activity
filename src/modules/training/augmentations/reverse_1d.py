"""Given a sequence with multiple channels, reverse it in time."""

import torch
from dataclasses import dataclass


@dataclass
class Reverse1D(torch.nn.Module):
    """Reverse augmentation for 1D signals."""

    p: float = 0.5

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the augmentation to the input signal."""
        augmented_x = x.clone()
        for i in range(x.shape[0]):
            if torch.rand(1) < self.p:
                augmented_x[i] = torch.flip(x[i], [-1])
        return augmented_x
