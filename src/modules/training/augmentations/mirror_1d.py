"""Given a sequence with multiple channels, mirror it around its mean."""

from dataclasses import dataclass

import torch


@dataclass
class Mirror1D(torch.nn.Module):
    """Mirror augmentation for 1D signals."""

    p: float = 0.5

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the augmentation to the input signal."""
        augmented_x = x.clone()
        for i in range(x.shape[0]):
            if torch.rand(1) < self.p:
                augmented_x[i] = -1 * x[i] + 2 * x[i].mean(dim=-1).unsqueeze(-1)
        return augmented_x
