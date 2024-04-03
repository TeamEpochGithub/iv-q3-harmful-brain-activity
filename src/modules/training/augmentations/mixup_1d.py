"""Mixup augmentation for 1d signals."""
import torch
from dataclasses import dataclass
import torch


@dataclass
class MixUp1D(torch.nn.Module):
    """CutMix augmentation for 1D signals."""

    p: float = 0.5

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the augmentation to the input signal."""
        indices = torch.arange(x.shape[0], device=x.device, dtype=torch.int)
        shuffled_indices = torch.randperm(indices.shape[0])

        augmented_x = x.clone()
        augmented_y = y.clone().float()
        for i in range(x.shape[0]):
            if torch.rand(1) < self.p:
                lambda_ = torch.rand(1, device=x.device)
                augmented_x[i] = lambda_ * x[i] + (1 - lambda_) * x[shuffled_indices[i]]
                augmented_y[i] = lambda_ * y[i] + (1 - lambda_) * y[shuffled_indices[i]]
        return augmented_x, augmented_y
