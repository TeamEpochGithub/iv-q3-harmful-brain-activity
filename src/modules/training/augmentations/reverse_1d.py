"""Given a sequence with multiple channels, reverse it in time."""

import torch


class Reverse1D(torch.nn.Module):
    """Reverse augmentation for 1D signals."""

    def __init__(self, p: float = 0.5) -> None:
        """Initialize the augmentation."""
        super().__init__()
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the augmentation to the input signal."""
        augmented_x = x.clone()
        for i in range(x.shape[0]):
            if torch.rand(1) < self.p:
                augmented_x[i] = torch.flip(x[i], [-1])
        return augmented_x
