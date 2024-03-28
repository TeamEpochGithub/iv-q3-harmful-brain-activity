"""Custom sequential class for augmentations."""

from dataclasses import dataclass, field
import torch

@dataclass
class CustomSequential:

    x_transforms: list = field(default_factory=list)
    xy_transforms: list = field(default_factory=list)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply the augmentations sequentially."""
        for transform in self.x_transforms:
            x = transform(x)
        for transform in self.xy_transforms:
            x, y = transform(x, y)
        return x, y