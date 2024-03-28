"""Custom sequential class for augmentations."""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class CustomSequential:
    """Custom sequential class for augmentations."""

    x_transforms: list[Any] = field(default_factory=list)
    xy_transforms: list[Any] = field(default_factory=list)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the augmentations sequentially."""
        if self.x_transforms is not None:
            for transform in self.x_transforms:
                x = transform(x)
        if self.xy_transforms is not None:
            for transform in self.xy_transforms:
                x, y = transform(x, y)
        return x, y
