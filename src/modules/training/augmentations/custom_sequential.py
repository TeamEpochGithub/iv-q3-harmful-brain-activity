"""Custom sequential class for augmentations."""

import torch

class CustomSequential():

    def __init__(self, x_transforms=[], xy_transforms=[]):
        """Initialize the custom sequential class."""
        super().__init__()
        self.x_transforms = x_transforms
        self.xy_transforms = xy_transforms

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply the augmentations sequentially."""
        for transform in self.x_transforms:
            x = transform(x)
        for transform in self.xy_transforms:
            x, y = transform(x, y)
        return x, y