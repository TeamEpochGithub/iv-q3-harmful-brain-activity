"""Custom sequential class for augmentations."""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class CustomApplyOne:
    """Custom sequential class for augmentations."""

    probabilities_tensor: torch.Tensor
    x_transforms: list[Any] = field(default_factory=list)
    xy_transforms: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Post initialization function of CustomApplyOne."""
        self.probabilities = []
        if self.x_transforms is not None:
            for transform in self.x_transforms:
                self.probabilities.append(transform.p)
        if self.xy_transforms is not None:
            for transform in self.xy_transforms:
                self.probabilities.append(transform.p)

        # Make tensor from probs
        self.probabilities_tensor = torch.tensor(self.probabilities_tensor)
        # Ensure sum is 1
        self.probabilities_tensor /= self.probabilities_tensor.sum()
        self.all_transforms = self.x_transforms + self.xy_transforms

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the augmentations sequentially."""
        transform = self.all_transforms[int(torch.multinomial(self.probabilities_tensor, 1, replacement=False).item())]
        if transform in self.x_transforms:
            x = transform(x)
        if transform in self.xy_transforms:
            x, y = transform(x, y)
        return x, y
