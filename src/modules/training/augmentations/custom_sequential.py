"""Custom sequential class for augmentations."""

from dataclasses import dataclass, field
from inspect import signature
from typing import Any

import torch


@dataclass
class CustomSequential:
    """Custom sequential class for augmentations."""

    transforms: list[Any] = field(default_factory=list)
    keep_order: bool = True
    rand_apply: int | None = None

    def __post_init__(self) -> None:
        """Check if the filter type is valid."""

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the augmentations sequentially."""
        # Randomly choose the augmentations
        if self.rand_apply is not None:
            chosen_aug = torch.randperm(len(self.transforms))[: self.rand_apply]
            if self.keep_order:
                chosen_aug_list = sorted(chosen_aug)
        else:
            chosen_aug_list = list(range(len(self.transforms)))

        # Apply the augmentations
        for i in chosen_aug_list:
            transform = self.transforms[i]

            # Check if the transform is for x or xy
            if len(signature(transform).parameters) == 1:
                x = transform(x)
            else:
                x, y = transform(x, y)
        return x, y
