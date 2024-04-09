"""Custom sequential class for augmentations."""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class CustomApplyOne:
    """Custom sequential class for augmentations."""

    x_transforms: list[Any] = field(default_factory=list)
    xy_transforms: list[Any] = field(default_factory=list)

    def __post_init__(self):
        self.probabilities = []
        if self.x_transforms is not None:
            for transform in self.x_transforms:
                self.probabilities.append(transform.p)
        if self.xy_transforms is not None:
            for transform in self.xy_transforms:
                self.probabilities.append(transform.p)

        # Make tensor from probs
        self.probabilities = torch.tensor(self.probabilities)
        # Ensure sum is 1
        self.probabilities /= self.probabilities.sum()
        self.all_transforms = self.x_transforms + self.xy_transforms

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the augmentations sequentially."""
        transform = self.all_transforms[torch.multinomial(self.probabilities, 1, replacement=False).item()]
        if transform in self.x_transforms:
            x = transform(x)
        if transform in self.xy_transforms:
            x, y = transform(x, y)
        if torch.any(torch.isnan(y)):
            print("nan")
        return x, y


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    probs = [0.1] * 5 + [0.25] * 2
    probs = torch.tensor(probs)
    outs = []
    sample_counts = torch.multinomial(probs.float(), 10000, replacement=True).int()
    sample_counts_np = sample_counts.numpy()
    plt.hist(sample_counts_np, bins=range(len(probs) + 1))
    plt.xlabel("Sample Count")
    plt.ylabel("Frequency")
    plt.title("Histogram of Sample Counts")
    plt.show()
