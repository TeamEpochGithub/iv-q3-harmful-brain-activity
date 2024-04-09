"""2D MixUp implementation for spectrogram data augmentation."""
from dataclasses import dataclass, field

import torch
from kornia.augmentation._2d.mix.mixup import RandomMixUpV2 as KorniaMixUp  # type: ignore[import-not-found]


@dataclass
class MixUp:
    """2D MixUp implementation for spectrogram data augmentation.

    :param lambda_val: The range of the mixup coefficient
    :param same_on_batch: Apply the same transformation across the batch
    :param p: The probability of applying the filter
    """

    lambda_val: tuple[float, float] = field(default=(0.0, 1.0))
    same_on_batch: bool = False
    p: float = 0.5

    def __post_init__(self) -> None:
        """Check if the filter type is valid."""
        self.mixup = KorniaMixUp(p=self.p, lambda_val=self.lambda_val, same_on_batch=self.same_on_batch, data_keys=["input", "class"])

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Randomly patch the input with another sample."""
        dummy_labels = torch.arange(x.size(0))
        augmented_x, augmentation_info = self.mixup(x, dummy_labels)

        y = y.float()
        y_result = y.clone()
        for i in range(augmentation_info.shape[0]):
            y_result[i] = y[i] * (1 - augmentation_info[i, 2]) + y[int(augmentation_info[i, 1])] * augmentation_info[i, 2]

        return augmented_x, y_result
