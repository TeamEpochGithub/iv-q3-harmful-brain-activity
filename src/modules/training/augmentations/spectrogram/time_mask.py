"""Randomly mask certain frequencies."""
from dataclasses import dataclass

import torch
from torchaudio.transforms import TimeMasking


@dataclass
class TimeMask:
    """Randomly mask certain timesteps.

    :param freq_mask_param: Maximum number of timesteps to mask
    :param apply_x_times: The number of times to apply the filter
    :param iid_masks: Whether to apply the same mask to all channels
    :param p: The probability of applying the filter
    """

    time_mask_param: int
    apply_x_times: int = 1
    iid_masks: bool = True
    p: float = 0.5

    def __post_init__(self) -> None:
        """Check if the filter type is valid."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly mask certain timesteps."""
        # Skip augmentation with probability 1-p
        if torch.rand(1) > self.p:
            return x

        for _ in range(self.apply_x_times):
            x = TimeMasking(time_mask_param=self.time_mask_param, iid_masks=self.iid_masks)(x)

        return x
