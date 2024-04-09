"""Randomly mask certain frequencies."""
from dataclasses import dataclass

import torch
from torchaudio.transforms import FrequencyMasking


@dataclass
class FrequencyMask:
    """Randomly mask certain frequencies.

    :param freq_mask_param: The number of frequency channels to mask
    :param apply_x_times: The number of times to apply the filter
    :param iid_masks: Whether to apply the same mask to all channels
    :param p: The probability of applying the filter
    """

    freq_mask_param: int
    apply_x_times: int = 1
    iid_masks: bool = True
    p: float = 0.5

    def __post_init__(self) -> None:
        """Check if the filter type is valid."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly mask certain frequencies."""
        # Skip augmentation with probability 1-p
        if torch.rand(1) > self.p:
            return x

        for _ in range(self.apply_x_times):
            x = FrequencyMasking(freq_mask_param=self.freq_mask_param, iid_masks=self.iid_masks)(x)

        return x
