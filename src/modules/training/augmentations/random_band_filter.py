"""Randomly pass or block a band of frequencies."""
from dataclasses import dataclass

import scipy
import torch
from torchaudio.functional import convolve


@dataclass
class RandomBandFilter:
    """Randomly pass a band of frequencies."""

    low_cutoff_range: list[float]
    high_cutoff_range: list[float]
    filter_type: str = "bandpass"
    p: float = 0.5

    def __post_init__(self) -> None:
        """Check if the filter type is valid."""
        if self.filter_type not in ["bandpass", "bandstop"]:
            raise ValueError(f"Invalid filter type: {self.filter_type}. Should be one of ['bandpass', 'bandstop'].")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a bandpass filter with random pass-band to the input signal."""
        if torch.rand(1) < self.p:
            # Randomly get cutoffs from the given ranges
            self.high_cutoff = torch.rand(1) * (self.high_cutoff_range[1] - self.high_cutoff_range[0]) + self.high_cutoff_range[0]
            self.low_cutoff = torch.rand(1) * (self.low_cutoff_range[1] - self.low_cutoff_range[0]) + self.low_cutoff_range[0]
            # Create a bandpass filter kernel
            band_win = scipy.signal.firwin(301, [self.low_cutoff.item(), self.high_cutoff.item()], window="hamming", fs=200, pass_zero=self.filter_type).astype("float32")
            self.band_win = torch.from_numpy(band_win).unsqueeze(0).unsqueeze(0)

            # Convolve the signal with the kernel
            if len(x.shape) == 2:
                x = x.unsqueeze(0)

            return convolve(x, self.band_win.to(x.device), mode="same")
        return x
