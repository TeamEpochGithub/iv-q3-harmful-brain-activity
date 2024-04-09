"""Randomly shift the phase of all the frequencies frequencies."""
from dataclasses import dataclass

import torch


@dataclass
class RandomAmplitudeShift(torch.nn.Module):
    """Randomly shift the phase of all the frequencies frequencies."""

    low: float = 0.5
    high: float = 1.5
    p: float = 0.5

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Random phase shift to each frequency of the fft of the input signal."""
        if torch.rand(1) < self.p:
            # take the rfft of the input tensor
            x_freq = torch.fft.rfft(x, dim=-1)
            # create a random array of complex numbers each with a random pahse but with magnitude of 1
            random_amplitude = torch.rand(*x_freq.shape, device=x.device, dtype=x.dtype) * (self.high - self.low) + self.low
            # multiply the rfft with the random amplitude
            x_freq = x_freq * random_amplitude
            # take the irfft of the result
            return torch.fft.irfft(x_freq, dim=-1)
        return x
