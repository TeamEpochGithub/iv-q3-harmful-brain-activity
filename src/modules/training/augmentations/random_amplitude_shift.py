"""Randomly shift the phase of all the frequencies frequencies."""
from dataclasses import dataclass

import numpy as np
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

# Example usage
if __name__ == "__main__":
    import numpy as np
    # read a single eeg sequence
    import pandas as pd
    eeg = pd.read_parquet('C:\\Users\\Tolga\\Desktop\\EPOCH-IV\\q3-harmful-brain-activity\\data\\raw\\train_eegs\\4244347802.parquet') 
    # get 1 channel and make it numpy
    signal = eeg.values.transpose(1,0)[np.newaxis, :]
    print(signal.shape)
    filter = RandomAmplitudeShift(p=1)
    for i in range(1):
        torch_result = filter(torch.from_numpy(signal).to('cuda'))
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 10))

        # Plot original signal
        plt.subplot(3, 2, 1)
        plt.plot(signal[0].transpose(1,0))
        plt.title('Original signal')

        # Plot original signal fft
        freqs = np.fft.rfftfreq(signal.shape[-1], 1/200)
        plt.subplot(3, 2, 2)
        plt.plot(freqs, np.abs(np.fft.rfft(signal))[0].transpose(1,0))
        plt.title('Amplitude Spectrum')

        # Plot low pass filter
        plt.subplot(3, 2, 3)
        plt.plot(torch_result.cpu().numpy()[0].transpose(1,0))
        plt.title('Random Amplitude Shifted signal')

        # Plot low pass filter fft

        plt.subplot(3, 2, 4)
        plt.plot(freqs, np.abs(np.fft.rfft(torch_result.cpu().numpy()))[0].transpose(1,0))
        plt.title('Amplitude Spectrum')

        # Compute the phases for the original signal
        phases = np.angle(np.fft.rfft(signal[0])).transpose(1,0)

        # Plot the phases
        plt.subplot(3, 2, 5)
        plt.plot(freqs, phases)
        plt.title('Phase Spectrum of original signal')
        
        # Compute the phases of the torch result
        phases = np.angle(np.fft.rfft(torch_result.cpu().numpy())[0]).transpose(1,0)

        plt.subplot(3, 2, 6)
        plt.plot(freqs, phases)
        plt.title('Phase Spectrum of torch result')

        # Adjust the spacing between the subplots
        plt.tight_layout()

    plt.show()
