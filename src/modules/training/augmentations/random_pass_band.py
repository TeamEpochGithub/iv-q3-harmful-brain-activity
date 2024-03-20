from dataclasses import dataclass
import torch
import scipy
import torch.nn.functional as F
from torchaudio.functional import convolve

@dataclass
class RandomBandFilter():
    """Randomly pass a band of frequencies."""
    low_cutoff_range: list[float]
    high_cutoff_range: list[float]
    filter_type: str = 'bandpass'

    def __post_init__(self):
        if self.filter_type not in ['bandpass', 'bandstop']:
            raise ValueError(f"Invalid filter type: {self.filter_type}. Should be one of ['bandpass', 'bandstop'].")
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a bandpass filter with random pass-band to the input signal."""
        # Randomly get cutoffs from the given ranges
        self.high_cutoff = torch.rand(1) * (self.high_cutoff_range[1] - self.high_cutoff_range[0]) + self.high_cutoff_range[0]
        self.low_cutoff = torch.rand(1) * (self.low_cutoff_range[1] - self.low_cutoff_range[0]) + self.low_cutoff_range[0]
        # Create a bandpass filter kernel
        self.band_win = torch.from_numpy(scipy.signal.firwin(301, [self.low_cutoff.item(), self.high_cutoff.item()], window='hamming', fs=200, pass_zero=self.filter_type).astype('float32')).unsqueeze(0).unsqueeze(0)
        # Convolve the signal with the kernel
        band_pass = convolve(x, self.band_win.to(x.device), mode='same')
        return band_pass


# Example usage
if __name__ == "__main__":
    import numpy as np
    # read a single eeg sequence
    import pandas as pd
    eeg = pd.read_parquet('C:\\Users\\Tolga\\Desktop\\EPOCH-IV\\q3-harmful-brain-activity\\data\\raw\\train_eegs\\687656140.parquet') 
    # get 1 channel and make it numpy
    signal = eeg['F7'].values
    filter = RandomBandFilter(low_cutoff_range=[0,1], high_cutoff_range=[6, 10], filter_type='bandstop')
    for i in range(2):
        torch_result = filter(torch.from_numpy(signal).to('cuda').unsqueeze(0).unsqueeze(0))
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 10))

        # Plot original signal
        plt.subplot(2, 2, 1)
        plt.plot(signal)
        plt.title('Original signal')

        # Plot original signal fft
        freqs = np.fft.rfftfreq(len(signal), 1/200)
        plt.subplot(2, 2, 2)
        plt.plot(freqs, np.abs(np.fft.rfft(signal)))
        plt.title('Original signal fft')

        # Plot low pass filter
        plt.subplot(2, 2, 3)
        plt.plot(torch_result[0][0].cpu().numpy(), 'r-')
        plt.title('Low pass filter')

        # Plot low pass filter fft
        freqs = np.fft.rfftfreq(len(signal), 1/200)
        plt.subplot(2, 2, 4)
        plt.plot(freqs, np.abs(np.fft.rfft(torch_result[0][0].cpu().numpy())))
        plt.title('Low pass filter fft')
        plt.legend([f'low_cutoff: {filter.low_cutoff.item()}\n' f'high_cutoff: {filter.high_cutoff.item()}'])

        # Adjust the spacing between the subplots
        plt.tight_layout()

    plt.show()
