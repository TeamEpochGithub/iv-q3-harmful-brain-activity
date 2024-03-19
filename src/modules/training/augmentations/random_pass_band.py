from dataclasses import dataclass
import torch
import scipy
import torch.nn.functional as F
from torchaudio.functional import convolve

@dataclass
class RandomPassBand():
    """Randomly pass a band of frequencies."""
    low_range: list[float]
    high_range: list[float]
    
    
    def __post_init__(self):
        """Post init."""
        # Create the 2 firwin windows
        self.high_cutoff = 10
        self.low_cutoff = 30
        self.band_win = torch.from_numpy(scipy.signal.firwin(101, [self.high_cutoff, self.low_cutoff], window='hamming', fs=200, pass_zero='bandpass').astype('float32')).unsqueeze(0).unsqueeze(0)
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # convolve the two kernels
        band_pass = convolve(x, self.band_win.to('cuda'), mode='same')
        return band_pass


# Example usage
if __name__ == "__main__":
    import numpy as np
    # read a single eeg sequence
    import pandas as pd
    eeg = pd.read_parquet('C:\\Users\\Tolga\\Desktop\\EPOCH-IV\\q3-harmful-brain-activity\\data\\raw\\train_eegs\\687656140.parquet') 
    # get 1 channel and make it numpy
    signal = eeg['F7'].values
    filter = RandomPassBand(low_range=[0.5], high_range=[30])
    kernel = filter.band_win.cpu().numpy()
    numpy_result = np.convolve(signal, kernel[0][0], mode='same')
    torch_result = filter(torch.from_numpy(signal).to('cuda').unsqueeze(0).unsqueeze(0))
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(signal)
    plt.title('Original signal')
    #plot the original fft
    freqs = np.fft.rfftfreq(len(signal), 1/200)
    plt.figure()
    plt.plot(freqs, np.abs(np.fft.rfft(signal)))
    plt.title('Original signal fft')


    plt.figure()
    plt.plot(numpy_result)
    plt.plot(torch_result[0][0].cpu().numpy(), 'r--')
    plt.legend(['numpy', 'torch'])
    plt.title('Low pass filter')
    plt.figure()
    # plot the fft of the signal
    # create frequency axis
    freqs = np.fft.rfftfreq(len(signal), 1/200)
    plt.plot(freqs, np.abs(np.fft.rfft(numpy_result)))
    plt.show()
