from unittest import TestCase

import torch
import numpy as np
import pandas as pd

from src.modules.transformation.spectrogram.eeg_to_spectrogram import EEGToSpectrogram
from src.typing.typing import XData

def setup_data_sine() -> XData:
    def create_test_eeg():
        # Parameters
        num_steps = 18000  # Number of steps per signal
        num_signals = 20   # Number of different signals

        # Generate time steps
        time_steps = torch.arange(0, num_steps) / num_steps  # Normalized time steps [0, 1]

        # Preallocate tensor for signals
        signals = torch.zeros((num_steps, num_signals))

        # Generate signals
        for i in range(num_signals):
            signals[:, i] = torch.sin(2 * np.pi * i * time_steps)

        # Convert the signals tensor to a NumPy array
        signals_array = signals.numpy()

        # Define the column names as specified
        column_names = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'EKG']

        # Create a pandas DataFrame
        signals_df = pd.DataFrame(data=signals_array, columns=column_names)
        return signals_df

    return XData(eeg={ 0: create_test_eeg() }, kaggle_spec=None, eeg_spec=None, meta=None)

def expected_data_sine(input: XData) -> XData:
    return XData(eeg=input.eeg, kaggle_spec=None, eeg_spec={ 0: torch.load('tests/modules/transformation/spectrogram/test_eeg_spec.pt') }, meta=None)

class TestEEGToSpectrogram(TestCase):
    def test_transform(self):
        data = setup_data_sine()
        eeg_to_spectrogram = EEGToSpectrogram(200)
        eeg_spec = eeg_to_spectrogram.transform(data).eeg_spec
        expected = expected_data_sine(data).eeg_spec
        for key in eeg_spec.keys():
            torch.testing.assert_allclose(eeg_spec[key], expected[key])

    # Implement this test as soon as indexing works
    # def test_indexing(self):
    #     pass
