"""Test the butter filter block"""
from unittest import TestCase

import numpy as np
import pandas as pd

from src.modules.transformation.eeg.butter import ButterFilter
from src.typing.typing import XData


def generate_sine(frequency: float, duration: float, sample_rate: int) -> np.ndarray:
    """Generate a sine wave."""
    x = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * frequency * x)


def setup_data() -> XData:
    eeg = {
        0: pd.DataFrame(generate_sine(15, 5, 200)),
        1: pd.DataFrame(generate_sine(0.1, 5, 200)),
    }
    meta = pd.DataFrame([1, 2, 3, 4, 5])
    return XData(eeg=eeg, kaggle_spec=None, eeg_spec=None, meta=meta, shared=None)


class TestButterFilter(TestCase):
    def test_lowpass(self):
        data = setup_data()
        butter = ButterFilter(0, 5, 6, 200)
        eeg = butter.transform(data).eeg

        # high frequency should be mostly gone
        self.assertTrue(np.isclose(eeg[0], 0.0, atol=0.15).all())

        # low frequency should mostly still exist
        self.assertTrue(np.isclose(eeg[1].values[:,0], generate_sine(0.1, 5, 200), atol=0.2).all())

    def test_bandpass(self):
        data = setup_data()
        butter = ButterFilter(5, 7, 6, 200)
        eeg = butter.transform(data).eeg

        # high frequency should be mostly gone
        self.assertTrue(np.isclose(eeg[0], 0.0, atol=0.1).all())

        # low frequency should be gone
        self.assertTrue(np.isclose(eeg[1], 0.0, atol=0.01).all())
