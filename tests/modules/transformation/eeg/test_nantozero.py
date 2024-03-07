from unittest import TestCase

import numpy as np
import pandas as pd

from src.modules.transformation.eeg.nantozero import NaNToZero
from src.typing.typing import XData


def setup_data_nan() -> XData:
    eeg = {
        0: pd.DataFrame([1, 2, 3, 4, 5], dtype=float),
        1: pd.DataFrame([6, 7, 8, 9, 10], dtype=float),
    }
    eeg[0].iloc[1] = np.nan
    eeg[1].iloc[3] = np.nan
    meta = pd.DataFrame([1, 2, 3, 4, 5])
    return XData(eeg=eeg, kaggle_spec=None, eeg_spec=None, meta=meta)


def expected_data_zero() -> XData:
    eeg = {
        0: pd.DataFrame([1, 2, 3, 4, 5], dtype=float),
        1: pd.DataFrame([6, 7, 8, 9, 10], dtype=float),
    }
    eeg[0].iloc[1] = 0
    eeg[1].iloc[3] = 0
    meta = pd.DataFrame([1, 2, 3, 4, 5])
    return XData(eeg=eeg, kaggle_spec=None, eeg_spec=None, meta=meta)


class TestNaNToZero(TestCase):
    def test_transform(self):
        data = setup_data_nan()
        nan_to_zero = NaNToZero()
        eeg = nan_to_zero.transform(data).eeg
        expected = expected_data_zero().eeg
        for key in eeg.keys():
            pd.testing.assert_frame_equal(eeg[key], expected[key])
