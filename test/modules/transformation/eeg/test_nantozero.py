from unittest import TestCase

import pandas as pd

from src.typing.typing import XData

from src.modules.transformation.eeg.nantozero import NaNToZero
import numpy as np


def setup_data_nan() -> XData:
    # set up data with some nan values
    eeg = {
        "eeg1": pd.DataFrame([1, 2, 3, 4, 5], dtype=float),
        "eeg2": pd.DataFrame([6, np.nan, 8, 9, 10], dtype=float),
    }
    spec = {
        "spec1": pd.DataFrame([1, 2, 3, 4, 5]),
        "spec2": pd.DataFrame([6, 7, 8, 9, 10]),
    }
    meta = pd.DataFrame([1, 2, 3, 4, 5])
    return eeg, spec, meta


def expected_data_nan() -> XData:
    # expected data with nan values set to zero
    eeg = {
        "eeg1": pd.DataFrame([1, 2, 3, 4, 5], dtype=float),
        "eeg2": pd.DataFrame([6, 0, 8, 9, 10], dtype=float),
    }
    spec = {
        "spec1": pd.DataFrame([1, 2, 3, 4, 5]),
        "spec2": pd.DataFrame([6, 7, 8, 9, 10]),
    }
    meta = pd.DataFrame([1, 2, 3, 4, 5])
    return eeg, spec, meta


class TestNaNToZero(TestCase):

    def test_transform(self):
        data = setup_data_nan()
        nan_to_zero = NaNToZero()
        transformed_data = nan_to_zero.transform(data)
        expected = expected_data_nan()
        eeg, spec, meta = transformed_data
        for key in eeg.keys():
            pd.testing.assert_frame_equal(eeg[key], expected[0][key])
