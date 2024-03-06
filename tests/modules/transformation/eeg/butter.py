"""Test the butter filter block"""
from unittest import TestCase

import numpy as np
import pandas as pd

from src.modules.transformation.eeg.butter import ButterFilter
from src.modules.transformation.eeg.butter import butter_lowpass_filter
from src.typing.typing import XData


def setup_data() -> XData:
    # set up data with some nan values
    eeg = {
        "eeg1": pd.DataFrame({
            'e1': [0,1,0,0,0,-1],
            'e2': [0,0,0,0,0,0]
        }, dtype=float),
        "eeg2": pd.DataFrame({
            'e1': [1, 2, 30, 4, 5],
            'e2': [6, 7, 8, 9, 10]
        })
    }
    spec = None
    meta = pd.DataFrame([1, 2, 3, 4, 5])
    return eeg, spec, meta


def expected_data() -> XData:
    data = setup_data()
    for key in data[0]:
        eeg = data[0][key]
        for col in eeg.columns:
            eeg[col] = butter_lowpass_filter(eeg[col])
    return data


class TestButterFilter(TestCase):
    def test_transform(self):
        data = setup_data()
        butter = ButterFilter()
        transformed_data = butter.transform(data)
        expected = expected_data()
        eeg, spec, meta = transformed_data
        for key in eeg.keys():
            pd.testing.assert_frame_equal(eeg[key], expected[0][key])
