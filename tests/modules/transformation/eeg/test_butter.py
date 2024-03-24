"""Test the butter filter block"""
from unittest import TestCase

import numpy as np
import pandas as pd

from src.modules.transformation.eeg.butter import ButterFilter
from src.typing.typing import XData


def setup_data() -> XData:
    eeg = {
        0: pd.DataFrame([1, 2, 3, 4, 5]),
        1: pd.DataFrame([6, 7, 8, 9, 10]),
    }
    meta = pd.DataFrame([1, 2, 3, 4, 5])
    return XData(eeg=eeg, kaggle_spec=None, eeg_spec=None, meta=meta, shared=None)


def expected_data_low() -> XData:
    data = setup_data()
    eegs = data.eeg
    butter = ButterFilter(0, 20, 2, 200)
    for key in eegs:
        eeg = eegs[key]
        for col in eeg.columns:
            eeg[col] = butter.butter_lowpass_filter(eeg[col])
    return data


def expected_data_band() -> XData:
    data = setup_data()
    eegs = data.eeg
    butter = ButterFilter(0.5, 20, 2, 200)
    for key in eegs:
        eeg = eegs[key]
        for col in eeg.columns:
            eeg[col] = butter.butter_bandpass_filter(eeg[col])
    return data


class TestButterFilter(TestCase):
    def test_lowpass(self):
        data = setup_data()
        butter = ButterFilter(0, 20, 2, 200)
        eeg = butter.transform(data).eeg
        expected = expected_data_low().eeg
        for key in eeg.keys():
            pd.testing.assert_frame_equal(eeg[key], expected[key])

    def test_bandpass(self):
        data = setup_data()
        butter = ButterFilter(0.5, 20, 2, 200)
        eeg = butter.transform(data).eeg
        expected = expected_data_band().eeg
        for key in eeg.keys():
            pd.testing.assert_frame_equal(eeg[key], expected[key])
