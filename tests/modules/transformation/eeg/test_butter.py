"""Test the butter filter block"""
from unittest import TestCase

import numpy as np
import pandas as pd

from src.modules.transformation.eeg.butter import ButterFilter
from src.modules.transformation.eeg.butter import butter_lowpass_filter
from src.typing.typing import XData


def setup_data() -> XData:
    eeg = {
        0: pd.DataFrame([1, 2, 3, 4, 5]),
        1: pd.DataFrame([6, 7, 8, 9, 10]),
    }
    meta = pd.DataFrame([1, 2, 3, 4, 5])
    return XData(eeg=eeg, kaggle_spec=None, eeg_spec=None, meta=meta)


def expected_data() -> XData:
    data = setup_data()
    eegs = data.eeg
    for key in eegs:
        eeg = eegs[key]
        for col in eeg.columns:
            eeg[col] = butter_lowpass_filter(eeg[col])
    return data


class TestButterFilter(TestCase):
    def test_transform(self):
        data = setup_data()
        butter = ButterFilter()
        eeg = butter.transform(data).eeg
        expected = expected_data().eeg
        for key in eeg.keys():
            pd.testing.assert_frame_equal(eeg[key], expected[key])
