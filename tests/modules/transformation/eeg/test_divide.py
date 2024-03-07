from unittest import TestCase

import pandas as pd

from src.modules.transformation.eeg.divide import Divide
from src.typing.typing import XData


def setup_data() -> XData:
    eeg = {
        0: pd.DataFrame([1, 2, 3, 4, 5]),
        1: pd.DataFrame([6, 7, 8, 9, 10]),
    }
    meta = pd.DataFrame([1, 2, 3, 4, 5])
    return XData(eeg=eeg, kaggle_spec=None, eeg_spec=None, meta=meta)


def expected_data() -> XData:
    eeg = {
        0: pd.DataFrame([0.5, 1, 1.5, 2, 2.5]),
        1: pd.DataFrame([3, 3.5, 4, 4.5, 5]),
    }
    meta = pd.DataFrame([1, 2, 3, 4, 5])
    return XData(eeg=eeg, kaggle_spec=None, eeg_spec=None, meta=meta)


class TestDivide(TestCase):
    def test_transform(self):
        data = setup_data()
        divide = Divide(value=2)
        eeg = divide.transform(data).eeg
        expected = expected_data().eeg
        for key in eeg.keys():
            pd.testing.assert_frame_equal(eeg[key], expected[key])
