from unittest import TestCase

import pandas as pd

from src.typing.typing import XData

from src.modules.transformation.eeg.divide import Divide


def setup_data() -> XData:
    eeg = {
        "eeg1": pd.DataFrame([1, 2, 3, 4, 5]),
        "eeg2": pd.DataFrame([6, 7, 8, 9, 10]),
    }
    spec = {
        "spec1": pd.DataFrame([1, 2, 3, 4, 5]),
        "spec2": pd.DataFrame([6, 7, 8, 9, 10]),
    }
    meta = pd.DataFrame([1, 2, 3, 4, 5])
    return eeg, spec, meta


def expected_data() -> XData:
    eeg = {
        "eeg1": pd.DataFrame([0.5, 1, 1.5, 2, 2.5]),
        "eeg2": pd.DataFrame([3, 3.5, 4, 4.5, 5]),
    }
    spec = {
        "spec1": pd.DataFrame([1, 2, 3, 4, 5]),
        "spec2": pd.DataFrame([6, 7, 8, 9, 10]),
    }
    meta = pd.DataFrame([1, 2, 3, 4, 5])
    return eeg, spec, meta


class TestDivide(TestCase):

    def test_transform(self):
        data = setup_data()
        divide = Divide(value=2)
        transformed_data = divide.transform(data)
        expected = expected_data()
        eeg, spec, meta = transformed_data
        for key in eeg.keys():
            pd.testing.assert_frame_equal(eeg[key], expected[0][key])
