from unittest import TestCase

import pandas as pd

from src.modules.transformation.eeg.clip import ClipEEG
from src.typing.typing import XData


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


class TestClipEEG(TestCase):
    def test_transform(self):
        data = setup_data()
        clip = ClipEEG(lower=2, upper=8)
        transformed_data = clip.transform(data)
        eeg, spec, meta = transformed_data
        for key in eeg.keys():
            self.assertTrue((eeg[key] >= 2).all().all())
            self.assertTrue((eeg[key] <= 8).all().all())
