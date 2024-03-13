from unittest import TestCase

import pandas as pd

from src.modules.transformation.eeg.clip import ClipEEG
from src.typing.typing import XData


def setup_data() -> XData:
    eeg = {
        0: pd.DataFrame([1, 2, 3, 4, 5]),
        1: pd.DataFrame([6, 7, 8, 9, 10]),
    }
    meta = pd.DataFrame([1, 2, 3, 4, 5])
    return XData(eeg=eeg, kaggle_spec=None, eeg_spec=None, meta=meta, shared=None)


class TestClipEEG(TestCase):
    def test_transform(self):
        data = setup_data()
        clip = ClipEEG(lower=2, upper=8)
        eeg = clip.transform(data).eeg
        for key in eeg.keys():
            self.assertTrue((eeg[key] >= 2).all().all())
            self.assertTrue((eeg[key] <= 8).all().all())

    def test_transform_no_upper(self):
        data = setup_data()
        clip = ClipEEG(lower=2, upper=None)
        eeg = clip.transform(data).eeg
        for key in eeg.keys():
            self.assertTrue((eeg[key] >= 2).all().all())

    def test_transform_no_lower(self):
        data = setup_data()
        clip = ClipEEG(lower=None, upper=8)
        eeg = clip.transform(data).eeg
        for key in eeg.keys():
            self.assertTrue((eeg[key] <= 8).all().all())