from unittest import TestCase

import pandas as pd
import torch

from src.modules.transformation.clip import Clip
from src.typing.typing import XData


def setup_data() -> XData:
    eeg = {
        0: pd.DataFrame([1, 2, 3, 4, 5]),
        1: pd.DataFrame([6, 7, 8, 9, 10]),
    }
    kaggle_spec = {
        0: torch.tensor([1, 2, 3, 4, 5]),
        1: torch.tensor([6, 7, 8, 9, 10]),
    }
    eeg_spec = {
        0: torch.tensor([1, 2, 3, 4, 5]),
        1: torch.tensor([6, 7, 8, 9, 10]),
    }

    return XData(eeg=eeg, kaggle_spec=kaggle_spec, eeg_spec=eeg_spec, meta=None, shared=None)


class TestClip(TestCase):
    def test_transform_eeg(self):
        data = setup_data()
        test_data = setup_data()

        clip = Clip(lower=2, upper=8, eeg=True)
        data = clip.transform(data)

        for key in data.eeg.keys():
            self.assertTrue((data.eeg[key] >= 2).all().all())
            self.assertTrue((data.eeg[key] <= 8).all().all())
        for key in data.kaggle_spec.keys():
            self.assertTrue((data.kaggle_spec[key] == test_data.kaggle_spec[key]).all())
        for key in data.eeg_spec.keys():
            self.assertTrue((data.eeg_spec[key] == test_data.eeg_spec[key]).all())

    def test_transform_eeg_no_upper(self):
        data = setup_data()
        clip = Clip(lower=2, upper=None, eeg=True)
        eeg = clip.transform(data).eeg
        for key in eeg.keys():
            self.assertTrue((eeg[key] >= 2).all().all())

    def test_transform_eeg_no_lower(self):
        data = setup_data()
        clip = Clip(lower=None, upper=8, eeg=True)
        eeg = clip.transform(data).eeg
        for key in eeg.keys():
            self.assertTrue((eeg[key] <= 8).all().all())

    def test_transform_all(self):
        data = setup_data()
        clip = Clip(lower=2, upper=8, eeg=True, kaggle_spec=True, eeg_spec=True)
        data = clip.transform(data)

        for key in data.eeg.keys():
            self.assertTrue((data.eeg[key] >= 2).all().all())
            self.assertTrue((data.eeg[key] <= 8).all().all())
        for key in data.kaggle_spec.keys():
            self.assertTrue((data.kaggle_spec[key] >= 2).all().all())
            self.assertTrue((data.kaggle_spec[key] <= 8).all().all())
        for key in data.eeg_spec.keys():
            self.assertTrue((data.eeg_spec[key] >= 2).all().all())
            self.assertTrue((data.eeg_spec[key] <= 8).all().all())
