from unittest import TestCase

import pandas as pd
import torch
import copy

from src.modules.transformation.spectrogram.log import Log
from src.typing.typing import XData


def setup_data() -> XData:
    kaggle_spec = {
        0: torch.tensor([1, 2, 3, 4, 5]),
        1: torch.tensor([6, 7, 8, 9, 10]),
    }
    eeg_spec = {
        0: torch.tensor([1, 2, 3, 4, 5]),
        1: torch.tensor([6, 7, 8, 9, 10]),
    }

    return XData(eeg=None, kaggle_spec=kaggle_spec, eeg_spec=eeg_spec, meta=None, shared=None)


class TestLog(TestCase):
    def test_transform(self):
        data = setup_data()
        test_data = copy.deepcopy(data)

        log = Log(kaggle_spec=True)
        data = log.transform(data)

        for key in data.kaggle_spec.keys():
            self.assertTrue((data.kaggle_spec[key] == test_data.kaggle_spec[key].log()).all())
        for key in data.eeg_spec.keys():
            self.assertTrue((data.eeg_spec[key] == test_data.eeg_spec[key]).all())

    def test_transform_all(self):
        data = setup_data()
        log = Log(kaggle_spec=True, eeg_spec=True)
        data = log.transform(data)
        for key in data.kaggle_spec.keys():
            self.assertTrue((data.kaggle_spec[key] == setup_data().kaggle_spec[key].log()).all())
        for key in data.eeg_spec.keys():
            self.assertTrue((data.eeg_spec[key] == setup_data().eeg_spec[key].log()).all())
