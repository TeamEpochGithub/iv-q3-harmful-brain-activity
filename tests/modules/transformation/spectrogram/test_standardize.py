from unittest import TestCase

import torch
import copy

from src.modules.transformation.spectrogram.standardize import Standardize
from src.typing.typing import XData



def setup_data() -> XData:

    kaggle_spec = {
        0: torch.arange(0, 75).reshape(3, 5, 5),
        1: torch.arange(0, 75).reshape(3, 5, 5),
    }

    eeg_spec = {
        0: torch.rand(3, 5, 5),
        1: torch.rand(3, 5, 5),
    }

    return XData(eeg=None, kaggle_spec=kaggle_spec, eeg_spec=eeg_spec, meta=None, shared=None)


class TestLog(TestCase):
    def test_standardize(self):
        data = setup_data()
        test_data = copy.deepcopy(data)

        standardize = Standardize(kaggle_spec=True)
        data = standardize.transform(data)

        for key in data.kaggle_spec.keys():
            for i in range(data.kaggle_spec[key].shape[0]):
                img = data.kaggle_spec[key][i]
                m = img.float().mean()
                s = img.float().std()

                self.assertTrue(torch.isclose(m, torch.tensor(0.0)))
                self.assertTrue(torch.isclose(s, torch.tensor(1.0)))

        print(data.eeg_spec[0][0])
        for key in data.eeg_spec.keys():
            self.assertTrue((data.eeg_spec[key] == test_data.eeg_spec[key]).all())

    def test_standardize_all(self):
        data = setup_data()

        img = data.eeg_spec[0][0]
        m = img.float().mean()
        s = img.float().std()

        standardize = Standardize(kaggle_spec=True, eeg_spec=True)
        data = standardize.transform(data)

        for key in data.kaggle_spec.keys():
            for i in range(data.kaggle_spec[key].shape[0]):
                img = data.kaggle_spec[key][i]
                m = img.float().mean()
                s = img.float().std()

                self.assertTrue(torch.isclose(m, torch.tensor(0.0)))
                self.assertTrue(torch.isclose(s, torch.tensor(1.0)))

        for key in data.eeg_spec.keys():
            for i in range(data.eeg_spec[key].shape[0]):
                img = data.eeg_spec[key][i]
                m = img.float().mean()
                s = img.float().std()

                self.assertTrue(torch.isclose(m, torch.tensor(0.0), atol=1e-5))
                self.assertTrue(torch.isclose(s, torch.tensor(1.0), atol=1e-5))
