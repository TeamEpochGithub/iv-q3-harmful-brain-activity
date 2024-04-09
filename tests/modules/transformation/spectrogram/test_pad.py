from unittest import TestCase

import torch
import copy

from src.modules.transformation.spectrogram.pad import Pad
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


class TestPad(TestCase):
    def test_padding_kaggle_spec_one(self):
        """Test the padding of the kaggle spectrogram data. Pad only the last dimension, left and right by 2 (i.e. the final image is aligned center)."""
        data = setup_data()
        test_data = copy.deepcopy(data)

        pad = Pad(pad_list=[2, 2], kaggle_spec=True)
        data = pad.transform(data)

        for key in data.kaggle_spec.keys():
            self.assertTrue(data.kaggle_spec[key].shape[0] == test_data.kaggle_spec[key].shape[0])
            self.assertTrue(data.kaggle_spec[key].shape[1] == test_data.kaggle_spec[key].shape[1])
            self.assertTrue(data.kaggle_spec[key].shape[2] == test_data.kaggle_spec[key].shape[2] + 4)

    def test_padding_kaggle_spec_two(self):
        """Test the padding of the kaggle spectrogram data. Pad the top by 5 and left by 2 (i.e. the final image is aligned bottom right)."""
        data = setup_data()
        test_data = copy.deepcopy(data)

        pad = Pad(pad_list=[2, 0, 5, 0], kaggle_spec=True)
        data = pad.transform(data)

        for key in data.kaggle_spec.keys():
            self.assertTrue(data.kaggle_spec[key].shape[0] == test_data.kaggle_spec[key].shape[0])
            self.assertTrue(data.kaggle_spec[key].shape[1] == test_data.kaggle_spec[key].shape[1] + 5)
            self.assertTrue(data.kaggle_spec[key].shape[2] == test_data.kaggle_spec[key].shape[2] + 2)
