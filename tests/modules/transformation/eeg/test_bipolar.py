from unittest import TestCase

import numpy as np
import pandas as pd

from src.modules.transformation.eeg.bipolar import BipolarEEG
from src.typing.typing import XData


def generate_eeg() -> pd.DataFrame:
    electrode_names = [
        'Fp1', 'F7', 'T3', 'T5', 'O1',
        'Fp2', 'F8', 'T4', 'T6', 'O2',
        'F3', 'C3', 'P3',
        'Fz', 'Cz', 'Pz',
        'F4', 'C4', 'P4', 'EKG'
    ]
    return pd.DataFrame(np.random.rand(5, len(electrode_names)), columns=electrode_names)


def setup_data_bipolar() -> XData:
    eeg = {
        0: generate_eeg(),
        1: generate_eeg(),
    }
    meta = pd.DataFrame([1, 2, 3, 4, 5])
    return XData(eeg=eeg, kaggle_spec=None, eeg_spec=None, meta=meta, shared=None)


class TestBipolarEEG(TestCase):
    def test_map_full_contains_all_electrodes(self):
        data = setup_data_bipolar()
        bipolar = BipolarEEG(use_full_map=True, keep_ekg=True)
        eeg = bipolar.transform(data).eeg
        for key in eeg.keys():
            self.assertEqual(19, len(eeg[key].columns))
        assert 'LT1' in eeg[0].columns
        assert 'LT4' in eeg[0].columns
        assert 'EKG' in eeg[0].columns
        assert 'Fp1' not in eeg[0].columns

    def test_map_half_contains_half_electrodes(self):
        data = setup_data_bipolar()
        bipolar = BipolarEEG(use_full_map=False, keep_ekg=True)
        eeg = bipolar.transform(data).eeg
        for key in eeg.keys():
            self.assertEqual(10, len(eeg[key].columns))
        assert 'LT1' in eeg[0].columns
        assert 'LT4' not in eeg[0].columns
        assert 'EKG' in eeg[0].columns
        assert 'Fp1' not in eeg[0].columns

    def test_map_half_no_ekg_contains(self):
        data = setup_data_bipolar()
        bipolar = BipolarEEG(use_full_map=False, keep_ekg=False)
        eeg = bipolar.transform(data).eeg
        for key in eeg.keys():
            self.assertEqual(9, len(eeg[key].columns))
        assert 'LT1' in eeg[0].columns
        assert 'LT4' not in eeg[0].columns
        assert 'EKG' not in eeg[0].columns
        assert 'Fp1' not in eeg[0].columns

    def test_difference_correct(self):
        data = setup_data_bipolar()
        fp1 = data.eeg[0]['Fp1']
        f7 = data.eeg[0]['F7']

        bipolar = BipolarEEG(use_full_map=True, keep_ekg=True)
        eeg = bipolar.transform(data).eeg
        self.assertTrue(np.allclose(eeg[0]['LT1'], fp1 - f7))

