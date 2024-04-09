from unittest import TestCase

import numpy as np
import pandas as pd

from src.modules.transformation.eeg.select8 import Select8
from src.typing.typing import XData


INITIAL_COLUMNS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2','EKG']

def setup_eeg_data() -> XData:
    eeg = {
        0: pd.DataFrame(np.random.rand(10, 20), columns=INITIAL_COLUMNS),
        1: pd.DataFrame(np.random.rand(10, 20), columns=INITIAL_COLUMNS),
    }
    meta = pd.DataFrame([1, 2, 3, 4, 5])
    return XData(eeg=eeg, kaggle_spec=None, eeg_spec=None, meta=meta, shared=None)


class TestSelect8(TestCase):
    def test_custom_transform(self):
        data = setup_eeg_data()
        transformer = Select8()
        transformed_data = transformer.transform(data)
        for key in transformed_data.eeg:
            self.assertEqual(list(transformed_data.eeg[key].columns), ['Fp1', 'T3', 'C3', 'O1', 'Fp2', 'C4', 'T4', 'O2'])
        self.assertEqual(list(transformed_data.meta[0]), [1, 2, 3, 4, 5])
        self.assertEqual(transformed_data.kaggle_spec, None)
        self.assertEqual(transformed_data.eeg_spec, None)
        self.assertEqual(transformed_data.shared, None)
        self.assertEqual(transformed_data.eeg[0].shape, (10, 8))
        self.assertEqual(transformed_data.eeg[1].shape, (10, 8))