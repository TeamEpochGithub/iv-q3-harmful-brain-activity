"""Quantize data for transformation."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.typing import XData


@dataclass
class Quantizer(VerboseTransformationBlock):
    """Quantization for eeg signals.

    :param classes: Classes to quantize to
    """

    classes: int = 1

    def custom_transform(self, data: XData, **kwargs: Any) -> XData:
        """Quantize the data.

        :param data: The X data to transform, as tuple (eeg, spec, meta)
        :return: The transformed data
        """
        eeg = data.eeg
        if eeg is None:
            raise ValueError("No EEG data to transform")
        for key in tqdm(eeg.keys(), desc="Quantizing EEG data"):
            eeg[key] = eeg[key].apply(self.quantize_data)
        return data

    def quantize_data(self, data: pd.DataFrame) -> npt.NDArray[np.float32]:
        """Quantize data for eegs.

        :param data: Eeg data
        :return: Quantized data
        """
        classes = self.classes
        return self.mu_law_encoding(data, classes)

    def mu_law_encoding(self, data: pd.DataFrame, mu: int) -> npt.NDArray[np.float32]:
        """Encode data by mu law.

        :param data: Input data
        :return: Encoded data
        """
        return np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
