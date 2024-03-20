"""Quantize data for transformation"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.typing import XData
from typing import Any
from dataclasses import dataclass

@dataclass
class Quantizer(VerboseTransformationBlock):
    """Quantization for eeg signals."""

    classes: int = 1

    def custom_transform(self, data: XData, **kwargs: Any) -> XData:
        """Quantize the data

        :param data: The X data to transform, as tuple (eeg, spec, meta)
        :return: The transformed data
        """
        eeg = data.eeg
        if eeg is None:
            raise ValueError("No EEG data to transform")
        for key in tqdm(eeg.keys(), desc="Quantizing EEG data"):
            eeg[key] = eeg[key].apply(self.quantize_data)
        return data

    def quantize_data(self, data: pd.DataFrame):
        classes = self.classes
        mu_x = self.mu_law_encoding(data, classes)

        return mu_x

    def mu_law_encoding(self, data: pd.DataFrame, mu):
        return np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)

