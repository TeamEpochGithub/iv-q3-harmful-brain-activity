"""Downsampling for eeg signals."""
from dataclasses import dataclass
from typing import Any

import numpy as np
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.typing import XData


@dataclass
class Downsample(VerboseTransformationBlock):
    """Downsampling for eeg signals."""

    downsample_factor: int = 5
    operation: str | None = None

    def custom_transform(self, data: XData, **kwargs: Any) -> XData:
        """Downsample the eeg signals.

        :param data: The X data to transform, as tuple (eeg, spec, meta)
        :return: The transformed data
        """
        eeg = data.eeg
        if eeg is None:
            raise ValueError("No EEG data to transform")

        if self.operation is not None and self.operation in ["mean", "std", "min", "max", "median", "sum", "var", "skew", "kurtosis"]:
            for key in tqdm(eeg.keys(), desc=f"Downsampling EEG data with {self.operation} by {self.downsample_factor}"):
                eeg[key] = eeg[key].rolling(window=self.downsample_factor).agg(self.operation).ffill().bfill().astype(np.float32)
                eeg[key] = eeg[key][:: self.downsample_factor]

        if self.operation is None:
            for key in tqdm(eeg.keys(), desc="Downsampling EEG data"):
                eeg[key] = eeg[key][:: self.downsample_factor]

        if data.shared is not None and "eeg_freq" in data.shared:
            data.shared["eeg_freq"] //= self.downsample_factor
        return data
