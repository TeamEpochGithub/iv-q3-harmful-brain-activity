"""Downsampling for eeg signals."""
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.typing import XData


@dataclass
class Downsample(VerboseTransformationBlock):
    """Downsampling for eeg signals."""

    downsample_factor: int = 5

    def custom_transform(self, data: XData, **kwargs: Any) -> XData:
        """Downsample the eeg signals.

        :param data: The X data to transform, as tuple (eeg, spec, meta)
        :return: The transformed data
        """
        eeg = data.eeg
        if eeg is None:
            raise ValueError("No EEG data to transform")
        for key in tqdm(eeg.keys(), desc="Downsampling EEG data"):
            eeg[key] = eeg[key][:: self.downsample_factor]
        if data.shared is not None and "eeg_freq" in data.shared:
            data.shared["eeg_freq"] //= self.downsample_factor
        return data
