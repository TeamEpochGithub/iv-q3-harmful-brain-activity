"""Divide EEG signals by a constant value."""
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.typing import XData


@dataclass
class Divide(VerboseTransformationBlock):
    """Divide EEG signals by a constant value.

    :param value: The constant value to divide the EEG data by
    """

    value: float

    def custom_transform(self, data: XData, **kwargs: Any) -> XData:
        """Divide the EEG data by a constant value.

        :param data: The X data to transform, as tuple (eeg, spec, meta)
        :return: The transformed data
        """
        eeg = data.eeg
        if eeg is None:
            raise ValueError("No EEG data to transform")
        for key in tqdm(eeg, desc="Dividing EEG data"):
            eeg[key] = eeg[key] / self.value
        return data
