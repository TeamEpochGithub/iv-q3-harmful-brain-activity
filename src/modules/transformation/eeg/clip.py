"""This module contains the transformation block for clipping the EEG data."""
from dataclasses import dataclass

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.typing import XData


@dataclass
class ClipEEG(VerboseTransformationBlock):
    """Clip the EEG data to a specified range.

    :param min: The minimum value to clip the EEG data to, or None to not clip the minimum value
    :param max: The maximum value to clip the EEG data to, or None to not clip the maximum value
    """

    min: float | None = None
    max: float | None = None

    def custom_transform(self, data: XData, **kwargs) -> XData:
        """Clip the EEG data to a specified range.

        :param data: The X data to transform, as tuple (eeg, spec, meta)
        :return: The transformed data
        """
        eeg, spec, meta = data
        if self.min is not None and self.max is not None:
            for key in eeg.keys():
                eeg[key] = eeg[key].clip(self.min, self.max)
        return data
