"""Contains the transformation block for clipping the EEG data."""
from dataclasses import dataclass
from typing import Any

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.typing import XData


@dataclass
class ClipEEG(VerboseTransformationBlock):
    """Clip the EEG data to a specified range.

    :param lower: The minimum value to clip the EEG data to, or None to not clip the minimum value
    :param upper: The maximum value to clip the EEG data to, or None to not clip the maximum value
    """

    lower: float | None = None
    upper: float | None = None

    def custom_transform(self, data: XData, **kwargs: Any) -> XData:
        """Clip the EEG data to a specified range.

        :param data: The X data to transform, as tuple (eeg, spec, meta)
        :return: The transformed data
        """
        eeg, spec, meta = data
        if eeg is None:
            raise ValueError("No EEG data to transform")
        if self.lower is not None and self.upper is not None:
            for key in eeg:
                eeg[key] = eeg[key].clip(self.lower, self.upper)
        return data
