"""This module contains a transformation block that sets NaN values in the EEG data to zero."""

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.typing import XData


class NaNToZero(VerboseTransformationBlock):
    """An example transformation block for the transformation pipeline."""

    def custom_transform(self, data: XData, **kwargs) -> XData:
        """Set NaN values in the EEG data to zero

        :param data: The X data to transform, as tuple (eeg, spec, meta)
        :return: The transformed data
        """
        eeg, spec, meta = data
        for key in eeg.keys():
            eeg[key] = eeg[key].fillna(0)
        return data
