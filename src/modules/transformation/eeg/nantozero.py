"""Transformation block that sets NaN values in the EEG data to zero."""
from typing import Any

from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.typing import XData


class NaNToZero(VerboseTransformationBlock):
    """An example transformation block for the transformation pipeline."""

    def custom_transform(self, data: XData, **kwargs: Any) -> XData:
        """Set NaN values in the EEG data to zero.

        :param data: The X data to transform, as tuple (eeg, spec, meta)
        :return: The transformed data
        """
        eeg = data.eeg

        if eeg is None:
            raise ValueError("No EEG data to transform")
        for key in tqdm(eeg, desc="EEG - Setting NaN values to zero"):
            eeg[key] = eeg[key].fillna(0)

        kaggle_spec = data.kaggle_spec
        if kaggle_spec is not None:
            for key in tqdm(kaggle_spec, desc="Kaggle Spec - Setting NaN values to zero"):
                kaggle_spec[key] = kaggle_spec[key].nan_to_num(0.0)
        return data
