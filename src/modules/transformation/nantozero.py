"""Transformation block that sets NaN values in the EEG data to zero."""
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.typing import XData


@dataclass
class NaNToZero(VerboseTransformationBlock):
    """Transformation block that sets NaN values in the EEG data to zero.

    :param eeg: Apply the transformation to the EEG data
    :param kaggle_spec: Apply the transformation to the Kaggle spectrogram data
    :param eeg_spec: Apply the transformation to the EEG spectrogram data
    """

    eeg: bool = False
    kaggle_spec: bool = False
    eeg_spec: bool = False

    def custom_transform(self, data: XData, **kwargs: Any) -> XData:
        """Set NaN values in the EEG data to zero.

        :param data: The X data to transform, as tuple (eeg, spec, meta)
        :return: The transformed data
        """
        if self.eeg:
            if data.eeg is None:
                raise ValueError("Data type eeg is not present in the data.")
            for key in tqdm(data.eeg, desc="EEG - Setting NaN values to zero"):
                data.eeg[key] = data.eeg[key].fillna(0)

        if self.kaggle_spec:
            if data.kaggle_spec is None:
                raise ValueError("Data type kaggle_spec is not present in the data.")
            for key in tqdm(data.kaggle_spec, desc="Kaggle Spec - Setting NaN values to zero"):
                data.kaggle_spec[key] = data.kaggle_spec[key].nan_to_num(0.0)

        if self.eeg_spec:
            if data.eeg_spec is None:
                raise ValueError("Data type eeg_spec is not present in the data.")
            for key in tqdm(data.eeg_spec, desc="EEG Spec - Setting NaN values to zero"):
                data.eeg_spec[key] = data.eeg_spec[key].nan_to_num(0.0)

        return data
