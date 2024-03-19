"""Contains the transformation block for clipping the EEG data."""
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.typing import XData


@dataclass
class Clip(VerboseTransformationBlock):
    """Clip the EEG data to a specified range.

    :param lower: The minimum value to clip the EEG data to, or None to not clip the minimum value
    :param upper: The maximum value to clip the EEG data to, or None to not clip the maximum value
    :param eeg: Apply the transformation to the EEG data
    :param kaggle_spec: Apply the transformation to the Kaggle spectrogram data
    :param eeg_spec: Apply the transformation to the EEG spectrogram data
    """

    lower: float | None = None
    upper: float | None = None

    eeg: bool = False
    kaggle_spec: bool = False
    eeg_spec: bool = False

    def custom_transform(self, data: XData, **kwargs: Any) -> XData:
        """Clip the EEG data to a specified range.

        :param data: The X data to transform, as tuple (eeg, spec, meta)
        :return: The transformed data
        """
        if self.eeg:
            if data.eeg is None:
                raise ValueError("Data type eeg is not present in the data.")
            for key in tqdm(data.eeg, desc="EEG - Clipping"):
                data.eeg[key] = data.eeg[key].clip(self.lower, self.upper)
        
        if self.kaggle_spec:
            if data.kaggle_spec is None:
                raise ValueError("Data type kaggle_spec is not present in the data.")
            for key in tqdm(data.kaggle_spec, desc="Kaggle Spec - Clipping"):
                data.kaggle_spec[key] = data.kaggle_spec[key].clip(self.lower, self.upper)

        if self.eeg_spec:
            if data.eeg_spec is None:
                raise ValueError("Data type eeg_spec is not present in the data.")
            for key in tqdm(data.eeg_spec, desc="EEG Spec - Clipping"):
                data.eeg_spec[key] = data.eeg_spec[key].clip(self.lower, self.upper)

        return data
