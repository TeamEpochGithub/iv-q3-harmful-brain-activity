"""Contains the transformation block for clipping the EEG data."""
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.typing import XData


@dataclass
class Log(VerboseTransformationBlock):
    """Log transform spectrogram data.

    :param kaggle_spec: Apply the transformation to the Kaggle spectrogram data
    :param eeg_spec: Apply the transformation to the EEG spectrogram data
    """

    kaggle_spec: bool = False
    eeg_spec: bool = False

    def custom_transform(self, data: XData, **kwargs: Any) -> XData:
        """Clip the EEG data to a specified range.

        :param data: The X data to transform, as tuple (eeg, spec, meta)
        :return: The transformed data
        """

        if self.kaggle_spec:
            if data.kaggle_spec is None:
                raise ValueError("Data type kaggle_spec is not present in the data.")
            for key in tqdm(data.kaggle_spec, desc="Kaggle Spec - Log Transform"):
                data.kaggle_spec[key] = data.kaggle_spec[key].log()

        if self.eeg_spec:
            if data.eeg_spec is None:
                raise ValueError("Data type eeg_spec is not present in the data.")
            for key in tqdm(data.eeg_spec, desc="EEG Spec - Log Transform"):
                data.eeg_spec[key] = data.eeg_spec[key].log()

        return data
