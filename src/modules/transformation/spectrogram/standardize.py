"""Contains the transformation block for clipping the EEG data."""
from dataclasses import dataclass
from typing import Any

import torch
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.typing import XData


@dataclass
class Standardize(VerboseTransformationBlock):
    """Standardize the spectrogram data on an per image basis.

    :param kaggle_spec: Apply the transformation to the Kaggle spectrogram data
    :param eeg_spec: Apply the transformation to the EEG spectrogram data
    """

    kaggle_spec: bool = False
    eeg_spec: bool = False

    def _standardize(self, data: dict[int, torch.Tensor], description: str) -> None:
        """Standardize the spectrogram data.

        :param data: The X data to transform
        :param description: The description of the transformation
        :return: The transformed data
        """
        ep = 1e-6
        for key in tqdm(data, desc=description):
            if len(data[key].shape) != 3:
                raise ValueError(f"Data shape is not 3D, got {data[key].shape}")

            result = []
            for i in range(data[key].shape[0]):
                img = data[key][i]
                m = img.float().mean()
                s = img.float().std()

                img = img - m
                img = img / (s + ep)
                result.append(img)

            data[key] = torch.stack(result)

    def custom_transform(self, data: XData, **kwargs: Any) -> XData:
        """Standardize the spectrogram data.

        :param data: The X data to transform
        :return: The transformed data
        """
        if self.kaggle_spec:
            if data.kaggle_spec is None:
                raise ValueError("Data type kaggle_spec is not present in the data.")
            self._standardize(data.kaggle_spec, "Kaggle Spec - Standardize")

        if self.eeg_spec:
            if data.eeg_spec is None:
                raise ValueError("Data type eeg_spec is not present in the data.")
            self._standardize(data.eeg_spec, "EEG Spec - Standardize")

        return data
