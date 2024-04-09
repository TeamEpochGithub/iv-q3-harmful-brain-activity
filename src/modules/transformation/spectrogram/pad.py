"""Contains the transformation block for clipping the EEG data."""
from dataclasses import dataclass
from typing import Any

import torch
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.typing import XData


@dataclass
class Pad(VerboseTransformationBlock):
    """Pad the data to a specified size.

    :param pad_list: The size to pad the data to, in the format [left, right, top, bottom].
    :param pad_value: The value to pad the data with. Default is 0.0
    :param eeg: Apply the transformation to the EEG data
    :param kaggle_spec: Apply the transformation to the Kaggle spectrogram data
    :param eeg_spec: Apply the transformation to the EEG spectrogram data
    """

    pad_list: list[int]
    pad_value: float = 0.0

    kaggle_spec: bool = False
    eeg_spec: bool = False

    def custom_transform(self, data: XData, **kwargs: Any) -> XData:
        """Pad the data to a specified size.

        :param data: The X data to transform
        :return: The transformed data
        """
        if self.pad_list is None:
            raise ValueError("Pad Transformation: Pad list not defined.")

        if self.kaggle_spec and self.eeg_spec:
            raise ValueError("Pad Transformation: Both kaggle_spec and eeg_spec cannot be set to True.")

        working_data = None
        description = None
        if self.kaggle_spec:
            if data.kaggle_spec is None:
                raise ValueError("Data type kaggle_spec is not present in the data.")
            working_data = data.kaggle_spec
            description = "Kaggle Spec - Padding"

        elif self.eeg_spec:
            if data.eeg_spec is None:
                raise ValueError("Data type eeg_spec is not present in the data.")
            working_data = data.eeg_spec
            description = "EEG Spec - Padding"

        else:
            return data

        # Calculate the padding for terminal logging
        test_data = next(iter(working_data.values()))
        test_data = torch.nn.functional.pad(test_data, self.pad_list, value=self.pad_value)
        self.log_to_terminal(f"Padding {'kaggle' if self.kaggle_spec else 'eeg' }_spec data to size {test_data.shape}")

        # Apply the padding
        for key in tqdm(working_data, desc=description):
            working_data[key] = torch.nn.functional.pad(working_data[key], self.pad_list, value=self.pad_value)

        return data
