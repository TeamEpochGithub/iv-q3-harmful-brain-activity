"""Selects channels from the EEG data."""
from dataclasses import dataclass, field
from typing import Any

from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.typing import XData


@dataclass
class SelectChannels(VerboseTransformationBlock):
    """Select channels based on the indices provided."""

    channels: list[int] = field(default_factory=list)

    def custom_transform(self, data: XData, **transform_args: Any) -> XData:
        """Apply the transformation on the entire dataset.

        :param data: The data to transform.
        :return: The transformed data.
        """
        eeg = data.eeg
        if eeg is None:
            raise ValueError("No EEG data to transform")
        for key in tqdm(eeg, desc="Selecting channels..."):
            # Get number of columns of df
            num_cols = len(eeg[key].columns)
            # Check if all the channels are present
            if max(self.channels) > num_cols:
                raise ValueError("The channels provided are not present in the data")

            eeg[key] = eeg[key].iloc[:, self.channels]
        return data
