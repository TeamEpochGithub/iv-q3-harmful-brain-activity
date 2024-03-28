"""Class that ontains the rolling window transformation for EEG data."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.typing import XData


@dataclass
class Rolling(VerboseTransformationBlock):
    """Converts / adds rolling window features to the EEG data.

    :param transform_or_add: Whether to transform the data or add the new features
    :param channels: The channels to apply the rolling window to
    :param window_sizes: The size of the rolling window
    :param operations: The operations to apply to the rolling window
    """

    channels: list[int] = field(default_factory=list)
    window_sizes: list[int] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Check the parameters."""
        super().__post_init__()
        # Check if the length of the channels, window_sizes and operations are the same
        if len(self.channels) != len(self.window_sizes) or len(self.channels) != len(self.operations):
            raise ValueError("The length of channels, window_sizes and operations should be the same")

        # Check if the operations are valid
        for operation in self.operations:
            if operation not in ["mean", "std", "min", "max", "median", "sum", "var", "skew", "kurtosis"]:
                raise ValueError(f"Invalid operation: {operation}")

    def custom_transform(self, data: XData, **kwargs: Any) -> XData:
        """Apply the transformation.

        :param data: The X data to transform, as tuple (eeg, spec, meta)
        :return: The transformed data
        """
        eeg = data.eeg
        if eeg is None:
            raise ValueError("No EEG data to transform")
        for key in tqdm(eeg, desc="Apply rolling window to EEG"):
            for i, channel in enumerate(self.channels):
                eeg[key] = self._apply_rolling_window(eeg[key], channel, self.window_sizes[i], self.operations[i])
        return data

    def _apply_rolling_window(self, eeg: pd.DataFrame, channel: int, window_size: int, operation: str) -> pd.DataFrame:
        """Apply the rolling window to the EEG data.

        :param eeg: The EEG data to apply the rolling window to
        :param channel: The channel to apply the rolling window to
        :param window_size: The size of the rolling window
        :param operation: The operation to apply to the rolling window

        :return: The EEG data with the rolling window applied
        """
        eeg[f"channel_{channel}_{operation}_{window_size}"] = eeg.iloc[:, 0].rolling(window=window_size).agg(operation).ffill().bfill().astype(np.float32)
        return eeg
