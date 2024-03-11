"""Common type definitions for the project."""
from dataclasses import dataclass

import pandas as pd
import torch


@dataclass
class XData:
    """The X data to be used in the pipeline.

    :param eeg: The EEG data, as a dictionary of DataFrames
    :param kaggle_spec: The Kaggle spectrogram data, as a dictionary of Tensors
    :param eeg_spec: The EEG spectrogram data, as a dictionary of Tensors
    :param meta: The metadata, as a DataFrame
    """

    eeg: dict[int, pd.DataFrame] | None
    kaggle_spec: dict[int, torch.Tensor] | None
    eeg_spec: dict[int, torch.Tensor] | None
    meta: pd.DataFrame

    def __repr__(self) -> str:
        return "XData"
