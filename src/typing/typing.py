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
    :param shared: The shared data to be used in the pipeline. Contains frequency data, offset data, etc.
    """

    eeg: dict[int, pd.DataFrame] | None
    kaggle_spec: dict[int, torch.Tensor] | None
    eeg_spec: dict[int, torch.Tensor] | None
    meta: pd.DataFrame
    shared: dict[str, any] | None
