"""Butter filter for eeg signals."""
from typing import Any

import numpy as np
import pandas as pd
from numpy import typing as npt
from scipy.signal import butter, lfilter
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.typing import XData


def butter_lowpass_filter(data: pd.DataFrame, cutoff_freq: float = 20, sampling_rate: int = 200, order: int = 4) -> npt.NDArray[np.float32]:
    """Filter the data with a butter filter.

    Taken from "https://www.kaggle.com/code/nartaa/features-head-starter.
    :param data: The data to filter
    :param cutoff_freq: The cutoff frequency
    :param sampling_rate: The sampling rate
    :param order: The order of the filter
    """
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False, output="ba")
    return lfilter(b, a, data, axis=0).astype(np.float32)


class ButterFilter(VerboseTransformationBlock):
    """Butter filter for eeg signals."""

    def custom_transform(self, data: XData, **kwargs: Any) -> XData:
        """Filter the eeg signals with a butter filter.

        :param data: The X data to transform, as tuple (eeg, spec, meta)
        :return: The transformed data
        """
        eeg = data.eeg
        if eeg is None:
            raise ValueError("No EEG data to transform")
        for key in tqdm(eeg.keys(), desc="Butter Filtering EEG data"):
            eeg[key] = eeg[key].apply(butter_lowpass_filter)
        return data
