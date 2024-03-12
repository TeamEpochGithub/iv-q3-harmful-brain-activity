"""Contains the SumToOne transformation block that makes sure the labels for each eeg / spectrogram add up to 1."""
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock


@dataclass
class SumToOne(VerboseTransformationBlock):
    """Sum to One class."""

    def custom_transform(self, data: np.ndarray[Any, Any], **kwargs: Any) -> np.ndarray[Any, Any]:
        """Sum the labels to one.

        :param data: The y data to transform which is a 2D array (n_samples, n_experts).
        :return: The transformed data.
        """
        # For each row, make sure the sum of the labels is 1
        logging.info("Summing labels to one...")
        data = data / data.sum(axis=1)[:, None]
        logging.info("Labels summed to one!")
        return data
