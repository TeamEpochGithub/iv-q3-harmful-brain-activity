"""Gaussian target transformation block."""
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock


@dataclass
class GaussianTarget(VerboseTransformationBlock):
    """Create gaussian labels."""

    labels_length: int = 10000  # in samples
    sigma: int = 2  # in seconds

    def custom_transform(self, data: np.ndarray[Any, Any], **kwargs: Any) -> np.ndarray[Any, Any]:
        """Make gaussian curves for the labels.

        :param data: The y data to transform which is a 2D array (n_samples, n_experts).
        :return: The transformed data.
        """
        logging.info("Creating Gaussian labels...")
        # create a sequence that is 10000 samples long and has a gaussian curve with amplitude 1 at the center ith sigma 400
        num_samples = self.labels_length
        SECOND_OFFSET = 25
        time = np.linspace(-SECOND_OFFSET, SECOND_OFFSET, num_samples, dtype=np.float32)
        # Original Gaussian curve
        original_curve = np.exp(-(time**2) / (2 * self.sigma**2), dtype=np.float32)
        # Add 3rd dimension to the data
        labels_reshaped = data.copy().astype(np.float32)[:, :, np.newaxis]
        # Repeat the original curve for each sample in the gaussian
        labels_reshaped = labels_reshaped.repeat(num_samples, axis=2)
        # Multiply the original curve by the labels in-place
        labels_reshaped *= original_curve
        # put channel last
        labels_reshaped = labels_reshaped.transpose(0, 2, 1)
        logging.info("Gaussian labels created!")
        return labels_reshaped
