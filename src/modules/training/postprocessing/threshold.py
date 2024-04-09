"""Module for applying threshold."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from src.modules.training.verbose_training_block import VerboseTrainingBlock


@dataclass
class Threshold(VerboseTrainingBlock):
    """An example training block."""

    threshold: float = 0.05

    def custom_train(self, x: npt.NDArray[np.float32], y: npt.NDArray[np.float32], **train_args: Any) -> tuple[Any, Any]:
        """Train the model.

        :param x: The input data
        :param y: The target data
        :return: The predictions and the target data
        """
        return self.custom_predict(x), y

    def custom_predict(self, x: npt.NDArray[np.float32], **pred_args: Any) -> npt.NDArray[np.float32]:
        """Apply the threshold function.

        :param x: The predictions.
        :return: The thresholded predictions.
        """
        # Check if each prediction sums up to 1
        if np.sum(x, axis=1).all() != 1:
            raise ValueError("Predictions do not sum up to 1. Please apply a softmax function first.")

        self.log_to_terminal(f"Applying Threshold {self.threshold} to the predictions...")

        # If the value is below the threshold, set it to 0
        x[x < self.threshold] = 0

        # Make sure that the resulting predictions sum up to 1
        x /= np.sum(x, axis=1, keepdims=True)

        if np.sum(x, axis=1).all() != 1:
            raise ValueError("Something went wrong.")

        self.log_to_terminal("Threshold applied to the predictions!")
        return x
