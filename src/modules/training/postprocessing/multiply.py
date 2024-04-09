"""Module for applying Multiply."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from src.modules.training.verbose_training_block import VerboseTrainingBlock


@dataclass
class Multiply(VerboseTrainingBlock):
    """An example training block."""

    factor: float = 1

    def custom_train(self, x: torch.Tensor, y: npt.NDArray[np.float32], **train_args: Any) -> tuple[Any, Any]:
        """Train the model.

        :param x: The input data
        :param y: The target data
        :return: The predictions and the target data
        """
        return self.custom_predict(x), y

    def custom_predict(self, x: torch.Tensor, **pred_args: Any) -> npt.NDArray[np.float32]:
        """Apply the Multiply function.

        :param x: The predictions.
        :return: The multiplied predictions.
        """
        self.log_to_terminal(f"Applying Multiply to the predictions with factor {self.factor}...")
        return (x * self.factor).numpy()
