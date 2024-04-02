"""Module for applying Normalize."""
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from src.modules.training.verbose_training_block import VerboseTrainingBlock


class Normalize(VerboseTrainingBlock):
    """An example training block."""

    def custom_train(self, x: torch.Tensor, y: npt.NDArray[np.float32], **train_args: Any) -> tuple[Any, Any]:
        """Train the model.

        :param x: The input data
        :param y: The target data
        :return: The predictions and the target data
        """
        return self.custom_predict(x), y

    def custom_predict(self, x: torch.Tensor, **pred_args: Any) -> npt.NDArray[np.float32]:
        """Apply the Normalize function.

        :param x: The predictions.
        :return: The Normalizeed predictions.
        """
        self.log_to_terminal("Applying Normalize to the predictions...")
        # First min max scale the data
        x_minmax = (x - torch.min(x, dim=1).values[:, None]) / (torch.max(x, dim=1).values - torch.min(x, dim=1).values)[:, None]  # noqa: PD011

        # Then normalize the data
        x_sum = torch.sum(x_minmax, dim=1)
        new_x = x_minmax / x_sum[:, None]
        self.log_to_terminal("Normalize applied to the predictions!")
        return new_x.numpy()
