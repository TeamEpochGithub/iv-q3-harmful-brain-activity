"""Module for applying softmax."""
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from src.modules.training.verbose_training_block import VerboseTrainingBlock


class Softmax(VerboseTrainingBlock):
    """An example training block."""

    def custom_train(self, x: torch.Tensor, y: npt.NDArray[np.float32], **train_args: Any) -> tuple[Any, Any]:
        """Train the model.

        :param x: The input data
        :param y: The target data
        :return: The predictions and the target data
        """
        return self.custom_predict(x), y

    def custom_predict(self, x: torch.Tensor, **pred_args: Any) -> npt.NDArray[np.float32]:
        """Apply the softmax function.

        :param x: The predictions.
        :return: The softmaxed predictions.
        """
        self.log_to_terminal("Applying softmax to the predictions...")
        new_x = torch.softmax(x, dim=1).numpy()
        self.log_to_terminal("Softmax applied to the predictions!")
        return new_x
