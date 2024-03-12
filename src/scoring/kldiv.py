"""KLDiv scorer class."""
from typing import Any

import numpy as np
import torch
from torch.nn import KLDivLoss

from src.scoring.scorer import Scorer


class KLDiv(Scorer):
    """Abstract scorer class from which other scorers inherit from."""

    def __init__(self, name: str = "KLDiv") -> None:
        """Initialize the scorer with a name."""
        super().__init__(name)

    def __call__(self, y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]) -> float:
        """Calculate the Kullback-Leibler divergence between two probability distributions.

        :param y_true: The true labels.
        :param y_pred: The predicted labels.
        :return: The Kullback-Leibler divergence between the two probability distributions.
        """
        # For each row, make sure the sum of the labels is 1
        y_true = y_true / y_true.sum(axis=1)[:, None]

        # Convert both to torch tensors
        y_pred = torch.tensor(y_pred)  # type: ignore[assignment]
        target = torch.tensor(y_true)

        # Calculate the KLDivLoss
        criterion = KLDivLoss(reduction="batchmean")
        return criterion(torch.log(torch.clamp(y_pred, min=10**-15, max=1 - 10**-15)), target)  # type: ignore[call-overload]

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return self.name