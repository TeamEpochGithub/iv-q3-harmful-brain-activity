"""KLDiv scorer class."""
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import KLDivLoss

from src.scoring.scorer import Scorer


class KLDiv(Scorer):
    """Abstract scorer class from which other scorers inherit from."""

    def __init__(self, name: str = "KLDiv") -> None:
        """Initialize the scorer with a name."""
        super().__init__(name)

    def __call__(self, y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any], **kwargs: dict[str, pd.DataFrame]) -> float:
        """Calculate the Kullback-Leibler divergence between two probability distributions.

        :param y_true: The true labels.
        :param y_pred: The predicted labels.
        :return: The Kullback-Leibler divergence between the two probability distributions.
        """
        # Normalize the true labels to be a probability distribution
        y_true = y_true / y_true.sum(axis=1)[:, None]

        # Get the metadata
        metadata = kwargs.get("metadata", None).reset_index(drop=True)  # type: ignore[union-attr]
        scores = metadata.groupby("eeg_id").apply(self.score_group, y_true=y_true, y_pred=y_pred)
        return scores.mean()

    def score_group(self, group: pd.DataFrame, y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]) -> float:
        """Calculate the Kullback-Leibler divergence between two probability distributions.

        :param group: The group to calculate the score for.
        :param y_true: The true labels.
        :param y_pred: The predicted labels.
        :return: The Kullback-Leibler divergence between the two probability distributions.
        """
        # Get the indices of the current group
        indices = group.index
        # Get the true and predicted labels for the current group
        y_true_group = y_true[indices]
        y_pred_group = y_pred[indices]

        # Calculate the KLDiv for the current group
        return self.calc_kldiv(y_true_group, y_pred_group)

    def calc_kldiv(self, y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]) -> float:
        """Calculate the Kullback-Leibler divergence between two probability distributions.

        :param y_true: The true labels.
        :param y_pred: The predicted labels.
        :return: The Kullback-Leibler divergence between the two probability distributions.
        """
        # Convert both to torch tensors
        y_pred = torch.tensor(y_pred)  # type: ignore[assignment]
        target = torch.tensor(y_true)

        # Calculate the KLDivLoss
        criterion = KLDivLoss(reduction="batchmean")
        return criterion(torch.log(torch.clamp(y_pred, min=10**-15, max=1 - 10**-15)), target)  # type: ignore[call-overload]

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return self.name
