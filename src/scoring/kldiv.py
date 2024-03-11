""" KLDiv scorer class. """
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
        # Convert both to torch tensors
        input = torch.tensor(y_pred)
        target = torch.tensor(y_true)
        criterion = KLDivLoss(reduction='batchmean')
        return criterion(torch.log(torch.clamp(input, min=10 ** -15, max=1 - 10 ** -15)), target)

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return self.name
