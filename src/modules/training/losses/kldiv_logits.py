"""KL Div class contains a wrapper of the KLDivLoss function from PyTorch. It is used to calculate the Kullback-Leibler divergence between two probability distributions."""

import torch
from torch import nn
from torch.nn import KLDivLoss


class CustomKLDivLogitsLoss(nn.Module):
    """Custom KLDivLoss class with log_softmax. Wrapper for the KLDivLoss function from PyTorch, takes in raw predictions."""

    def __init__(self, reduction: str = "batchmean", weighted: bool = False) -> None:  # noqa: FBT001, FBT002
        """Initialize the CustomKLDivLoss class.

        :param reduction: The reduction method to use. Default is "batchmean".
        :parapm
        """
        super().__init__()
        self.reduction = reduction
        self.weighted = weighted

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CustomKLDivLoss class."""
        criterion = KLDivLoss(reduction=self.reduction)

        # Multiply factor for each element in the batch based on the sum of the targets
        factor = target.sum(dim=1, keepdim=True)
        # For each row, make sure the sum of the labels is 1
        target = target / factor

        # Calculate the KLDivLoss

        # Apply a log softmax to the predictions
        pred = torch.nn.functional.log_softmax(pred, dim=1)

        # Calculate the KLDivLoss
        loss = criterion(pred, target)

        if self.weighted:
            loss = loss * factor.mean()
        return loss

    def __repr__(self) -> str:
        """Return representation of the CustomKLDivLoss class."""
        return f"CustomKLDivLogitsLoss(reduction={self.reduction}, weighted={self.weighted})"
