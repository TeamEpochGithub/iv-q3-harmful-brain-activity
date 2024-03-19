"""KL Div class contains a wrapper of the KLDivLoss function from PyTorch. It is used to calculate the Kullback-Leibler divergence between two probability distributions."""

import torch
from torch import nn
from torch.nn import KLDivLoss


class CustomKLDivLoss(nn.Module):
    """Custom KLDivLoss class. Wrapper for the KLDivLoss function from PyTorch."""

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
        loss = criterion(torch.log(torch.clamp(pred, min=10**-15, max=1 - 10**-15)), target)

        if self.weighted:
            loss = loss * factor.mean()
        return loss
