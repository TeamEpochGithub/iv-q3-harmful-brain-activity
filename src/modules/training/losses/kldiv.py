"""KL Div class contains a wrapper of the KLDivLoss function from PyTorch. It is used to calculate the Kullback-Leibler divergence between two probability distributions."""

import torch
from torch import nn
from torch.nn import KLDivLoss


class CustomKLDivLoss(nn.Module):
    """Custom KLDivLoss class. Wrapper for the KLDivLoss function from PyTorch."""

    def __init__(self, reduction: str = "batchmean") -> None:
        """Initialize the CustomKLDivLoss class.

        :param reduction: The reduction method to use. Default is "batchmean".
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CustomKLDivLoss class."""
        criterion = KLDivLoss(reduction=self.reduction)
        return criterion(torch.log(torch.clamp(pred, min=10**-15, max=1 - 10**-15)), target)
