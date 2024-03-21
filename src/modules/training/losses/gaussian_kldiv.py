"""KL Div class contains a wrapper of the KLDivLoss function from PyTorch. It is used to calculate the Kullback-Leibler divergence between two probability distributions."""

import torch
from torch import nn
from torch.nn import KLDivLoss


class GaussianKLDivLoss(nn.Module):
    """KLDivLoss class. Used for gaussian predictions without a softmax layer."""

    def __init__(self, reduction: str = "batchmean") -> None:
        """Initialize the class.

        :param reduction: The reduction method to use. Default is "batchmean".
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Do not use softmax outputs forthis loss function. Forward pass of the CustomKLDivLoss class.

        :param pred: The predicted values.
        :param target: The target values.
        :return: The loss value.
        """
        criterion = KLDivLoss(reduction=self.reduction)

        # Calculate the KLDivLoss

        return criterion(torch.log_softmax(torch.clamp(pred, min=10**-15, max=1 - 10**-15), dim=1), torch.softmax(target, dim=1))
