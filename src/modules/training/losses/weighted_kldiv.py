"""KL Div class contains a wrapper of the KLDivLoss function from PyTorch. It is used to calculate the Kullback-Leibler divergence between two probability distributions."""

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import CrossEntropyLoss


@dataclass
class WeightedKLDivLoss(nn.Module):
    """A KLdiv implementation using CE and logit space. This allows to use class weigths in the loss function."""

    class_weights: Iterable[float] = (1, 1, 1, 1, 1, 1)

    def __post_init__(self) -> None:
        """Post initialization function for the WeightedKLDivLoss class."""
        # make sure the sum of the weigths is equal tothe number of classes. In this case 6
        self.class_weights = [6 * weight / sum(self.class_weights) for weight in self.class_weights]
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of the WeightedKLDivLoss class.

        :param pred: The predicted values.
        :param target: The target values.
        :return: The loss value.
        """
        criterion = CrossEntropyLoss(weight=torch.tensor(self.class_weights).to(target.device))

        # Calculate the KLDivLoss
        return criterion(pred, target) + (torch.nan_to_num(torch.log(target), nan=0) * target).sum(dim=1).mean()
