"""KL Div class contains a wrapper of the KLDivLoss function from PyTorch. It is used to calculate the Kullback-Leibler divergence between two probability distributions."""

from dataclasses import dataclass
from typing import Iterable
import torch
from torch import nn
from torch.nn import CrossEntropyLoss


@dataclass
class WeightedKLDivLoss(nn.Module):
    """A KLdiv implementation using CE and logit space. This allows to use class weigths in the loss function."""

    class_weights: Iterable[float] = (1,1,1,1,1,1)

    def __post_init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Do not use softmax outputs for this loss function.
            Targets must sum to 1.
            Forward pass of the WeightedKLDivLoss class.
            
            :param pred: The predicted values.
            :param target: The target values.
            :return: The loss value."""

        criterion = CrossEntropyLoss(weight=torch.tensor(self.class_weights).to(target.device))

        # Calculate the KLDivLoss
        return criterion(pred, target) + (torch.nan_to_num(torch.log(target), nan=0) * target).sum(dim=1).mean()
