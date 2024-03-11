"""This class contains a wrapper of the KLDivLoss function from PyTorch. It is used to calculate the Kullback-Leibler divergence between two probability distributions."""

import torch
from torch import nn
from torch.nn import KLDivLoss


class CustomKLDivLoss(nn.Module):
    def __init__(self, reduction: str | None = "batchmean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        criterion = KLDivLoss(reduction="batchmean")
        return criterion(torch.log(torch.clamp(input, min=10**-15, max=1 - 10**-15)), target)
