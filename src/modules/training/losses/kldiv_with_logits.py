"""KLDivLossWithLogits module for loss with logits."""
from torch import Tensor, nn


class KLDivLossWithLogits(nn.KLDivLoss):
    """KLDivLossWithLogits loss class."""

    def __init__(self) -> None:
        """Initialize KLDivLossWithLogits class."""
        super().__init__(reduction="batchmean")

    def forward(self, y: Tensor, t: Tensor) -> Tensor:
        """Forward function of KLDivLossWithLogits.

        :param y: Predictions
        :param t: Labels
        :return: Loss tensor
        """
        y = nn.functional.log_softmax(y, dim=1)
        return super().forward(y, t)
