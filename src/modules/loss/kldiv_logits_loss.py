from torch import nn
import torch

class KLDivLogitsLoss(nn.KLDivLoss):
    """Kullback-Leibler divergence loss for logits."""
    
    def __init__(self, *args, **kwargs):
        """Initialise the Kullback-Leibler divergence loss for logits."""
        super().__init__(*args, **kwargs)
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Kullback-Leibler divergence loss for logits.
        
        :param input: The input tensor
        :param target: The target tensor
        :return: The loss
        """
        return super().forward(torch.log(input), target)