"""Module containing the FeatureOnly class."""
import torch
from torch import nn


class FeatureOnly(nn.Module):
    """A class for creating a feature only model."""

    def __init__(self, layer_sizes: list[int]) -> None:
        """Construct FeatureOnly class.

        :param layer_sizes: A list of integers representing the layer sizes.
        """
        super().__init__()
        self.activation = nn.SiLU()
        self.layers = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x_manual: torch.Tensor) -> torch.Tensor:
        """Forward method for the FeatureOnly class.

        :param x_manual: The input tensor for the manual features.
        :return: The output tensor after passing through the layers.
        """
        x = x_manual
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
