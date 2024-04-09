"""A head that concatenates the deep features with the manual features and passes them through a linear layer."""
import torch
from torch import nn


class FeatureConcatHead(nn.Module):
    """A head that concatenates the deep features with the manual features and passes them through a linear layer."""

    def __init__(self, model: nn.Module, in_features_deep: int, in_features_manual: int, hidden_size: int, num_classes: int) -> None:
        """Initialize the head.

        :param model: The model that extracts the deep features
        :param in_features_deep: The number of deep features
        :param in_features_manual: The number of manual features
        :param hidden_size: The size of the hidden layer
        :param num_classes: The number of classes
        """
        super().__init__()
        self.model = model
        self.fc1 = nn.Linear(in_features_deep + in_features_manual, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x_deep: torch.Tensor, x_manual: torch.Tensor) -> torch.Tensor:
        """Forward pass through the head.

        :param x_deep: The deep features
        :param x_manual: The manual features
        :return: The output of the head
        """
        x_deep = self.model(x_deep)
        x = torch.cat([x_deep, x_manual], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return f"FeatureConcatHead(model={self.model}, fc1={self.fc1}, fc2={self.fc2})"
