"""Torchvision model for 2D spectrogram classification, baseline model."""
import timm
import torch
from torch import nn

from src.logging_utils.logger import logger


class Torchvision(nn.Module):
    """Torchvision model for 2D spectrogram classification, baseline model.

    Input:
        X: (n_samples, n_channel, n_width, n_height)
        Y: (n_samples)

    Output:
        out: (n_samples)

    """

    def __init__(self, in_channels: int, out_channels: int, model: nn.Module) -> None:
        """Initialize the Torchvision model.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param model: The model to use.
        """
        super(Torchvision, self).__init__()  # noqa: UP008
        self.model = model
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.setup_model()

        self.model = timm.create_model("efficientnet_b0", pretrained=True, in_chans=in_channels, num_classes=out_channels)

        # Add a softmax layer to the end
        self.softmax = nn.Softmax(dim=-1)

    def setup_model(self) -> None:
        """Set up the first layer and last_layer based on the models architecture."""
        match self.model.__class__.__name__:
            case "EfficientNet":
                first = self.model.features[0][0]
                new_layer = nn.Conv2d(self.in_channels, first.out_channels, kernel_size=first.kernel_size, stride=first.stride, padding=first.padding, bias=False)
                self.model.features[0][0] = new_layer
                num_features = self.model.classifier[-1].in_features  # Get the number of inputs for the last layer
                self.model.classifier[-1] = nn.Linear(num_features, self.out_channels)  # Replace the last layer
            case "VGG":
                first = self.model.features[0]
                new_layer = nn.Conv2d(self.in_channels, first.out_channels, kernel_size=first.kernel_size, stride=first.stride, padding=first.padding, bias=False)
                self.model.features[0] = new_layer
                num_features = self.model.classifier[-1].in_features  # Get the number of inputs for the last layer
                self.model.classifier[-1] = nn.Linear(num_features, self.out_channels)  # Replace the last layer
            case "ResNet":
                first = self.model.conv1
                new_layer = nn.Conv2d(self.in_channels, first.out_channels, kernel_size=first.kernel_size, stride=first.stride, padding=first.padding, bias=False)
                self.model.conv1 = new_layer
                # Replace the last layer
                num_features = self.model.fc.in_features
                self.model.fc = nn.Linear(num_features, self.out_channels)
            case _:
                logger.warning("Model not fully implemented yet.. Might crash, reverting to baseline implementation.")
                first = self.model.features[0]
                new_layer = nn.Conv2d(self.in_channels, first.out_channels, kernel_size=first.kernel_size, stride=first.stride, padding=first.padding, bias=False)
                self.model.features[0] = new_layer
                num_features = self.model.classifier[-1].in_features  # Get the number of inputs for the last layer
                self.model.classifier[-1] = nn.Linear(num_features, self.out_channels)  # Replace the last layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Torchvision model.

        :param x: The input data.
        :return: The output data.
        """
        x = self.model(x)
        return self.softmax(x)
