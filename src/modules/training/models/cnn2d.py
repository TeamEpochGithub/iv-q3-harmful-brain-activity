"""CNN2D model for 2D spectrogram classification, baseline model."""
import torch
from torch import nn


class CNN2D(nn.Module):
    """CNN2D model for 2D spectrogram classification, baseline model.

    Input:
        X: (n_samples, n_channel, n_width, n_height)
        Y: (n_samples)

    Output:
        out: (n_samples)

    """

    def __init__(self, in_channels: int, out_channels: int, model: nn.Module) -> None:
        """Initialize the CNN2D model.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param model: The model to use.
        """
        super(CNN2D, self).__init__()  # noqa: UP008

        model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # Modify the classifier to output 6 classes
        num_features = model.classifier[-1].in_features  # Get the number of inputs for the last layer
        model.classifier[-1] = nn.Linear(num_features, out_channels)  # Replace the last layer

        self.model = model
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Add a softmax layer to the end
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CNN2D model.

        :param x: The input data.
        :return: The output data.
        """
        x = self.model(x)
        return self.softmax(x)
