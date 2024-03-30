"""Model for 5D EEG data classification."""
import time

import torch
from torch import nn

from src.modules.training.models.cnn3d.efficientnet import EfficientNet3D
from src.utils.to_3d_grid import to_3d_grid_vectorized


class Model(nn.Module):
    """Model for 5D EEG data classification.

    Input:
        X: (n_samples, n_channels, n_timepoints, n_width, n_height)
        Y: (n_samples, n_classes)

    Output:
        out: (n_samples, n_classes)

    """

    def __init__(self, in_channels: int, out_channels: int, model_type: str) -> None:
        """Initialize the Timm model.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param model_name: The model to use.
        """
        super(Model, self).__init__()  # noqa: UP008
        self.model_type = model_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = EfficientNet3D.from_name(self.model_type, override_params={'num_classes': self.out_channels}, in_channels=self.in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Timm model.

        :param x: The input data.
        :return: The output data.
        """
        # Convert to 5D
        start = time.time()
        x = to_3d_grid_vectorized(x, 9, 9)
        import torch.nn.functional as F
        # Interpolate to 32x32
        x = F.interpolate(x, size=(1000, 64, 64), mode='trilinear', align_corners=False)
        return self.model(x)














