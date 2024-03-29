"""Model that will make a grid and use eeg_net as a backbone."""

import torch
from torch import nn

from src.modules.training.models.eeg_net import EEGNet
from src.utils.to_3d_grid import to_3d_grid_vectorized


class GridEEGNet(nn.Module):
    """Model that will make a grid and use eeg_net as a backbone."""

    # num_classes: 6
    # in_channels: 9
    # fixed_kernel_size: 5
    # #linear_layer_features: 448
    # #linear_layer_features: 352 # Half Signal = 5_000
    # linear_layer_features: 304 # 1/4 1/5 1/6 Signal = 2_000
    # #linear_layer_features: 280 # 1/10 Signal = 1_000
    # kernels: [3,5,7,9,11]
    # dropout: 0.1

    def __init__(self, **kwargs) -> None:
        """Initialize the model."""
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 2, 3),
            nn.Conv2d(2, 4, 3),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
        )
        self.eeg_net = EEGNet(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x_grid = to_3d_grid_vectorized(x, 9, 9).permute(0, 2, 1, 3, 4)
        batch_size, sequence_len, C, H, W = x_grid.size()
        # Reshape to combine batch and sequence dimensions
        x_grid = x_grid.view(batch_size * sequence_len, C, H, W)

        # Apply convolutional layers
        x_features = self.conv(x_grid)

        # Output shape from conv layers, assuming (N, C', H', W')
        _, C_out, H_out, W_out = x_features.size()

        # Reshape back to separate batch and sequence dimensions
        x_features = x_features.view(batch_size, sequence_len, C_out, H_out, W_out).squeeze(-1).squeeze(-1)
        concat_features = torch.cat([x_features.permute(0, 2, 1), x], dim=1)
        return self.eeg_net(concat_features)


if __name__ == "__main__":
    test_in = torch.zeros(32, 1, 9, 9)

    model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.Conv2d(2, 4, 3), nn.ReLU(), nn.Conv2d(4, 8, 3), nn.Conv2d(8, 16, 3), nn.ReLU())
    out = model(test_in)
    print(out.shape)
