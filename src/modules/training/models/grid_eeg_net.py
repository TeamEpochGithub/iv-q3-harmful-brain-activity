"""Model that will make a grid and use eeg_net as a backbone."""

from typing import Any

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

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        """Initialize the model."""
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 2, 3),
            nn.Conv2d(2, 4, 3),
            nn.Conv2d(4, 2, 3),
            nn.Conv2d(2, 1, 3),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 2, 5),
            nn.Conv2d(2, 1, 5),

        )
        self.conv3 = nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(1, 2, 7),
            nn.ZeroPad2d(1),
            nn.Conv2d(2, 1, 7),

        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(1, 1, 9),

        )
        if kwargs.get("residual", False):
            self.residual = True
            del kwargs["residual"]

        self.batch_norm = nn.BatchNorm1d(24)

        self.eeg_net = EEGNet(**kwargs)  # type: ignore[arg-type]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x_grid = to_3d_grid_vectorized(x, 9, 9).permute(0, 2, 1, 3, 4)
        batch_size, sequence_len, C, H, W = x_grid.size()
        # Reshape to combine batch and sequence dimensions
        x_grid = x_grid.view(batch_size * sequence_len, C, H, W)

        all_features = []
        # Apply convolutional layers
        for module in [self.conv1, self.conv2, self.conv3, self.conv4]:

            x_features = module(x_grid)

            # Output shape from conv layers, assuming (N, C', H', W')
            _, C_out, H_out, W_out = x_features.size()

            # Reshape back to separate batch and sequence dimensions
            x_features = x_features.view(batch_size, sequence_len, C_out, H_out, W_out).squeeze(-1).squeeze(-1)
            all_features.append(x_features)
        all_features = torch.cat(all_features, dim=2)

        # concat the inital features with the conv features
        if self.residual:
            all_features = torch.cat([x, all_features.permute(0, 2, 1)], dim=1)
        all_features = self.batch_norm(all_features)
        return self.eeg_net(all_features)

if __name__ == "__main__":
    
    import torch

    x = torch.zeros(32,20,2000)
    conv = nn.Sequential(
            nn.Conv2d(1, 2, 3),
            nn.Conv2d(2, 4, 3),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
        )
    x_grid = to_3d_grid_vectorized(x, 9, 9).permute(0, 2, 1, 3, 4)
    batch_size, sequence_len, C, H, W = x_grid.size()
    
    for i in range(sequence_len):
            x_features = conv(x_grid[:, i, :, :, :])
            if i == 0:
                x_features_all = x_features
            else:
                x_features_all = torch.cat((x_features_all, x_features), dim=2)

    x_grid = x_grid.view(batch_size * sequence_len, C, H, W)

    # Apply convolutional layers
    x_features = conv(x_grid)

    # Output shape from conv layers, assuming (N, C', H', W')
    _, C_out, H_out, W_out = x_features.size()

    # Reshape back to separate batch and sequence dimensions
    x_features = x_features.view(batch_size, sequence_len, C_out, H_out, W_out).squeeze(-1).squeeze(-1)

    print(x_features.permute(0, 2, 1).shape)
    print(x_features_all.squeeze(-1).shape)
    print(torch.allclose(x_features.permute(0, 2, 1), x_features_all.squeeze(-1)))