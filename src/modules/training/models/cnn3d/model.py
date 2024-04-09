"""Model for 5D EEG data classification."""

import torch
from torch import nn

from src.modules.training.models.cnn3d.convgru import ConvGRU
from src.modules.training.models.cnn3d.convgru2 import ConvGRU2
from src.modules.training.models.cnn3d.efficientnet import EfficientNet3D
from src.modules.training.models.cnn3d.resnet import resnet18, resnet34, resnet50, resnet101
from src.utils.to_3d_grid import to_3d_grid_vectorized


class Model(nn.Module):
    """Model for 5D EEG data classification.

    Input:
        X: (n_samples, n_channels, n_timepoints, n_width, n_height)
        Y: (n_samples, n_classes)

    Output:
        out: (n_samples, n_classes)

    """

    def __init__(self, in_channels: int, out_channels: int, model_type: str, image_size: int, duration: int) -> None:
        """Initialize the Timm model.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param model_name: The model to use.
        :param image_size: The image size.
        :param duration: The duration.
        """
        super(Model, self).__init__()  # noqa: UP008
        self.model_type = model_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size
        self.duration = duration
        
        self.step_size = 1000

        if "efficientnet" in model_type:
            self.model = EfficientNet3D.from_name(self.model_type, override_params={"num_classes": self.out_channels}, in_channels=self.in_channels)
        elif "resnet" in model_type:
            if model_type == "resnet18":
                self.model = resnet18(num_classes=self.out_channels, in_channels=self.in_channels, sample_size=self.image_size, sample_duration=self.step_size)
            if model_type == "resnet34":
                self.model = resnet34(num_classes=self.out_channels, in_channels=self.in_channels, sample_size=self.image_size, sample_duration=self.step_size)
            if model_type == "resnet50":
                self.model = resnet50(num_classes=self.out_channels, in_channels=self.in_channels, sample_size=self.image_size, sample_duration=self.step_size)
            if model_type == "resnet101":
                self.model = resnet101(num_classes=self.out_channels, in_channels=self.in_channels, sample_size=self.image_size, sample_duration=self.step_size)
        elif "convgru" in model_type:
            self.model = ConvGRU(input_size=(self.image_size, self.image_size), input_dim=self.in_channels, hidden_dim=64,  kernel_size=(3, 3), num_layers=1, dtype=torch.cuda.FloatTensor,
                                 batch_first=True)


        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Timm model.

        :param x: The input data.
        :return: The output data.
        """
        # Convert to 5D
        x = to_3d_grid_vectorized(x, 9, 9)
        import torch.nn.functional as F

        if x.shape[-1] != self.image_size:
            x = F.interpolate(x, size=(self.duration, self.image_size, self.image_size), mode="trilinear", align_corners=False)

        if "efficientnet" in self.model_type or "resnet" in self.model_type:

            #Take the middle ~2 seconds.
            x = self.model(x)
            # steps = self.duration / self.step_size
            #
            # #Average the output of the steps
            # for i in range(int(steps)):
            #     if i == 0:
            #         res = self.model(x[:, :, i * self.step_size: (i + 1) * self.step_size, :, :])
            #     else:
            #         res += self.model(x[:, :, i * self.step_size: (i + 1) * self.step_size, :, :])
            # res = res / steps
            # return res



        if "convgru" in self.model_type:
            x, _ = self.model(x)
            x = x[0]
            x = x[:, -1, :, :]
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)


        return x
