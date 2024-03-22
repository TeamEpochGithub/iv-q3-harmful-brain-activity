"""Module containing EEGNet class."""
import torch
from torch import Tensor, nn


class ResNet1DBlock(nn.Module):
    """ResNet 1D block.

    :param in_channels: Number of in_channels
    :param out_channels: Number of out_channels
    :param kernel_size: Size of kernels
    :param stride: Stride size
    :param padding: Padding size
    :param downsampling: Factor to downsample with
    :param dilation: Dilation factor
    :param groups: Number of groups
    :param dropout: Dropout rate
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        downsampling: nn.Module,
        dilation: int = 1,
        groups: int = 1,
        dropout: float = 0.0,
    ) -> None:
        """Initialize ResNet 1D block.

        :param in_channels: Number of in_channels
        :param out_channels: Number of out_channels
        :param kernel_size: Size of kernels
        :param stride: Stride size
        :param padding: Padding size
        :param downsampling: Factor to downsample with
        :param dilation: Dilation factor
        :param groups: Number of groups
        :param dropout: Dropout rate
        """
        super(ResNet1DBlock, self).__init__()  # noqa: UP008

        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        # self.relu = nn.ReLU(inplace=False)
        # self.relu_1 = nn.PReLU()
        # self.relu_2 = nn.PReLU()
        self.relu_1 = nn.Hardswish()
        self.relu_2 = nn.Hardswish()

        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        self.maxpool = nn.MaxPool1d(
            kernel_size=2,
            stride=2,
            padding=0,
            dilation=dilation,
        )
        self.downsampling = downsampling

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for ResNet1DBlock.

        :param x: Input data
        :return: Transformed data
        """
        identity = x

        out = self.bn1(x)
        out = self.relu_1(out)
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu_2(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = self.maxpool(out)
        identity = self.downsampling(x)

        out += identity
        return out


class EEGNet(nn.Module):
    """EEGNet.

    :param kernels: List of kernel sizes
    :param in_channels: Number of in_channels
    :param fixed_kernel_size: The fixed_kernel_size
    :param num_classes: Output classes of the model
    :param linear_layer_features: Number of features in the linear layer
    :param dilation: Dilation factor
    :param groups: Number of groups
    """

    def __init__(
        self,
        kernels: list[int],
        in_channels: int,
        fixed_kernel_size: int,
        num_classes: int,
        linear_layer_features: int,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        """Initialize EEGNet.

        :param kernels: List of kernel sizes
        :param in_channels: Number of in_channels
        :param fixed_kernel_size: The fixed_kernel_size
        :param num_classes: Output classes of the model
        :param linear_layer_features: Number of features in the linear layer
        :param dilation: Dilation factor
        :param groups: Number of groups
        """
        super(EEGNet, self).__init__()  # noqa: UP008
        self.kernels = kernels
        self.planes = 24
        self.parallel_conv = nn.ModuleList()
        self.in_channels = in_channels

        for _, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.planes,
                kernel_size=(kernel_size),
                stride=1,
                padding=0,
                dilation=dilation,
                groups=groups,
                bias=False,
            )
            self.parallel_conv.append(sep_conv)

        self.bn1 = nn.BatchNorm1d(num_features=self.planes)
        # self.relu = nn.ReLU(inplace=False)
        # self.relu_1 = nn.ReLU()
        # self.relu_2 = nn.ReLU()
        self.relu_1 = nn.SiLU()
        self.relu_2 = nn.SiLU()

        self.conv1 = nn.Conv1d(
            in_channels=self.planes,
            out_channels=self.planes,
            kernel_size=fixed_kernel_size,
            stride=2,
            padding=2,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        self.block = self._make_resnet_layer(
            kernel_size=fixed_kernel_size,
            stride=1,
            dilation=dilation,
            groups=groups,
            padding=fixed_kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm1d(num_features=self.planes)
        self.avgpool = nn.AvgPool1d(kernel_size=6, stride=6, padding=2)

        self.rnn = nn.GRU(
            input_size=self.in_channels,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            # dropout=0.2,
        )

        self.fc = nn.Linear(in_features=linear_layer_features, out_features=num_classes)
        self.softmax = nn.Softmax()

    def _make_resnet_layer(
        self,
        kernel_size: int,
        stride: int,
        dilation: int = 1,
        groups: int = 1,
        blocks: int = 9,
        padding: int = 0,
        dropout: float = 0.0,
    ) -> nn.Module:
        """Make resnet layer for EEGNet model.

        :param kernel_size: Kernel size of resnet layer
        :param stride: Stride length
        :param dilation: Dilation factor
        :param groups: Number of groups
        :param blocks: Number of blocks
        :param padding: Amount of padding
        :param dropout: Dropout rate
        """
        layers = []

        for _ in range(blocks):
            downsampling = nn.Sequential(
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            )
            layers.append(
                ResNet1DBlock(
                    in_channels=self.planes,
                    out_channels=self.planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    downsampling=downsampling,
                    dilation=dilation,
                    groups=groups,
                    dropout=dropout,
                ),
            )
        return nn.Sequential(*layers)

    def extract_features(self, x: Tensor) -> Tensor:
        """Extract features from input data function.

        :param x: Input data
        :return: Extracted data
        """
        out_sep = []
        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)

        out = torch.cat(out_sep, dim=2)
        out = self.bn1(out)
        out = self.relu_1(out)
        out = self.conv1(out)

        out = self.block(out)
        out = self.bn2(out)
        out = self.relu_2(out)
        out = self.avgpool(out)

        out = out.reshape(out.shape[0], -1)
        rnn_out, _ = self.rnn(x.permute(0, 2, 1))
        new_rnn_h = rnn_out[:, -1, :]

        return torch.cat([out, new_rnn_h], dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of EEGNet.

        :param x: Input data
        :return: Output tensor
        """
        new_out = self.extract_features(x)
        return self.fc(new_out)
