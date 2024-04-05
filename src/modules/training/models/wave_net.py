"""WaveNet model for time series classification."""
import torch
from torch import Tensor, nn


class WaveBlock(nn.Module):
    """WaveBlock for WaveNet model.

    :param in_channels: Number of in channels
    :param out_channels: Number of out channels
    :param dilation_rates: Number of dilations
    :param kernel_size: The kernel size of convolutions
    """

    def __init__(self, in_channels: int, out_channels: int, dilation_rate: int, kernel_size: int = 3) -> None:
        """WaveNet building block.

        :param in_channels: number of input channels.
        :param out_channels: number of output channels.
        :param dilation_rate: how many levels of dilations are used.
        :param kernel_size: size of the convolving kernel.
        """
        super(WaveBlock, self).__init__()  # noqa: UP008
        self.num_rates = dilation_rate
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True))

        dilation_rates = [2**i for i in range(dilation_rate)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate * (kernel_size - 1)) / 2), dilation=dilation_rate),
            )
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate * (kernel_size - 1)) / 2), dilation=dilation_rate),
            )
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=True))

        for i in range(len(self.convs)):
            nn.init.xavier_uniform_(self.convs[i].weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(self.convs[i].bias)

        for i in range(len(self.filter_convs)):
            nn.init.xavier_uniform_(self.filter_convs[i].weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(self.filter_convs[i].bias)

        for i in range(len(self.gate_convs)):
            nn.init.xavier_uniform_(self.gate_convs[i].weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(self.gate_convs[i].bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of WaveBlock.

        :param x: Input data
        :return: Forwarded data
        """
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            tanh_out = torch.tanh(self.filter_convs[i](x))
            sigmoid_out = torch.sigmoid(self.gate_convs[i](x))
            x = tanh_out * sigmoid_out
            x = self.convs[i + 1](x)
            res = res + x
        return res


class WaveNetClassifier(nn.Module):
    """WaveNetClassifier for time series classification.

    :param input_channels: Number of input channels
    :param n_classes: Number of classes to predict
    """

    def __init__(self, input_channels: int = 3, n_classes: int = 3) -> None:
        """Initialize WaveNetClassifier.

        :param input_channels: Number of input channels
        :param n_classes: Number of classes to predict
        """
        super().__init__()

        self.wave_block1 = WaveBlock(input_channels, 16, 2)
        self.wave_block2 = WaveBlock(16, 32, 1)
        # self.wave_block3 = Wave_Block(32, 64, 4)
        # self.wave_block4 = Wave_Block(64, 128, 1)

        self.GRU = nn.GRU(input_size=32, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)

        self.fc = nn.Linear(128, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of WaveNet classifier.

        :param x: Input data
        :return: Output tensor
        """
        x = self.wave_block1(x)
        x = self.wave_block2(x)

        x = x.permute(0, 2, 1)
        x, _ = self.GRU(x)
        x = x[:, -1, :]

        x = self.dropout(x)
        return self.fc(x)
