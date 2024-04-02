"""WaveNet model for eeg data"""
from torch import nn
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Union

N_CLASSES = 6


class ComplexModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.5):
        super(ComplexModel, self).__init__()

        self.convolutional = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=3, padding=1),  # Convolutional layer with 32 filters
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Max pooling layer
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Compute the output size after the convolutional layers
        conv_output_size = self._get_conv_output_size(input_size)

        self.classifier = nn.Sequential(
            nn.Linear(conv_output_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )


class DilatedInceptionWaveNet(nn.Module):
    """WaveNet architecture with dilated inception conv."""

    def __init__(self,) -> None:
        """Initialize dilated inception wave net"""
        super().__init__()

        kernel_size = [2, 3, 6, 7]

        # Model blocks
        self.wave_module = nn.Sequential(
            _WaveBlock(12, 1, 16, kernel_size, _DilatedInception),
            _WaveBlock(8, 16, 32, kernel_size, _DilatedInception),
            _WaveBlock(4, 32, 64, kernel_size, _DilatedInception),
            _WaveBlock(1, 64, 64, kernel_size, _DilatedInception),
        )
        self.output = ComplexModel(64*4, 128, N_CLASSES, dropout_rate=0.1)
        # self.output = nn.Sequential(
        #     nn.Linear(64 * 4, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, N_CLASSES)
        # )

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass.

        Shape:
            x: (B, L, C)
        """
        x = inputs
        bs, length, in_dim = x.shape
        x = x.transpose(1, 2).unsqueeze(dim=2)  # (B, C, N, L), N is redundant

        x_ll_1 = self.wave_module(x[:, 0:1, :])
        x_ll_2 = self.wave_module(x[:, 1:2, :])
        x_ll = (F.adaptive_avg_pool2d(x_ll_1, (1, 1)) + F.adaptive_avg_pool2d(x_ll_2, (1, 1))) / 2

        x_rl_1 = self.wave_module(x[:, 2:3, :])
        x_rl_2 = self.wave_module(x[:, 3:4, :])
        x_rl = (F.adaptive_avg_pool2d(x_rl_1, (1, 1)) + F.adaptive_avg_pool2d(x_rl_2, (1, 1))) / 2

        x_lp_1 = self.wave_module(x[:, 4:5, :])
        x_lp_2 = self.wave_module(x[:, 5:6, :])
        x_lp = (F.adaptive_avg_pool2d(x_lp_1, (1, 1)) + F.adaptive_avg_pool2d(x_lp_2, (1, 1))) / 2

        x_rp_1 = self.wave_module(x[:, 6:7, :])
        x_rp_2 = self.wave_module(x[:, 7:8, :])
        x_rp = (F.adaptive_avg_pool2d(x_rp_1, (1, 1)) + F.adaptive_avg_pool2d(x_rp_2, (1, 1))) / 2

        x = torch.cat([x_ll, x_rl, x_lp, x_rp], axis=1).reshape(bs, -1)
        output = self.output(x)

        return output


class _WaveBlock(nn.Module):
    """WaveNet block.

    Args:
        kernel_size: kernel size, pass a list of kernel sizes for
            inception
    """

    def __init__(
        self,
        n_layers: int,
        in_dim: int,
        h_dim: int,
        kernel_size: Union[int, list[int]],
        conv_module: Optional[type[nn.Module]] = None,
    ) -> None:
        super().__init__()

        self.n_layers = n_layers
        self.dilation_rates = [2**layer for layer in range(n_layers)]

        self.in_conv = nn.Conv2d(in_dim, h_dim, kernel_size=(1, 1))
        self.gated_tcns = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        for layer in range(n_layers):
            c_in, c_out = h_dim, h_dim
            self.gated_tcns.append(
                _GatedTCN(
                    in_dim=c_in,
                    h_dim=c_out,
                    kernel_size=kernel_size,
                    dilation_factor=self.dilation_rates[layer],
                    conv_module=conv_module,
                )
            )
            self.skip_convs.append(nn.Conv2d(h_dim, h_dim, kernel_size=(1, 1)))

        # Initialize parameters
        nn.init.xavier_uniform_(self.in_conv.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.in_conv.bias)
        for i in range(len(self.skip_convs)):
            nn.init.xavier_uniform_(self.skip_convs[i].weight, gain=nn.init.calculate_gain("relu"))
            nn.init.zeros_(self.skip_convs[i].bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Shape:
            x: (B, C, N, L), where C denotes in_dim
            x_skip: (B, C', N, L), where C' denotes h_dim
        """
        # Input convolution
        x = self.in_conv(x)

        x_skip = x
        for layer in range(self.n_layers):
            x = self.gated_tcns[layer](x)
            x = self.skip_convs[layer](x)

            # Skip-connection
            x_skip = x_skip + x

        return x_skip


class _GatedTCN(nn.Module):
    """Gated temporal convolution layer.

    Parameters:
        conv_module: customized convolution module
    """

    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        kernel_size: Union[int, list[int]],
        dilation_factor: int,
        dropout: Optional[float] = None,
        conv_module: Optional[type[nn.Module]] = None,
    ) -> None:
        super().__init__()

        # Model blocks
        if conv_module is None:
            self.filt = nn.Conv2d(
                in_channels=in_dim, out_channels=h_dim, kernel_size=(1, kernel_size), dilation=dilation_factor
            )
            self.gate = nn.Conv2d(
                in_channels=in_dim, out_channels=h_dim, kernel_size=(1, kernel_size), dilation=dilation_factor
            )
        else:
            self.filt = conv_module(
                in_channels=in_dim, out_channels=h_dim, kernel_size=kernel_size, dilation=dilation_factor
            )
            self.gate = conv_module(
                in_channels=in_dim, out_channels=h_dim, kernel_size=kernel_size, dilation=dilation_factor
            )

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters:
            x: input sequence

        Return:
            h: output sequence

        Shape:
            x: (B, C, N, L), where L denotes the input sequence length
            h: (B, h_dim, N, L')
        """
        x_filt = F.tanh(self.filt(x))
        x_gate = F.sigmoid(self.gate(x))
        h = x_filt * x_gate
        if self.dropout is not None:
            h = self.dropout(h)

        return h


class _DilatedInception(nn.Module):
    """Dilated inception layer.

    Note that `out_channels` will be split across #kernels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: list[int],
        dilation: int
    ) -> None:
        super().__init__()

        # Network parameters
        n_kernels = len(kernel_size)
        assert out_channels % n_kernels == 0, "`out_channels` must be divisible by #kernels."
        h_dim = out_channels // n_kernels

        # Model blocks
        self.convs = nn.ModuleList()
        for k in kernel_size:
            self.convs.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=h_dim,
                    kernel_size=(1, k),
                    padding="same",
                    dilation=dilation),
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.


        Parameters:
            x: input sequence

        Return:
            h: output sequence

        Shape:
            x: (B, C, N, L), where C = in_channels
            h: (B, C', N, L'), where C' = out_channels
        """
        x_convs = []
        for conv in self.convs:
            x_conv = conv(x)
            x_convs.append(x_conv)
        h = torch.cat(x_convs, dim=1)

        return h
