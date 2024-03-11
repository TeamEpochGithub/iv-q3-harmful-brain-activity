"""CNN1D model for 1D signal classification, baseline model."""
import torch
from torch import nn

from src.logging_utils.logger import logger


class CNN1D(nn.Module):
    """CNN1D model for 1D signal classification, baseline model.

    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    Output:
        out: (n_samples)

    Parameters
    ----------
        n_classes: number of classes.
    """

    def __init__(self, in_channels: int, out_channels: int, n_len_seg: int, n_classes: int, verbose: bool = False) -> None:  # noqa: FBT001, FBT002
        """Initialize the CNN1D model.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param n_len_seg: The length of the segment.
        :param n_classes: The number of classes.
        :param verbose: Whether to print out the shape of the data at each step.
        """
        super(CNN1D, self).__init__()  # noqa: UP008

        self.n_len_seg = n_len_seg
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.verbose = verbose

        # (batch, channels, length)
        self.cnn = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=16, stride=2)
        # (batch, channels, length)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.out_channels, nhead=8, dim_feedforward=128, dropout=0.5)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.dense = nn.Linear(out_channels, n_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CNN1D model.

        :param x: The input data.
        :return: The output data.
        """
        self.n_channel, self.n_length = x.shape[-1], x.shape[-2]
        if self.n_length % self.n_len_seg != 0:
            raise ValueError("Input n_length should divided by n_len_seg")
        self.n_seg = self.n_length // self.n_len_seg

        out = x

        if self.verbose:
            logger.info(out.shape)
        # (n_samples, n_length, n_channel) -> (n_samples*n_seg, n_len_seg, n_channel)
        out = out.view(-1, self.n_len_seg, self.n_channel)
        if self.verbose:
            logger.info(out.shape)
        # (n_samples*n_seg, n_len_seg, n_channel) -> (n_samples*n_seg, n_channel, n_len_seg)
        out = out.permute(0, 2, 1)
        # cnn
        out = self.cnn(out)
        # global avg, (n_samples*n_seg, out_channels)
        out = out.mean(-1)
        if self.verbose:
            logger.info(out.shape)
        out = out.view(-1, self.n_seg, self.out_channels)
        out = self.transformer_encoder(out)
        if self.verbose:
            logger.info(out.shape)
        out = out.mean(-2)
        out = self.dense(out)

        return self.softmax(out)
