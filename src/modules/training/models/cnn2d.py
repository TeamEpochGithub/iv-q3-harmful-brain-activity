"""CNN2D model for 2D spectrogram classification, baseline model."""
import torch
from torch import nn

from src.logging_utils.logger import logger


class CNN2D(nn.Module):
    """CNN2D model for 2D spectrogram classification, baseline model.

    Input:
        X: (n_samples, n_channel, n_width, n_height)
        Y: (n_samples)

    Output:
        out: (n_samples)

    """

    def __init__(self, in_channels: int, out_channels: int, model: nn.Module) -> None:
        """Initialize the CNN1D model.

        :param in_channels: The number of input channels.
        :param out_channels: The number of output channels.
        :param model: The model to use.
        """
        super(CNN2D, self).__init__()  # noqa: UP008

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = model

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
