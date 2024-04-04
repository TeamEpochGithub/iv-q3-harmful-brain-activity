import torch.nn.functional as F
import torchaudio.transforms as T
from segmentation_models_pytorch import Unet
from torch import nn
import torch

from src.modules.training.models.multi_res_bi_GRU import MultiResidualBiGRU
from src.modules.training.models.unet_decoder import UNet1DDecoder


class MultiResidualBiGRUwSpectrogramCNN(nn.Module):
    def __init__(self, in_channels: int, out_channels, n_fft=127, n_layers=5, dropout: float = 0.0, hop_length=1):
        super(MultiResidualBiGRUwSpectrogramCNN, self).__init__()
        # TODO exclude some of the features from the spectrogram
        self.encoder = Unet(
            encoder_name="resnet34",
            in_channels=in_channels,
            encoder_weights=None,
            classes=1,
            encoder_depth=5,
        )
        self.spectrogram = nn.Sequential(
            T.Spectrogram(n_fft=n_fft, hop_length=hop_length),
            T.AmplitudeToDB(top_db=80),
            SpecNormalize(),
        )
        self.GRU = MultiResidualBiGRU(
            input_size=in_channels,
            hidden_size=(n_fft+1)//2,
            out_size=out_channels,
            n_layers=n_layers,
            bidir=True,
            activation="relu",
            flatten=False,
            dropout=dropout,
            internal_layers=1,
            model_name="",
        )
        # will shape the encoder outputs to the same shape as the original inputs
        self.liner = nn.Linear(in_features=(n_fft+1)//2, out_features=in_channels)
        self.pool_stage = nn.AvgPool1d(9)
        self.last_linear = nn.Linear(in_features=224, out_features=1)

        self.decoder1 = UNet1DDecoder(
            n_channels=(n_fft+1)//2,
            n_classes=out_channels,
            bilinear=False,
            scale_factor=2,
            duration=224,
        )
        self.decoder2 = UNet1DDecoder(
            n_channels=(n_fft+1)//2,
            n_classes=in_channels,
            bilinear=False,
            scale_factor=2,
            duration=224,
        )
        self.batch_norm = nn.BatchNorm1d(in_channels)
        self.last_conv = nn.Sequential(nn.Conv1d(in_channels=6, out_channels=18, kernel_size=5, stride=1, padding=2),
                                       nn.AvgPool1d(14),
                                       nn.Conv1d(in_channels=18, out_channels=9, kernel_size=3, stride=1),
                                       nn.AvgPool1d(14),
                                       nn.Conv1d(in_channels=9, out_channels=6, kernel_size=1, stride=1))

    def forward(self, x, use_activation=True):
        x = F.pad(x, (8, 8, 0, 0))
        x_spec = self.spectrogram(x)
        x_encoded = self.encoder(x_spec).squeeze(1)

        x_decoded = self.decoder1(x_encoded)

        x_encoded_linear = self.liner(x_encoded)
        x_downsampled = self.pool_stage(x)
        x_encoded_linear = x_downsampled.permute(0, 2, 1) + x_encoded_linear

        y, _ = self.GRU(x_encoded_linear, use_activation=use_activation)
        out = y.permute(0, 2, 1) + x_decoded.permute(0, 2, 1)

        out = self.last_conv(out).squeeze(-1)
        return out


class SpecNormalize(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x: (batch, channel, freq, time)

        min_ = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_ = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

        return (x - min_) / (max_ - min_ + self.eps)
