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
        
        self.channel_reduction1 = nn.Conv2d(9,1,1)
        self.channel_reduction2 = nn.Conv1d(9,6,1)
        self.pool_stage = nn.AvgPool1d(9)
        self.last_linear = nn.Linear(in_features=224, out_features=1)

        self.decoder = UNet1DDecoder(
            n_channels=(n_fft+1)//2,
            n_classes=in_channels,
            bilinear=False,
            scale_factor=2,
            duration=224,
        )
        self.batch_norm1 = nn.BatchNorm1d(in_channels)
        self.batch_norm2 = nn.BatchNorm1d(in_channels)
        self.batch_norm3 = nn.BatchNorm1d(in_channels)

    def forward(self, x, use_activation=True):
        x = self.batch_norm1(x)
        x = F.pad(x, (8, 8, 0, 0))
        # Make spectrogram
        x_spec = self.spectrogram(x)
        # Reduce channels of spectrogram
        x_spec_reduced = self.channel_reduction1(x_spec).squeeze(1)
        # Decode (encode) the spectrograms
        x_decoded = self.decoder(x_spec_reduced)
        # Reduce the number of channels of x_decoded
        x_decoded_reduced = self.channel_reduction2(x_decoded.permute(0,2,1))

        # Pool the original inputs
        x_pooled = self.pool_stage(x)
        # Sum outputs forthe gru input
        gru_in = self.batch_norm2(x_pooled).permute(0,2,1) + self.batch_norm3(x_decoded.permute(0,2,1)).permute(0,2,1)
        gru_out, _ = self.GRU(gru_in, use_activation=use_activation)

        last_linear_in = gru_out.permute(0,2,1) + x_decoded_reduced
        out = self.last_linear(last_linear_in).squeeze(-1)
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
