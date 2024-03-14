import torch.nn.functional as F
import torchaudio.transforms as T
from segmentation_models_pytorch import Unet
from torch import nn

from src.modules.training.models.multi_res_bi_GRU import MultiResidualBiGRU
from src.modules.training.models.unet_decoder import UNet1DDecoder


class MultiResidualBiGRUwSpectrogramCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
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
            T.Spectrogram(n_fft=63, hop_length=1),
            T.AmplitudeToDB(top_db=80),
            SpecNormalize(),
        )
        self.GRU = MultiResidualBiGRU(
            input_size=in_channels,
            hidden_size=32,
            out_size=out_channels,
            n_layers=5,
            bidir=True,
            activation="relu",
            flatten=False,
            dropout=0,
            internal_layers=1,
            model_name="",
        )
        # will shape the encoder outputs to the same shape as the original inputs
        self.liner = nn.Linear(in_features=32, out_features=in_channels)
        self.linear2 = nn.Linear(in_features=10016, out_features=1)

        self.decoder = UNet1DDecoder(
            n_channels=32,
            n_classes=out_channels,
            bilinear=False,
            scale_factor=2,
            # hardcoded for now
            # TODO make this a config
            duration=10016,
        )

    def forward(self, x, use_activation=True):
        x = F.pad(x, (0, 0, 0, 16))
        x = x.permute(0, 2, 1)
        x_spec = self.spectrogram(x)
        x_encoded = self.encoder(x_spec).squeeze(1)
        # The rest of the features are subsampled and passed to the decoder
        # as residual features

        x_decoded = self.decoder(x_encoded)

        x_encoded = x_encoded.permute(0, 2, 1)
        x_encoded_linear = self.liner(x_encoded)

        x_encoded_linear += x.permute(0, 2, 1)

        y, _ = self.GRU(x_encoded_linear, use_activation=use_activation)
        out = self.linear2(y.permute(0, 2, 1) + x_decoded.permute(0, 2, 1))
        out = nn.functional.softmax(out, dim=1)
        return out.permute(0, 2, 1)


class SpecNormalize(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # x: (batch, channel, freq, time)

        min_ = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_ = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

        return (x - min_) / (max_ - min_ + self.eps)
