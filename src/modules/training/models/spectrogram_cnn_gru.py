import torch.nn.functional as F
import torchaudio.transforms as T
from segmentation_models_pytorch import Unet
from torch import nn
import torch

from src.modules.training.models.multi_res_bi_GRU import MultiResidualBiGRU
from src.modules.training.models.unet_decoder import UNet1DDecoder


class MultiResidualBiGRUwSpectrogramCNN(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers=5):
        super(MultiResidualBiGRUwSpectrogramCNN, self).__init__()
        # TODO exclude some of the features from the spectrogram
        self.encoder = Unet(
            encoder_name="resnet34",
            in_channels=3,
            encoder_weights=None,
            classes=1,
            encoder_depth=5,
        )

        self.GRU = MultiResidualBiGRU(
            input_size=in_channels,
            hidden_size=512,
            out_size=out_channels,
            n_layers=n_layers,
            bidir=True,
            activation="relu",
            flatten=False,
            dropout=0,
            internal_layers=1,
            model_name="",
        )
        # will shape the encoder outputs to the same shape as the original inputs
        self.linear1 = nn.Linear(in_features=512, out_features=in_channels)
        self.linear2 = nn.Linear(in_features=512, out_features=2000)

        self.decoder = UNet1DDecoder(
            n_channels=512,
            n_classes=out_channels,
            bilinear=False,
            scale_factor=2,
            duration=512,
        )
        self.batch_norm = nn.BatchNorm1d(in_channels)

    def forward(self, x, use_activation=True):
        eeg = x[0]
        spec = x[1]
        eeg = self.batch_norm(eeg)
        encoded_spec = self.encoder(spec).squeeze(1)
        # The rest of the features are subsampled and passed to the decoder
        # as residual features

        x_decoded = self.decoder(encoded_spec)

        encoded_spec = encoded_spec.permute(0, 2, 1)
        x_encoded_linear = self.linear1(encoded_spec)
        # Some activation function here
        x_encoded_linear2 = self.linear2(x_encoded_linear.permute(0, 2, 1))

        x_encoded_linear = eeg.permute(0, 2, 1) + x_encoded_linear2.permute(0, 2, 1)

        y, _ = self.GRU(x_encoded_linear, use_activation=use_activation)
        out = y.permute(0, 2, 1) + x_decoded.permute(0, 2, 1)
        return out.permute(0, 2, 1)

