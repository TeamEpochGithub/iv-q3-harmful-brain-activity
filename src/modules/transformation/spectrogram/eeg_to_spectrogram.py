"""Example transformation block for the transformation pipeline."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.typing import XData

EEG_TEST_LEN_SEC = 50


@dataclass
class EEGToSpectrogram(VerboseTransformationBlock):
    """Creates a spectrogram from EEG data.

    :param eeg_sample_rate: The sample rate of the EEG data. If the EEG data was downsampled
        in a previous block, this will be the sample rate of the downsampled data.
    :param size: The size of the spectrogram when indexed to 50s. Due to varying EEG lengths
        (always larger or equal to 50s), the spectrograms created by this block will be
        larger than this size.
    :param fitting_method: Can be 'pad' or 'crop'. This parameter specifies the method which
        fits the spectrogram to the specified size. In practice, this will influence the final
        size of the spectrogram by a small amount to make sure it can be cropped or padded
        to the specified size during the indexing process.
    """

    eeg_sample_rate: int
    size: tuple[int, int] = (128, 320)
    fitting_method: str = "pad"

    def _create_mel_spectrogram(self, eeg: pd.DataFrame, group_channels: list[str]) -> torch.Tensor:
        eeg_tensor = torch.tensor(eeg.values, dtype=torch.float32)

        n_mels = self.size[0]
        hop_length = (EEG_TEST_LEN_SEC * self.eeg_sample_rate) // self.size[1]

        if self.fitting_method == "pad":
            hop_length += 1  # Make sure the spectrogram will be smaller than the specified size

        # Initialize MelSpectrogram transformation
        transform = MelSpectrogram(
            sample_rate=self.eeg_sample_rate,
            n_fft=1024,
            win_length=128,
            hop_length=hop_length,
            f_min=0.0,
            f_max=40,
            pad=0,
            n_mels=n_mels,
            window_fn=torch.hann_window,
            power=2.0,
            normalized=True,
            center=True,
            pad_mode="reflect",
            norm=None,
            mel_scale="htk",
        )

        group_tensor = eeg_tensor[:, eeg.columns.isin(group_channels)]

        spec_parts = []
        for i in range(group_tensor.shape[1] - 1):
            # Substract the next channel from the current channel
            diff = group_tensor[:, i] - group_tensor[:, i + 1]

            # Apply the MelSpectrogram transformation
            mel_spec = transform(diff)

            # Convert to decibels
            mel_spec_db = 10 * torch.log10(mel_spec + 1e-9)

            # Standardize the data to -1 to 1
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / mel_spec_db.std()

            # Save
            spec_parts.append(mel_spec_db)

        spectrogram = torch.mean(torch.stack(spec_parts), dim=0)

        # Set any NaN values to 0
        spectrogram[torch.isnan(spectrogram)] = 0

        return spectrogram

    def custom_transform(self, data: XData, **kwargs: Any) -> XData:
        """Apply a custom transformation to the data.

        :param data: The data to transform
        :param kwargs: Any additional arguments
        :return: The transformed data
        """
        if data.eeg is None:
            raise ValueError("No EEG data provided")

        if data.eeg_spec is None:
            data.eeg_spec = {}

        # Setup Params
        self.groups = [
            ["Fp1", "F7", "T3", "T5", "O1"],  # LL
            ["Fp1", "F3", "C3", "P3", "O1"],  # LP
            ["Fp2", "F8", "T4", "T6", "O2"],  # RP
            ["Fp2", "F4", "C4", "P4", "O2"],  # RR
        ]

        for eeg_id, eeg in tqdm(data.eeg.items(), desc="Creating EEG Spectrograms"):
            for group_channels in self.groups:
                spectrogram = self._create_mel_spectrogram(eeg, group_channels)

                # Save the spectrogram
                if eeg_id in data.eeg_spec:
                    data.eeg_spec[eeg_id] = torch.cat((data.eeg_spec[eeg_id], spectrogram.unsqueeze(0)))
                else:
                    data.eeg_spec[eeg_id] = spectrogram.unsqueeze(0)

        return data
