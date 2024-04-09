"""Transformation block that extracts features from the EEG data."""
from dataclasses import dataclass
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.typing import XData


@dataclass
class ExtractFeatures(VerboseTransformationBlock):
    """Transformation block that sets NaN values in the EEG data to zero."""

    def custom_transform(self, data: XData, **kwargs: Any) -> XData:
        """Apply the transformation.

        :param data: The X data to transform
        :return: The transformed data
        """
        if data.eeg is None:
            raise ValueError("No EEG data to transform")
        if data.shared is None:
            raise ValueError("No shared data to read EEG frequency from")

        # create a new dataframe, where each row has the features extracted per row in metadata
        features = []
        for _, row in tqdm(data.meta.iterrows(), desc="Extracting features", total=len(data.meta)):
            # get the 50 sec window
            eeg = data.eeg[int(row["eeg_id"])]
            offset = int(row["eeg_label_offset_seconds"])
            start = offset * data.shared["eeg_freq"]
            end = start + data.shared["eeg_freq"] * data.shared["eeg_len_s"]
            sequence = eeg.iloc[start:end]

            # extract features from the EEG data
            feature_row = self._extract_from_sequence(sequence, data.shared["eeg_freq"])
            features.append(feature_row)

        # create a dataframe from the list of series
        data.features = pd.DataFrame(features)
        return data

    def _extract_from_sequence(self, eeg: pd.DataFrame, sampling_rate: int) -> pd.DataFrame:  # noqa: ARG002
        """Extract features from the EEG data.

        :param eeg: The EEG data
        :return: The extracted features
        """
        result = pd.Series()

        # Mean of each channel
        result["mean"] = eeg.mean().mean()

        # Standard deviation of each channel
        result["std"] = eeg.std().mean()

        # def freqs(col):
        #     fft = np.abs(np.fft.fft(eeg[col]))[:len(eeg) // 2]
        #     freq = np.fft.fftfreq(len(eeg), 1 / sampling_rate)[:len(eeg) // 2]
        #     fft = fft[freq < 5]
        #     freq = freq[freq < 5]
        #     return freq, fft
        #
        # fft_arr = []
        # for col in eeg.columns:
        #     freq, fft = freqs(col)
        #     fft_arr.append(fft)
        # avg_fft = np.mean(fft_arr, axis=0)
        #
        # custom_feature = eeg.diff().abs().mean(axis=1)
        # custom_feature = custom_feature.rolling(100).median().ffill().bfill()
        #
        # similarities = []
        # dots = []
        # for left, right in [
        #     ('LT1', 'RT1'),
        #     ('LT2', 'RT2'),
        #     ('LP1', 'RP1'),
        #     ('LP2', 'RP2')]:
        #     similarities.append(np.corrcoef(eeg[left], eeg[right])[0, 1])
        #     dots.append(np.dot(eeg[left], eeg[right]))

        amplitude_left = eeg[[col for col in eeg.columns if "L" in col]].abs().mean().mean()
        amplitude_right = eeg[[col for col in eeg.columns if "R" in col]].abs().mean().mean()

        normalized_amplitude_difference_index = abs(amplitude_left - amplitude_right) / (amplitude_left + amplitude_right)

        # result['low_activity'] = (custom_feature < 0.02).mean()
        # result['high_activity'] = (custom_feature > 0.1).mean()
        # result['freq_peak'] = np.argmax(avg_fft)
        # result['freq_peak_val'] = np.max(avg_fft)
        # result['similarity'] = np.mean(similarities)
        # result['dots'] = np.mean(dots)
        # result['2-5hz'] = np.sum(avg_fft[(freq > 2) & (freq < 5)])
        # result['2-5hz_over_peak'] = result['2-5hz'] / result['freq_peak_val']
        # result['peak_ratio'] = np.max(avg_fft) / np.mean(avg_fft)
        result["NDAI"] = normalized_amplitude_difference_index

        return pd.DataFrame(result).T.fillna(0)
