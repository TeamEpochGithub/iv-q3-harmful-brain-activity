# Make a pytorch dataset
from dataclasses import dataclass

import pandas as pd
from torch.utils.data import Dataset

from src.typing.typing import XData


@dataclass
class MainDataset(Dataset):
    """
    Main dataset for EEG data.
    """
    data_type: str

    X: XData | None = None
    y: pd.DataFrame | None = None
    indices: list[int] | None = None

    def setup(self, X: XData, y: pd.DataFrame, indices: list[int]):
        """Set up the dataset."""
        self.X = X
        self.y = y
        self.indices = indices

    def setup_prediction(self, X: XData):
        """Set up the dataset for prediction."""
        self.X = X
        self.indices = list(range(len(X.meta)))

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.indices)

    def __getitem__(self, idx):
        # Check if the data is set up, we need X.
        if self.X is None:
            raise ValueError("X Data not set up.")
        if self.indices is None:
            raise ValueError("Indices not set up.")

        # Create a switch statement to handle the different data types
        match self.data_type:
            case 'eeg':
                return self._eeg_getitem(idx)
            case 'kaggle_spec':
                return self._kaggle_spec_getitem(idx)
            case 'eeg_spec':
                return self._eeg_spec_getitem(idx)
            case _:
                raise ValueError(f"Data type {self.data_type} not recognized.")

    def _eeg_getitem(self, idx):
        """Get an item from the EEG dataset.

        :param idx: The index to get.
        :return: The EEG data and the labels.
        """
        idx = self.indices[idx]
        metadata = self.X.meta
        all_eegs = self.X.eeg
        eeg_frequency = self.X.shared['eeg_freq']
        offset = self.X.shared['eeg_label_offset_s']

        # Get the eeg id from the idx in the metadata
        eeg_id = metadata.iloc[idx]['eeg_id']
        eeg_label_offset_seconds = int(metadata.iloc[idx]['eeg_label_offset_seconds'])
        eeg = all_eegs[eeg_id]

        # Get the start and end of the eeg data
        start = eeg_label_offset_seconds * eeg_frequency
        end = (eeg_label_offset_seconds * eeg_frequency) + (offset * eeg_frequency)

        # Get the correct 50 second window of eeg data
        eeg = eeg.iloc[start:end, :]

        if self.y is None:
            return eeg.to_numpy(), None

        # Get the 6 labels of the experts, if they exist
        labels = self.y[idx, :]
        # For each row, make sure the sum of the labels is 1
        labels = labels / labels.sum(axis=1)[:, None]
        return eeg.to_numpy(), labels

    def _kaggle_spec_getitem(self, idx):
        """Get an item from the Kaggle spectrogram dataset.

        :param idx: The index to get.
        :return: The Kaggle spectrogram data and the labels.
        """
        # TODO: Implement this in a future issue
        pass

    def _eeg_spec_getitem(self, idx):
        """Get an item from the EEG spectrogram dataset.

        :param idx: The index to get.
        :return: The EEG spectrogram data and the labels.
        """
        # TODO: Implement this in a future issue
        pass
