"""Main dataset for EEG / Spectrogram data."""
import copy
import typing
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.typing.typing import XData


@dataclass
class MainDataset(Dataset):  # type: ignore[type-arg]
    """Main dataset for EEG data."""

    data_type: str
    X: XData | None = None
    y: pd.DataFrame | None = None
    indices: list[int] | None = None
    augmentations: Any | None = None
    use_aug: bool = field(hash=False, repr=False, init=False, default=False)

    def setup(self, X: XData, y: pd.DataFrame, indices: list[int], *, use_aug: bool = False, subsample_data: bool = False) -> None:
        """Set up the dataset."""
        self.X = X
        self.y = y
        self.indices = indices
        if subsample_data:
            X_meta = copy.deepcopy(self.X.meta.iloc[indices])
            # append an index column to the meta data
            X_meta["index"] = copy.deepcopy(X_meta.index)
            # Get the first occurance of each eeg_id
            unique_indices = X_meta.groupby("eeg_id").first()["index"]
            # Use the unique indices to index the meta data
            self.indices = unique_indices.to_list()
        self.use_aug = use_aug

    def setup_prediction(self, X: XData) -> None:
        """Set up the dataset for prediction."""
        self.X = X
        self.indices = list(range(len(X.meta)))

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.indices)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        """Get an item from the dataset.

        :param idx: The index to get.
        :return: The data and the labels.
        """
        # Check if the data is set up, we need X.
        if self.X is None:
            raise ValueError("X Data not set up.")
        if self.indices is None:
            raise ValueError("Indices not set up.")

        # Create a switch statement to handle the different data types
        match self.data_type:
            case "eeg":
                x, y = self._eeg_getitem(idx)
            case "kaggle_spec":
                x, y = self._kaggle_spec_getitem(idx)
            case "eeg_spec":
                x, y = self._eeg_spec_getitem(idx)
            case _:
                raise ValueError(f"Data type {self.data_type} not recognized.")

        if self.augmentations is not None and self.use_aug:
            x = self.augmentations(torch.from_numpy(x).to("cuda")).squeeze(0)

        return x, y

    @typing.no_type_check
    def _eeg_getitem(self, idx: int) -> tuple[Any, Any]:  # type: ignore[no-untyped-def]
        """Get an item from the EEG dataset.

        :param idx: The index to get.
        :return: The EEG data and the labels.
        """
        idx = self.indices[idx]
        metadata = self.X.meta
        all_eegs = self.X.eeg
        eeg_frequency = self.X.shared["eeg_freq"]
        offset = self.X.shared["eeg_len_s"]

        # Get the eeg id from the idx in the metadata
        eeg_id = metadata.iloc[idx]["eeg_id"]
        eeg_label_offset_seconds = int(metadata.iloc[idx]["eeg_label_offset_seconds"])
        eeg = all_eegs[eeg_id]

        # Get the start and end of the eeg data
        start = eeg_label_offset_seconds * eeg_frequency
        end = (eeg_label_offset_seconds * eeg_frequency) + (offset * eeg_frequency)

        # Get the correct 50 second window of eeg data
        eeg = eeg.iloc[start:end, :]

        if self.y is None:
            return eeg.to_numpy(), []

        # Get the 6 labels of the experts, if they exist
        labels = self.y[idx, :]
        return eeg.to_numpy(), labels

    @typing.no_type_check
    def _kaggle_spec_getitem(self, idx: int) -> tuple[Any, Any]:
        """Get an item from the Kaggle spectrogram dataset.

        :param idx: The index to get.
        :return: The Kaggle spectrogram data and the labels.
        """
        idx = self.indices[idx]
        metadata = self.X.meta
        all_specs = self.X.kaggle_spec
        frequency = self.X.shared["kaggle_spec_freq"]
        offset = self.X.shared["kaggle_spec_len_s"]

        # Get the eeg and spectrogram id from the idx in the metadata
        spec_id = metadata.iloc[idx]["spectrogram_id"]
        spec_label_offset_seconds = metadata.iloc[idx]["spectrogram_label_offset_seconds"]
        spectrogram = all_specs[spec_id]

        # Get the start and end of the spectrogram data
        start = int(spec_label_offset_seconds * frequency)
        end = int((spec_label_offset_seconds * frequency) + (offset * frequency))

        # Slice the 4 channel spectrogram
        spectrogram = spectrogram[:, :, start:end]

        if self.y is None:
            return spectrogram, []

        # Get the 6 labels of the experts, if they exist
        labels = self.y[idx, :]

        return spectrogram, labels

    @typing.no_type_check
    def _eeg_spec_getitem(self, idx: int) -> tuple[Any, Any]:
        """Get an item from the EEG spectrogram dataset.

        :param idx: The index to get.
        :return: The EEG spectrogram data and the labels.
        """
        idx = self.indices[idx]
        metadata = self.X.meta
        eeg_frequency = self.X.shared["eeg_freq"]
        eeg_length = self.X.shared["eeg_len_s"]

        # Get the eeg id from the idx in the metadata
        eeg_id = metadata.iloc[idx]["eeg_id"]
        eeg_label_offset_seconds = int(metadata.iloc[idx]["eeg_label_offset_seconds"])

        # Get the spectrogram
        spectrogram = self.X.eeg_spec[eeg_id]

        # Calculate the indeces of the correct 50 second window of the egg data
        start = eeg_label_offset_seconds * self.X.shared["eeg_spec_freq"]
        end = (eeg_label_offset_seconds * self.X.shared["eeg_spec_freq"]) + (eeg_length * eeg_frequency)

        # Convert these to indices to spectrogram indeces (propotional to the length of the eeg data)
        start = int((start * self.X.eeg_spec[eeg_id].shape[2]) / self.X.eeg[eeg_id].shape[0])
        end = int((end * self.X.eeg_spec[eeg_id].shape[2]) / self.X.eeg[eeg_id].shape[0])

        ## Make sure the spectrogram is always the same length (same as the spectrograms created by 50s eeg data)
        current_length = end - start
        length_diff = current_length - self.X.shared["eeg_spec_test_spectrogram_size"][1]
        end -= length_diff

        # Get the snippet of the spectrogram
        spectrogram = spectrogram[:, :, start:end]

        # Pad/Crop the spectrogram to the correct size
        space_left = abs(spectrogram.shape[2] - self.X.shared["eeg_spec_size"][1]) // 2
        space_right = abs(spectrogram.shape[2] - self.X.shared["eeg_spec_size"][1]) - space_left

        if self.X.shared["eeg_spec_fitting_method"] == "pad":
            spectrogram = torch.nn.functional.pad(spectrogram, (space_left, space_right))
        elif self.X.shared["eeg_spec_fitting_method"] == "crop":
            spectrogram = spectrogram[:, :, space_left:-space_right]

        # Return the spectrogram and the labels
        if self.y is None:
            return spectrogram, []
        labels = self.y[idx, :]
        return spectrogram, labels
