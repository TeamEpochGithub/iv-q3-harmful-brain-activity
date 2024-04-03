"""Main dataset for EEG / Spectrogram data."""
import copy
import typing
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

from src.typing.typing import XData


@dataclass
class MainDataset(Dataset):  # type: ignore[type-arg]
    """Main dataset for EEG data."""

    data_type: str
    X: XData | None = None
    y: npt.NDArray[np.float32] | None = None
    get_item_custom: Any | None = None
    augmentations: Any | None = None
    use_aug: bool = field(hash=False, repr=False, init=True, default=False)
    subsample_method: str | None = None

    def __post_init__(self) -> None:
        """Set up the dataset."""
        if self.X is None:
            raise ValueError("XData not set up.")
        self.X.meta = self.X.meta.reset_index(drop=True)
        if self.subsample_method == "random":
            X_meta = copy.deepcopy(self.X.meta)
            # append an index column to the meta data
            X_meta["index"] = copy.deepcopy(X_meta.index)

            # Get a random occurance of each eeg_id
            # Set sample seed for consistent results
            seed = 42
            unique_indices = X_meta.groupby("eeg_id").sample(1, random_state=seed)["index"]

            # Use the unique indices to index the meta data
            self.X = replace(self.X)  # shallow copy of XData, so only meta is changed
            self.X.meta = X_meta.loc[unique_indices].reset_index(drop=True)

            self.indices = unique_indices.to_list()
            # use self indices to index the y data
            if self.y is not None:
                self.y = self.y[self.indices, :]

        elif self.subsample_method == "running_random":
            # Create a mapping of idx to unique eeg_id
            self.id_mapping = dict(enumerate(self.X.meta["eeg_id"].unique()))
            # Group the metadata by eeg_id
            self.grouped = self.X.meta.groupby("eeg_id")

    def setup_prediction(self, X: XData) -> None:
        """Set up the dataset for prediction."""
        self.X = X

    def __len__(self) -> int:
        """Get the length of the dataset."""
        # Trick the dataloader into thinking the dataset is smaller than it is
        if self.subsample_method == "running_random":
            if self.X is None:
                raise ValueError("X Data not set up.")
            return len(self.X.meta["eeg_id"].unique())
        return len(self.X)  # type: ignore[arg-type]

    def __getitems__(self, indices: list[int]) -> tuple[Any, Any]:
        """Get multiple items from the dataset and apply augmentations if necessary."""
        all_x = []
        all_y = []

        # Read the data in a loop
        for idx in indices:
            x, y = self.__getitem__(idx)
            all_x.append(x)
            all_y.append(y)
        # Create a tensor from the list of tensors
        all_x_tensor = torch.stack(all_x)
        # If labels exist, create a tensor from the list of tensors
        if isinstance(all_y[0], torch.Tensor):
            all_y_tensor = torch.stack(all_y)
        else:
            all_y_tensor = torch.empty(1)
        # Apply augmentations if necessary
        if self.augmentations is not None and self.use_aug:
            all_x_tensor, all_y_tensor = self.augmentations(all_x_tensor.to("cuda"), all_y_tensor.to("cuda"))
        return all_x_tensor, all_y_tensor

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        """Get an item from the dataset.

        :param idx: The index to get.
        :return: The data and the labels.
        """
        # Check if the data is set up, we need X.
        if self.X is None:
            raise ValueError("X Data not set up.")

        if self.subsample_method == "running_random":
            # Using the mapping get the eeg_id for this idx
            eeg_id = self.id_mapping[idx]
            # Get the indices for this eeg_id
            indices = self.grouped.get_group(eeg_id)
            # Get a random index from the indices
            idx = indices.sample(1, random_state=42).index[0]
            # Now idx is the dataframe index and not the idx of the dataset

        # Create a switch statement to handle the different data types
        match self.data_type:
            case "eeg":
                x, y = self._eeg_getitem(idx)
                x = x.transpose(1, 0)
                # y = torch.from_numpy(y)
                # if self.augmentations is not None and self.use_aug:
                #     x_torch = torch.from_numpy(x)
                #     x = self.augmentations(x_torch.unsqueeze(0)).squeeze(0)
            case "kaggle_spec":
                x, y = self._kaggle_spec_getitem(idx)
                # if self.augmentations is not None and self.use_aug:
                #     x = self.augmentations(x).squeeze(0)
            case "eeg_spec":
                x, y = self._eeg_spec_getitem(idx)
                # if self.augmentations is not None and self.use_aug:
                #     x = self.augmentations(x).squeeze(0)
            case "custom":
                x, y = self._custom_getitem(idx)
            case _:
                raise ValueError(f"Data type {self.data_type} not recognized.")
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        return x, y

    @typing.no_type_check
    def _eeg_getitem(self, idx: int) -> tuple[Any, Any]:  # type: ignore[no-untyped-def]
        """Get an item from the EEG dataset.

        :param idx: The index to get.
        :return: The EEG data and the labels.
        """
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
    def _eeg_spec_getitem(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get an item from the EEG spectrogram dataset.

        :param idx: The index to get.
        :return: The EEG spectrogram data and the labels.
        """
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

    @typing.no_type_check
    def _custom_getitem(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get an item from the EEG spectrogram dataset.

        :param idx: The index to get.
        :return: The EEG spectrogram data and the labels.
        """
        if self.get_item_custom is None:
            raise ValueError("Custom get item is not set.")

        X_eeg, X_kaggle_spec, X_eeg_spec = None, None, None

        if self.X.eeg is not None:
            X_eeg, _ = self._eeg_getitem(idx)

        if self.X.kaggle_spec is not None:
            X_kaggle_spec, _ = self._kaggle_spec_getitem(idx)

        if self.X.eeg_spec is not None:
            X_eeg_spec, _ = self._eeg_spec_getitem(idx)

        idx = self.indices[idx]
        labels = [] if self.y is None else self.y[idx, :]

        return self.get_item_custom(X_eeg, X_kaggle_spec, X_eeg_spec, labels)
