"""Common type definitions for the project."""
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch


@dataclass
class XData:
    """The X data to be used in the pipeline.

    :param eeg: The EEG data, as a dictionary of DataFrames
    :param kaggle_spec: The Kaggle spectrogram data, as a dictionary of Tensors
    :param eeg_spec: The EEG spectrogram data, as a dictionary of Tensors
    :param meta: The metadata, as a DataFrame
    :param shared: The shared data to be used in the pipeline. Contains frequency data, offset data, etc.
    """

    eeg: dict[int, pd.DataFrame] | None
    kaggle_spec: dict[int, torch.Tensor] | None
    eeg_spec: dict[int, torch.Tensor] | None
    meta: pd.DataFrame
    shared: dict[str, Any] | None

    def __getitem__(self, key: slice | int | list[int]):
        """Enables slice indexing on the meta attribute using iloc and filters other attributes based on eeg_id."""
        if isinstance(key, slice) or isinstance(key, int) or isinstance(key, list):
            sliced_meta = self.meta.iloc[key]
            eeg_ids = set(sliced_meta['eeg_id'])
            
            # # Filtering the dictionaries to keep only entries with keys in eeg_ids
            # filtered_eeg = {k: v for k, v in self.eeg.items() if k in eeg_ids} if self.eeg else None
            # filtered_kaggle_spec = {k: v for k, v in self.kaggle_spec.items() if k in eeg_ids} if self.kaggle_spec else None
            # filtered_eeg_spec = {k: v for k, v in self.eeg_spec.items() if k in eeg_ids} if self.eeg_spec else None
            
            return XData(eeg=self.eeg, 
                         kaggle_spec=self.kaggle_spec, 
                         eeg_spec=self.eeg_spec, 
                         meta=sliced_meta, 
                         shared=self.shared)
        else:
            raise TypeError("Invalid argument type.")

    def __len__(self) -> int:
        """Return the length of the meta attribute."""
        return len(self.meta)

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return "XData"
