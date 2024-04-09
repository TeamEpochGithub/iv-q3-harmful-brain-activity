"""Contains the MeanWindow"""
import copy
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock


@dataclass
class MeanWindowVotes(VerboseTransformationBlock):
    """Mean window to one class."""

    threshold: float = 25.0

    def custom_transform(self, data: np.ndarray[Any, Any], **kwargs: Any) -> np.ndarray[Any, Any]:
        """Average the votes of neighboring/overlapping windows

        :param data: The y data to transform which is a 2D array (n_samples, n_experts).
        :return: The transformed data.
        """
        # For each row, make sure the sum of the labels is 1
        logging.info(f"Averaging the votes of windows that are <= {self.threshold} apart...")
        meta = copy.deepcopy(kwargs["metadata"])
        # Group meta by eeg_id
        grouped_meta = meta.groupby("eeg_id")
        out = np.copy(data)
        out = out.astype(np.float32)
        # Iterate over each group
        for eeg_id, group in tqdm(grouped_meta):
            # Get the indices of rows in data that correspond to the current eeg_id
            indices = group.index
            for idx in indices:
                similar_items = group[abs(group["eeg_label_offset_seconds"] - meta.iloc[idx]["eeg_label_offset_seconds"]) < self.threshold]
                out[idx, :] = np.mean(data[similar_items.index, :], axis=0, dtype=np.float32)
                if np.all(out[idx, :] == 0):
                    print("Nan found")
        # Return the transformed data
        return out
