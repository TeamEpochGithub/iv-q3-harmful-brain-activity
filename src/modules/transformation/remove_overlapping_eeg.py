"""Transformation block that takes removes overlapping eeg samples from the data."""
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm
import numpy as np
from epochalyst.pipeline.model.training.training_block import TrainingBlock
from src.typing.typing import XData


@dataclass
class EEGOverlapFilter(TrainingBlock):
    """A transformation block that removes overlapping eeg samples from the data."""

    def custom_train(self, X: XData, y: np.ndarray, **kwargs: Any) -> XData:
        """Remove overlapping EEGs.

        :param data: The X data to transform, as tuple (eeg, spec, meta)
        :param y: The y data to transform
        :param kwargs: The training arguments
        :return: The transformed data
        """
        # Separate the meta data from X
        meta = X.meta
        # append an index column to the meta data
        meta['index'] = range(len(meta))
        # Get the first occurance of each eeg_id
        unique_eegs = meta.groupby('eeg_id').first()
        # Use the index column from X to index the y data
        y_unique = y[unique_eegs['index']]
        # Remove the index column from the meta data
        meta.pop('index')
        # Overwrite X.meta with the unique_eegs
        X.meta = unique_eegs

        return X, y_unique

    def custom_predict(self, X: XData, **pred_args: Any) -> Any:
        """Return the input data unchanged.
        
        :param X: The input data.
        :param pred_args: The prediction arguments.
        :return: The input data."""
        return X