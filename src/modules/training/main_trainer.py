"""Module for example training block."""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from epochalyst.pipeline.model.training.torch_trainer import TorchTrainer
from numpy import typing as npt
from torch.utils.data import Dataset

from src.modules.logging.logger import Logger
from src.typing.typing import XData


@dataclass
class MainTrainer(TorchTrainer, Logger):
    """Main training block for training EEG / Spectrogram models.

    :param dataset: The dataset to use for training.
    """
    dataset: Dataset | None = None

    def create_datasets(self, x: XData, y: npt.NDArray[np.float32], train_indices: list[int], test_indices: list[int], cache_size: int = -1, ) -> tuple[Dataset, Dataset]:
        """Override custom create_datasets to allow for for training and validation.

        :param x: The input data.
        :param y: The target variable.
        :param train_indices: The indices to train on.
        :param test_indices: The indices to test on.
        :return: The training and validation datasets.
        """
        # Set up the train dataset
        train_dataset = deepcopy(self.dataset)
        train_dataset.setup(x, y, train_indices)

        # Set up the test dataset
        if test_indices is not None:
            test_dataset = deepcopy(self.dataset)
            test_dataset.setup(x, y, test_indices)
        else:
            test_dataset = None

        return train_dataset, test_dataset
