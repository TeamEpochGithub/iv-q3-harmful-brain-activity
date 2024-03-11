from src.modules.logging.logger import Logger
from epochalyst.pipeline.model.training.torch_trainer import TorchTrainer
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
from torch import Tensor
from src.modules.torch_datasets.basic_eeg_dataset import EEGDataset

class VerboseTorchBlock(TorchTrainer, Logger):

    def __init__(self, **kwargs):
        """Initialise the verbose torch block."""
        super().__init__(**kwargs)
        self.log_to_terminal("Initialising verbose torch block")

    def create_datasets(
        self,
        x: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        train_indices: list[int],
        test_indices: list[int],
        cache_size: int = -1,
    ) -> tuple[Dataset[tuple[Tensor, ...]], Dataset[tuple[Tensor, ...]]]:
        """Create the datasets for training and validation.

        :param x: The input data.
        :param y: The target variable.
        :param train_indices: The indices to train on.
        :param test_indices: The indices to test on.
        :return: The training and validation datasets.
        """
        metadata = x[2]
        eeg = x[0]
        y_dataset = y
        train_dataset = EEGDataset(eeg, metadata.iloc[train_indices, :], y)
        


        return train_dataset, y_dataset
