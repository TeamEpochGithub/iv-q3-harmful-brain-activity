"""Module for example training block."""
import numpy as np
from epochalyst.pipeline.model.training.torch_trainer import TorchTrainer
from pandas._typing import npt
from torch import Tensor
from torch.utils.data import Dataset

from src.modules.logging.logger import Logger
from src.typing.typing import XData


class MainTrainer(TorchTrainer, Logger):
    """Main training block for training EEG / Spectrogram models.

    :param TorchTrainer: The torch trainer class.
    :param Logger: The logger class.
    """

    dataset: Dataset

    def create_datasets(self, x: XData, y: npt.NDArray[np.float32], train_indices: list[int], test_indices: list[int], cache_size: int = -1, ) -> tuple[
        Dataset[tuple[Tensor, ...]], Dataset[tuple[Tensor, ...]]]:
        """Override custom create_datasets to allow for for training and validation.

        :param x: The input data.
        :param y: The target variable.
        :param train_indices: The indices to train on.
        :param test_indices: The indices to test on.
        :return: The training and validation datasets.
        """
        # Give X and y to the dataset object
        self.dataset.setup(x, y)

        x_dataset = TensorDataset(
            torch.tensor(x[train_indices]), torch.tensor(y[train_indices])
        )
        y_dataset = TensorDataset(
            torch.tensor(x[test_indices]), torch.tensor(y[test_indices])
        )


        return x_dataset, y_dataset
