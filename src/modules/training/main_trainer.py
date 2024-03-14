"""Module for example training block."""
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from epochalyst.pipeline.model.training.torch_trainer import TorchTrainer
from numpy import typing as npt
from torch import Tensor
from torch.utils.data import Dataset

import wandb
from src.modules.logging.logger import Logger
from src.typing.typing import XData


@dataclass
class MainTrainer(TorchTrainer, Logger):
    """Main training block for training EEG / Spectrogram models.

    :param dataset: The dataset to use for training.
    """

    dataset: Dataset[Any] = field(default_factory=Dataset)
    model_name: str = "WHAT_ARE_YOU_TRAINING_PUT_A_NAME_IN_THE_MAIN_TRAINER"  # No spaces allowed

    def create_datasets(
        self,
        x: XData,
        y: npt.NDArray[np.float32],
        train_indices: list[int],
        test_indices: list[int],
        cache_size: int = -1,  # noqa: ARG002
    ) -> tuple[Dataset[Any], Dataset[Any]]:
        """Override custom create_datasets to allow for for training and validation.

        :param x: The input data.
        :param y: The target variable.
        :param train_indices: The indices to train on.
        :param test_indices: The indices to test on.
        :return: The training and validation datasets.
        """
        # Set up the train dataset
        train_dataset = deepcopy(self.dataset)
        train_dataset.setup(x, y, train_indices)  # type: ignore[attr-defined]

        # Set up the test dataset
        if test_indices is not None:
            test_dataset = deepcopy(self.dataset)
            test_dataset.setup(x, y, test_indices)  # type: ignore[attr-defined]
        else:
            test_dataset = None

        return train_dataset, test_dataset

    def create_prediction_dataset(self, x: npt.NDArray[np.float32]) -> Dataset[Any]:
        """Create the prediction dataset.

        :param x: The input data.
        :return: The prediction dataset.
        """
        predict_dataset = deepcopy(self.dataset)
        predict_dataset.setup_prediction(x)  # type: ignore[attr-defined]
        return predict_dataset

    def _concat_datasets(
        self,
        train_dataset: Dataset[tuple[Tensor, ...]],
        test_dataset: Dataset[tuple[Tensor, ...]],  # noqa: ARG002
        train_indices: list[int],  # noqa: ARG002
        test_indices: list[int],  # noqa: ARG002
    ) -> Dataset[tuple[Tensor, ...]]:
        """Concatenate the training and test datasets according to original order specified by train_indices and test_indices.

        :param train_dataset: The training dataset.
        :param test_dataset: The test dataset.
        :param train_indices: The indices for the training data.
        :param test_indices: The indices for the test data.
        :return: A new dataset containing the concatenated data in the original order.
        """
        indices = list(range(len(train_dataset.X.meta)))  # type: ignore[attr-defined]
        train_dataset.indices = indices  # type: ignore[attr-defined]
        return train_dataset

    def _save_model(self) -> None:
        super()._save_model()
        if wandb.run:
            model_artifact = wandb.Artifact(self.model_name, type="model")
            model_artifact.add_file(f"{self.model_directory}/{self.get_hash()}.pt")
            wandb.log_artifact(model_artifact)
