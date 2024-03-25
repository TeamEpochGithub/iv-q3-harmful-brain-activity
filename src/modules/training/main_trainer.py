"""Module for example training block."""
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import wandb
from epochalyst.logging.section_separator import print_section_separator
from epochalyst.pipeline.model.training.torch_trainer import TorchTrainer
from numpy import typing as npt
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.modules.logging.logger import Logger
from src.typing.typing import XData


@dataclass
class MainTrainer(TorchTrainer, Logger):
    """Main training block for training EEG / Spectrogram models.

    :param dataset: The dataset to use for training.
    """

    dataset: Dataset[Any] = field(default_factory=Dataset)
    model_name: str = "WHAT_ARE_YOU_TRAINING_PUT_A_NAME_IN_THE_MAIN_TRAINER"  # No spaces allowed
    fold: int = field(default=-1, init=False, repr=False, compare=False)

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

        # from src.utils.visualize_vote_distribution import visualize_vote_distribution
        # visualize_vote_distribution(y, train_indices, test_indices)
        train_dataset = deepcopy(self.dataset)
        train_dataset.setup(x, y, train_indices, use_aug=True, subsample_data=True)  # type: ignore[attr-defined]

        # Set up the test dataset
        if test_indices is not None:
            test_dataset = deepcopy(self.dataset)
            test_dataset.setup(x, y, test_indices, use_aug=False, subsample_data=True)  # type: ignore[attr-defined]
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
        test_indices: list[int],
    ) -> Dataset[tuple[Tensor, ...]]:
        """Concatenate the training and test datasets according to original order specified by train_indices and test_indices.

        :param train_dataset: The training dataset.
        :param test_dataset: The test dataset.
        :param train_indices: The indices for the training data.
        :param test_indices: The indices for the test data.
        :return: A new dataset containing the concatenated data in the original order.
        """
        # Create a deep copy of the train dataset
        pred_dataset = deepcopy(train_dataset)
        pred_dataset.setup(train_dataset.X, train_dataset.y, test_indices)  # type: ignore[attr-defined]
        return pred_dataset

    def custom_predict(
        self,
        x: npt.NDArray[np.float32],
        **pred_args: Any,
    ) -> npt.NDArray[np.float32]:
        """Predict on the test data.

        :param x: The input to the system.
        :return: The output of the system.
        """
        print_section_separator(f"Predicting model: {self.model.__class__.__name__}")
        self.log_to_debug(f"Predicting model: {self.model.__class__.__name__}")

        # Check if pred_args contains batch_size
        curr_batch_size = pred_args.get("batch_size", self.batch_size)

        # Create dataset
        pred_dataset = self.create_prediction_dataset(x)
        pred_dataloader = DataLoader(
            pred_dataset,
            batch_size=curr_batch_size,
            shuffle=False,
        )

        # Check if supposed to predict with a single model, or ensemble the fol models
        model_folds = pred_args.get("model_folds", None)

        # Predict with a single model
        if model_folds is None or model_folds == -1:
            self._load_model()
            return self.predict_on_loader(pred_dataloader)

        # Ensemble the fold models:
        predictions = []
        for i in range(model_folds):
            self.log_to_terminal(f"Predicting with model fold {i+1}/{model_folds}")
            self.fold = i  # set the fold, which updates the hash
            self._load_model()  # load the model for this fold
            predictions.append(self.predict_on_loader(pred_dataloader))

        return np.mean(predictions, axis=0)

    def predict_on_loader(
        self,
        loader: DataLoader[tuple[Tensor, ...]],
    ) -> npt.NDArray[np.float32]:
        """Predict on the loader.

        :param loader: The loader to predict on.
        :return: The predictions.
        """
        self.log_to_terminal("Predicting on the test data")
        self.model.eval()
        predictions = []
        with torch.no_grad(), tqdm(loader, unit="batch", disable=False) as tepoch:
            for data in tepoch:
                X_batch = data[0].to(self.device).float()
                y_pred = torch.softmax(self.model(X_batch), dim=1).cpu().numpy()
                predictions.extend(y_pred)
        self.log_to_terminal("Done predicting")
        return np.array(predictions)

    def custom_train(
        self,
        x: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        **train_args: Any,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Train the model.

        Overwritten to intercept the fold number from train args to the model.

        :param x: The input data.
        :param y: The target variable.
        :return The predictions and the labels.
        """
        self.fold = train_args.get("fold", -1)
        return super().custom_train(x, y, **train_args)

    def get_hash(self) -> str:
        """Get the hash of the block.

        Override the get_hash method to include the fold number in the hash.

        :return: The hash of the block.
        """
        if self.fold == -1:
            return self._hash
        return f"{self._hash}-{self.fold}"

    def _save_model(self) -> None:
        super()._save_model()
        if wandb.run:
            model_artifact = wandb.Artifact(self.model_name, type="model")
            model_artifact.add_file(f"{self.model_directory}/{self.get_hash()}.pt")
            wandb.log_artifact(model_artifact)

    def predict_on_loader(
        self, loader: DataLoader[tuple[Tensor, ...]]
    ) -> npt.NDArray[np.float32]:
        """Predict on the loader.

        :param loader: The loader to predict on.
        :return: The predictions.
        """
        self.log_to_terminal("Predicting on the test data")
        self.model.eval()
        predictions = []
        with torch.no_grad(), tqdm(loader, unit="batch", disable=False) as tepoch:
            for data in tepoch:
                X_batch = data[0].to(self.device).float()

                y_pred = torch.softmax(self.model(X_batch),dim=1).cpu().numpy()
                predictions.extend(y_pred)

        self.log_to_terminal("Done predicting")
        return np.array(predictions)