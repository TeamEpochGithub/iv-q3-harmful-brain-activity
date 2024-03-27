"""Module for example training block."""
import copy
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import wandb
from epochalyst.logging.section_separator import print_section_separator
from epochalyst.pipeline.model.training.torch_trainer import TorchTrainer
from numpy import typing as npt
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.modules.logging.logger import Logger
from src.modules.training.datasets.main_dataset import MainDataset
from src.typing.typing import XData


@dataclass
class MainTrainer(TorchTrainer, Logger):
    """Main training block for training EEG / Spectrogram models.

    :param model_name: The name of the model. No spaces allowed
    :param dataset: The dataset to use for training.
    :param two_stage: Whether to use two-stage training. See: https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/477461
    :param two_stage_kl_threshold: The threshold for dividing the dataset into two stages.
    :param two_stage_evaluator_threshold: The threshold for dividing the dataset into two stages, based on total number of votes.
     Note: remove the sum to one block from the target pipeline for this to work
    """

    dataset_args: dict[str, Any] = field(default_factory=dict)
    model_name: str = "WHAT_ARE_YOU_TRAINING_PUT_A_NAME_IN_THE_MAIN_TRAINER"  # No spaces allowed
    two_stage: bool = False
    two_stage_kl_threshold: float | None = None
    two_stage_evaluator_threshold: int | None = None
    _fold: int = field(default=-1, init=False, repr=False, compare=False)
    _stage: int = field(default=-1, init=False, repr=False, compare=False)

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
        train_data = x[train_indices]
        train_labels = y[train_indices]

        test_data = x[test_indices]
        test_labels = y[test_indices]

        train_dataset = MainDataset(X=train_data, y=train_labels, use_aug=True, **self.dataset_args)

        # Set up the test dataset
        if test_indices is not None:
            self.dataset_args["subsample_method"] = "random"
            test_dataset = MainDataset(X=test_data, y=test_labels, use_aug=False, **self.dataset_args)
        else:
            test_dataset = None

        # Make a backup of the original metadata for the scorer preds to work
        self.meta_backup = deepcopy(x.meta)
        self.y_backup = deepcopy(y)

        return train_dataset, test_dataset

    def create_prediction_dataset(self, x: XData) -> Dataset[Any]:
        """Create the prediction dataset.

        :param x: The input data.
        :return: The prediction dataset.
        """
        predict_dataset = MainDataset(X=x, use_aug=False, **self.dataset_args)
        predict_dataset.setup_prediction(x)
        return predict_dataset

    def _concat_datasets(
        self,
        train_dataset: MainDataset,  # noqa: ARG002
        test_dataset: MainDataset,
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
        # Since concat dataset is called before training starts we deepcopy to not corrupt the original dataset
        pred_dataset = copy.deepcopy(test_dataset)
        # Modify the pred_dataset metadata
        if pred_dataset.X is None:
            raise ValueError("XData should not be None in the prediction dataset.")
        pred_dataset.X.meta = self.meta_backup.iloc[test_indices, :].reset_index(drop=True)
        pred_dataset.y = self.y_backup[test_indices, :]
        return pred_dataset

    def custom_predict(self, x: XData, **pred_args: Any) -> torch.Tensor:
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

        # If using two-stage training, use the second stage
        if self.two_stage:
            self._stage = 1

        # Check if supposed to predict with a single model, or ensemble the fold models
        model_folds = pred_args.get("model_folds", None)

        # Predict with a single model
        if model_folds is None or model_folds == -1:
            self._load_model()
            return self.predict_on_loader(pred_dataloader)

        # Ensemble the fold models:
        predictions = []
        for i in range(model_folds):
            self.log_to_terminal(f"Predicting with model fold {i+1}/{model_folds}")
            self._fold = i  # set the fold, which updates the hash
            self._load_model()  # load the model for this fold
            predictions.append(self.predict_on_loader(pred_dataloader))

        predictions_stack = torch.stack(predictions)
        return torch.mean(predictions_stack, dim=0)

    def predict_on_loader(
        self,
        loader: DataLoader[tuple[Tensor, ...]],
    ) -> torch.Tensor:
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
                y_pred = self.model(X_batch).cpu()
                predictions.extend(y_pred)
        self.log_to_terminal("Done predicting")
        return torch.stack(predictions)

    def custom_train(
        self,
        x: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        **train_args: Any,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Train the model.

        Overwritten to intercept the fold number and enable two-stage training.

        :param x: The input data.
        :param y: The target variable.
        :return The predictions and the labels.
        """
        self._fold = train_args.get("fold", -1)
        if not self.two_stage:
            return super().custom_train(x, y, **train_args)

        # Two-stage training
        self.log_to_terminal("Two-stage training")
        train_indices = np.array(train_args.get("train_indices", range(len(y))))
        if self.two_stage_kl_threshold is not None and self.two_stage_evaluator_threshold is not None:
            raise ValueError("Cannot use both KL and evaluator threshold for two-stage training")

        if self.two_stage_kl_threshold is not None:
            peak_kl = self.compute_peak_kl(y[train_indices])
            train_indices_stage1 = list(train_indices)  # first stage is all data
            train_indices_stage2 = list(train_indices[peak_kl < self.two_stage_kl_threshold])  # second stage is with low KL
        elif self.two_stage_evaluator_threshold is not None:
            n_evaluators = y[train_indices].sum(axis=1)
            train_indices_stage1 = list(train_indices[n_evaluators <= self.two_stage_evaluator_threshold])
            train_indices_stage2 = list(train_indices[n_evaluators > self.two_stage_evaluator_threshold])
        else:
            raise ValueError("No two-stage threshold provided")

        self.log_to_terminal(f"Split data into two stages, sizes: {len(train_indices_stage1)} / {len(train_indices_stage2)}")

        self._stage = 0
        self.log_to_terminal("Training stage 1")
        train_args["train_indices"] = train_indices_stage1
        super().custom_train(x, y, **train_args)

        self._stage = 1
        self.log_to_terminal("Training stage 2")
        train_args["train_indices"] = train_indices_stage2
        return super().custom_train(x, y, **train_args)

    def compute_peak_kl(self, y: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Compute the KL-loss against a uniform distribution.

        This is used to determine how peaked a distribution is, for dividing the two stages.
        See: https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/477461

        :param y: The target variable.
        :return: The KL-divergence between the target and a uniform distribution.
        """
        normed = torch.tensor(y / y.sum(axis=1, keepdims=True)) + 1e-5
        uniform = torch.tensor([1 / 6] * 6)
        kl = nn.functional.kl_div(torch.log(normed), uniform, reduction="none")
        return kl.sum(dim=1).numpy()

    def get_hash(self) -> str:
        """Get the hash of the block.

        Override the get_hash method to include the fold number in the hash.

        :return: The hash of the block.
        """
        result = self._hash
        if self._fold != -1:
            result += f"_f{self._fold}"
        if self._stage != -1:
            result += f"_s{self._stage}"
        return result

    def _save_model(self) -> None:
        super()._save_model()
        if wandb.run:
            model_artifact = wandb.Artifact(self.model_name, type="model")
            model_artifact.add_file(f"{self.model_directory}/{self.get_hash()}.pt")
            wandb.log_artifact(model_artifact)
