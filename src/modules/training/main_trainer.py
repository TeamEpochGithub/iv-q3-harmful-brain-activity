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
    :param dataset_args: The MainDataset args to use for training.
    :param two_stage: Whether to use two-stage training. See: https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/477461
    :param two_stage_kl_threshold: The threshold for dividing the dataset into two stages.
    :param two_stage_evaluator_threshold: The threshold for dividing the dataset into two stages, based on total number of votes.
    :param two_stage_pretrain_full: Whether to train the first stage on the full dataset.
    :param two_stage_split_test: Whether to split the test data into two stages as well.
    :param early_stopping: Whether to do early stopping.
     Note: remove the sum to one block from the target pipeline for this to work
    """

    dataset_args: dict[str, Any] = field(default_factory=dict)
    model_name: str = "WHAT_ARE_YOU_TRAINING_PUT_A_NAME_IN_THE_MAIN_TRAINER"  # No spaces allowed
    two_stage: bool = False
    two_stage_kl_threshold: float | None = None
    two_stage_evaluator_threshold: int | None = None
    two_stage_pretrain_full: bool = False
    two_stage_split_test: bool = False
    early_stopping: bool = True
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
            test_dataset_args = self.dataset_args.copy()
            test_dataset_args["subsample_method"] = "random"
            test_dataset = MainDataset(X=test_data, y=test_labels, use_aug=False, **test_dataset_args)
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
        pred_args = self.dataset_args.copy()
        pred_args["subsample_method"] = None
        predict_dataset = MainDataset(X=x, use_aug=False, **pred_args)
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
        # Create a new dataloader from the dataset of the input dataloader with collate_fn
        loader = DataLoader(loader.dataset, batch_size=loader.batch_size, shuffle=False, collate_fn=collate_fn)  # type: ignore[arg-type]
        with torch.no_grad(), tqdm(loader, unit="batch", disable=False) as tepoch:
            for data in tepoch:
                X_batch = data[0].to(self.device).float()
                y_pred = self.model(X_batch).cpu()
                predictions.extend(y_pred)
        self.log_to_terminal("Done predicting")
        return torch.stack(predictions)

    def custom_train(
        self,
        x: XData,
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
        test_indices = np.array(train_args.get("test_indices", []))

        # Split data according to the chosen criterion
        train_indices_stage1, train_indices_stage2 = self._split_criterion(train_indices, y)
        if self.two_stage_split_test:
            test_indices_stage1, test_indices_stage2 = self._split_criterion(test_indices, y)
        else:
            test_indices_stage1, test_indices_stage2 = list(test_indices), list(test_indices)
        self.log_to_terminal(
            f"Split data into two stages, train sizes: {len(train_indices_stage1)} / {len(train_indices_stage2)},"
            f" test sizes: {len(test_indices_stage1)} / {len(test_indices_stage2)}",
        )

        self._stage = 0
        self.log_to_terminal("Training stage 1")
        train_args["train_indices"] = train_indices_stage1
        train_args["test_indices"] = test_indices_stage1
        super().custom_train(x, y, **train_args)

        self._stage = 1
        self.log_to_terminal("Training stage 2")
        train_args["train_indices"] = train_indices_stage2
        train_args["test_indices"] = test_indices_stage2
        if self.two_stage_split_test:
            super().custom_train(x, y, **train_args)

            # predict again on the entire test data for scoring later on to work
            test_meta = x.meta.iloc[test_indices, :]
            x_test = XData(x.eeg, x.kaggle_spec, x.eeg_spec, test_meta, x.shared)
            return self.custom_predict(x_test), y  # type: ignore[return-value]
        return super().custom_train(x, y, **train_args)

    def _split_criterion(self, indices: npt.NDArray[np.float32], y: npt.NDArray[np.float32]) -> tuple[list[int], list[int]]:
        """Split the indices based on the criterion from the two stage parameters.

        :param indices: The indices to split.
        :param y: The target variable.
        :return: The indices for the two stages.
        """
        if self.two_stage_kl_threshold is not None and self.two_stage_evaluator_threshold is not None:
            raise ValueError("Cannot use both KL and evaluator threshold for two-stage training")

        if self.two_stage_kl_threshold is not None:
            peak_kl = self.compute_peak_kl(y[indices])
            indices_1 = indices[peak_kl >= self.two_stage_kl_threshold]
            indices_2 = indices[peak_kl < self.two_stage_kl_threshold]
        elif self.two_stage_evaluator_threshold is not None:
            n_evaluators = y[indices].sum(axis=1)
            indices_1 = indices[n_evaluators <= self.two_stage_evaluator_threshold]
            indices_2 = indices[n_evaluators > self.two_stage_evaluator_threshold]
        else:
            raise ValueError("No two-stage threshold provided, set either two_stage_kl_threshold or two_stage_evaluator_threshold")

        if self.two_stage_pretrain_full:
            indices_1 = indices

        return list(indices_1), list(indices_2)

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

    def create_dataloaders(
        self,
        train_dataset: Dataset[tuple[Tensor, ...]],
        test_dataset: Dataset[tuple[Tensor, ...]],
    ) -> tuple[DataLoader[tuple[Tensor, ...]], DataLoader[tuple[Tensor, ...]]]:
        """Create the dataloaders for training and validation.

        :param train_dataset: The training dataset.
        :param test_dataset: The validation dataset.
        :return: The training and validation dataloaders.
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,  # type: ignore[arg-type]
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,  # type: ignore[arg-type]
        )
        return train_loader, test_loader

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
            collate_fn=collate_fn,  # type: ignore[arg-type]
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

        test_predictions = torch.stack(predictions)
        return torch.mean(test_predictions, dim=0)

    def _train_one_epoch(
        self,
        dataloader: DataLoader[tuple[Tensor, ...]],
        epoch: int,
    ) -> float:
        """Train the model for one epoch.

        :param dataloader: The dataloader to train on.
        :param epoch: The current epoch.
        :return: The loss for the epoch.
        """
        self.log_to_terminal(f"Learning rate: {self.initialized_optimizer.param_groups[0]['lr']}")
        return super()._train_one_epoch(dataloader, epoch)

    def _early_stopping(self) -> bool:
        """Check if early stopping should be done.

        :return: Whether to do early stopping.
        """
        if not self.early_stopping:
            return False
        return super()._early_stopping()


def collate_fn(batch: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    """Collate function for the dataloader.

    :param batch: The batch to collate.
    :return: The collated batch.
    """
    X, y = batch
    return X, y