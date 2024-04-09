"""KLDiv scorer class."""
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.nn import KLDivLoss
from torch.nn.functional import softmax

from src.scoring.scorer import Scorer


class KLDiv(Scorer):
    """Abstract scorer class from which other scorers inherit from."""

    voter_threshold: int
    kldiv_thresholds: list[float] | None

    def __init__(self, name: str = "KLDiv", voter_threshold: int = 0, kldiv_thresholds: list[float] | None = None) -> None:
        """Initialize the scorer with a name.

        :param name: The name of the scorer.
        :param voter_threshold: Only include samples with more (exclusive) than voter_threshold voters.
        """
        super().__init__(name)
        self.voter_threshold = voter_threshold
        self.kldiv_thresholds = kldiv_thresholds

    def __call__(self, y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any], **kwargs: pd.DataFrame) -> float:
        """Calculate the Kullback-Leibler divergence between two probability distributions.

        :param y_true: The true labels.
        :param y_pred: The predicted labels.
        :return: The Kullback-Leibler divergence between the two probability distributions.
        """
        # Filter out samples with less than voter_threshold voters (sum of y)
        if self.voter_threshold is not None:
            indices = y_true.sum(axis=1) > self.voter_threshold
        else:
            indices = np.ones(y_true.shape[0], dtype=bool)

        # Normalize the true labels to be a probability distribution
        y_true = y_true / y_true.sum(axis=1)[:, None]

        # Get the metadata
        metadata = kwargs.get("metadata", None).reset_index(drop=True)  # type: ignore[union-attr]
        if metadata is None or not isinstance(metadata, pd.DataFrame):
            return -1

        # Filter the metadata
        y_true = y_true[indices]
        y_pred = y_pred[indices]
        metadata = metadata.iloc[indices].reset_index(drop=True)

        grouped_eeg_id = metadata.groupby("eeg_id")
        scores = grouped_eeg_id.apply(lambda group: self.score_group(group, y_true, y_pred))

        return scores.mean()

    def score_group(self, group: pd.DataFrame, y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]) -> float:
        """Calculate the Kullback-Leibler divergence between two probability distributions.

        :param group: The group to calculate the score for.
        :param y_true: The true labels.
        :param y_pred: The predicted labels.
        :return: The Kullback-Leibler divergence between the two probability distributions.
        """
        # Get the indices of the current group
        indices = group.index
        # Get the true and predicted labels for the current group
        y_true_group = y_true[indices]
        y_pred_group = y_pred[indices]

        # Calculate the KLDiv for the current group
        return self.calc_kldiv(y_true_group, y_pred_group)

    def calc_kldiv(self, y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]) -> float:
        """Calculate the Kullback-Leibler divergence between two probability distributions.

        :param y_true: The true labels.
        :param y_pred: The predicted labels.
        :return: The Kullback-Leibler divergence between the two probability distributions.
        """
        # Convert both to torch tensors
        y_pred = torch.tensor(y_pred)  # type: ignore[assignment]

        target = torch.tensor(y_true)

        # Calculate the KLDivLoss
        criterion = KLDivLoss(reduction="batchmean")
        return criterion(torch.log(torch.clamp(y_pred, min=10**-15, max=1 - 10**-15)), target)  # type: ignore[call-overload]

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return self.name

    def visualize_preds(self, y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any], output_folder: Path, **kwargs: pd.DataFrame) -> tuple[float, float]:
        """Visualize the predictions.

        :param y_true: The true labels.
        :param y_pred: The predicted labels.
        :param output_folder: The output folder to save the visualization.

        :return: The accuracy and the f1 score of the predictions.
        """
        # Suppress warnings
        warnings.filterwarnings("ignore")
        logger = logging.getLogger("matplotlib")
        logger.setLevel(logging.WARNING)

        # First without vother threshold
        self._visualize_kldiv(y_true, y_pred, output_folder / "kldiv.png")
        self._visualize_confusion_matrix(y_true, y_pred, output_folder / "confusion_matrix.png")
        self._visualize_label_distribution(y_true, y_pred, output_folder / "label_distribution.png")
        self._visualize_num_voters(y_true, y_pred, output_folder / "num_voters_correct.png")
        accuracy, f1 = self._calculate_metrics(y_true, y_pred)

        # Then with voter threshold
        if self.voter_threshold is not None and self.voter_threshold > 0:
            indices = y_true.sum(axis=1) > self.voter_threshold
            y_true = y_true[indices]
            y_pred = y_pred[indices]

            self._visualize_kldiv(y_true, y_pred, output_folder / "kldiv_vt.png")
            self._visualize_confusion_matrix(y_true, y_pred, output_folder / "confusion_matrix_vt.png")
            self._visualize_label_distribution(y_true, y_pred, output_folder / "label_distribution_vt.png")
            self._visualize_num_voters(y_true, y_pred, output_folder / "num_voters_correct_vt.png")
            accuracy, f1 = self._calculate_metrics(y_true, y_pred)

        return accuracy, f1

    def _visualize_kldiv(self, y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any], output_file: Path) -> None:
        if self.kldiv_thresholds is None:
            return

        y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)
        y_true_tensor = torch.tensor(y_true, dtype=torch.float32)

        # Convert y_pred to log probabilities for KLDivLoss
        y_pred_tensor = torch.log(y_pred_tensor)
        y_true_tensor = softmax(y_true_tensor, dim=1)

        # Initialize KLDivLoss with no reduction to get individual scores
        kldiv_criteria = KLDivLoss(reduction="none")
        kldiv_scores = kldiv_criteria(y_pred_tensor, y_true_tensor)
        kldiv_scores = kldiv_scores.mean(dim=1).numpy()

        # Create a matrix to store the percentage for each threshold and category
        threshold_percentages = np.zeros((y_true_tensor.shape[1], len(self.kldiv_thresholds)))

        total_predictions_per_category = (y_pred_tensor.argmax(dim=1).numpy()[:, None] == np.arange(y_true_tensor.shape[1])).sum(axis=0)

        # Calculate percentages for each category and threshold
        for i, threshold in enumerate(self.kldiv_thresholds):
            below_threshold = kldiv_scores < threshold
            for category in range(y_true_tensor.shape[1]):
                pred_category = y_pred_tensor.argmax(dim=1).numpy() == category
                correct_and_below_threshold = np.sum(below_threshold & pred_category)
                # Convert to percentage
                if total_predictions_per_category[category] > 0:
                    threshold_percentages[category, i] = (correct_and_below_threshold / total_predictions_per_category[category]) * 100
                else:
                    threshold_percentages[category, i] = 100

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ["Seizure", "Lpd", "Gpd", "Lrda", "Grda", "Other"]
        im = ax.imshow(threshold_percentages, cmap="viridis", aspect="auto", vmin=0, vmax=100)

        # Label the axis
        ax.set_xticks(np.arange(len(self.kldiv_thresholds)))
        ax.set_yticks(np.arange(len(categories)))
        ax.set_xticklabels(self.kldiv_thresholds)
        ax.set_yticklabels(categories)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(categories)):
            for j in range(len(self.kldiv_thresholds)):
                percentage = f"{threshold_percentages[i, j]:.1f}%"
                ax.text(j, i, percentage, ha="center", va="center", color="w" if threshold_percentages[i, j] < 50 else "black")

        ax.set_title("Percentage of Predictions below KL-Divergence Thresholds")
        ax.set_xlabel("KL-Divergence Thresholds")
        ax.set_ylabel("Categories")
        fig.tight_layout()
        plt.colorbar(im, label="Percentage (%)")

        # Ensure the output folder exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file)

    def _visualize_confusion_matrix(self, y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any], output_file: Path) -> None:
        # Get the labels
        label_names = np.array(["Seizure", "Lpd", "Gpd", "Lrda", "Grda", "Other"])
        y_pred_final = np.argmax(y_pred, axis=1)
        y_true_final = np.argmax(y_true, axis=1)

        # Create the confusion matrix
        confusion = confusion_matrix(y_true_final, y_pred_final)

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(confusion, annot=True, fmt="d", ax=ax)

        # Set the labels
        ax.set_xticklabels(label_names)
        ax.set_yticklabels(label_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")

        # Ensure the output folder exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file)

    def _visualize_label_distribution(self, y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any], output_file: Path) -> None:
        # Convert the array to a pandas DataFrame
        df_pred = pd.DataFrame(y_pred, columns=["Seizure", "Lpd", "Gpd", "Lrda", "Grda", "Other"])
        df_true = pd.DataFrame(y_true, columns=["Seizure", "Lpd", "Gpd", "Lrda", "Grda", "Other"])

        # Melt the DataFrame to long format
        df_pred_long = df_pred.melt(var_name="Label", value_name="Value")
        df_true_long = df_true.melt(var_name="Label", value_name="Value")

        # Plotting
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        sns.kdeplot(data=df_pred_long, x="Value", hue="Label", fill=True, ax=ax[0])
        sns.kdeplot(data=df_true_long, x="Value", hue="Label", fill=True, ax=ax[1])

        ax[0].set_title("Predicted label distribution")
        ax[1].set_title("True label distribution")

        # Ensure the output folder exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file)
        plt.close()

    def _visualize_num_voters(self, y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any], output_file: Path) -> None:
        label_names = np.array(["Seizure", "Lpd", "Gpd", "Lrda", "Grda", "Other"])

        y_pred_final = np.argmax(y_pred, axis=1)
        y_true_final = np.argmax(y_true, axis=1)

        num_voters = y_true.sum(axis=1)
        expert_consensus = label_names[y_true_final]
        predict_consensus = label_names[y_pred_final]
        correct = y_pred_final == y_true_final

        all_df = pd.DataFrame(
            {"y_true": y_true_final, "y_pred": y_pred_final, "acc": correct, "num_voters": num_voters, "true_name": expert_consensus, "pred_name": predict_consensus},
        )

        # Plotting
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        sns.barplot(data=all_df, x="num_voters", y="acc", ax=ax[0])
        sns.countplot(data=all_df, x="num_voters", hue="true_name", ax=ax[1])

        ax[0].set_title("Accuracy based on the number of voters")
        ax[1].set_title("Number of voters based on the true label")

        # Ensure the output folder exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file)

    def _calculate_metrics(self, y_true: np.ndarray[Any, Any], y_pred: np.ndarray[Any, Any]) -> tuple[float, float]:
        """Calculate the accuracy and the f1 score of the predictions.

        :param y_true: The true labels.
        :param y_pred: The predicted labels.
        :return: The accuracy and the f1 score of the predictions.
        """
        # Calculate the accuracy and the f1 score of the predictions use scikit-learn
        accuracy = sklearn.metrics.accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
        f1 = sklearn.metrics.f1_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), average="weighted")

        return accuracy, f1
