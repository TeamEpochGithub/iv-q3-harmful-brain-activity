"""Visualize the vote distribution of the training and validation sets."""
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def visualize_vote_distribution(y: npt.NDArray[np.float32], train_indices: Iterable[int], test_indices: Iterable[int]) -> None:
    """Visualize the vote distribution of the training and validation sets."""
    train_indices = np.array(list(train_indices))
    test_indices = np.array(list(test_indices))
    vote_columns = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
    # Aggregate vote counts for the training set
    train_votes = y[train_indices].sum(axis=0)
    train_votes = train_votes / train_votes.sum()  # Normalize to get the proportion of votes
    # Aggregate vote counts for the validation set
    test_votes = y[test_indices].sum(axis=0)
    test_votes = test_votes / test_votes.sum()  # Normalize to get the proportion of votes
    # Setting up the subplot for training and validation vote distribution
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Training set bar chart
    ax[0].bar(range(len(train_votes)), train_votes, color="skyblue")
    ax[0].set_title("Training Set Vote Distribution")
    ax[0].set_xticks(range(len(train_votes)))  # Set the x-ticks locations
    ax[0].set_xticklabels(vote_columns, rotation=45, ha="right")
    ax[0].set_ylabel("Normalized Vote Count")

    # Validation set bar chart
    ax[1].bar(range(len(test_votes)), test_votes, color="orange")
    ax[1].set_title("Validation Set Vote Distribution")
    ax[0].set_xticks(range(len(test_votes)))  # Set the x-ticks locations
    ax[1].set_xticklabels(vote_columns, rotation=45, ha="right")

    fig.tight_layout()
    fig.show()
