"""The main script for Cross Validation. Takes in the raw data, does CV and logs the results."""
import os
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import randomname
import wandb
from epochalyst.logging.section_separator import print_section_separator
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.config.cross_validation_config import CVConfig
from src.logging_utils.logger import logger
from src.scoring.scorer import Scorer
from src.typing.typing import XData
from src.utils.script.lock import Lock
from src.utils.seed_torch import set_torch_seed
from src.utils.setup import load_training_data, setup_config, setup_data, setup_pipeline, setup_wandb

warnings.filterwarnings("ignore", category=UserWarning)
# Makes hydra give full error messages
os.environ["HYDRA_FULL_ERROR"] = "1"

# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_cv", node=CVConfig)


@hydra.main(version_base=None, config_path="conf", config_name="cv")
def run_cv(cfg: DictConfig) -> None:  # TODO(Jeffrey): Use CVConfig instead of DictConfig
    """Do cv on a model pipeline with K fold split. Entry point for Hydra which loads the config file."""
    # Run the cv config with a dask client, and optionally a lock
    optional_lock = Lock if not cfg.allow_multiple_instances else nullcontext
    with optional_lock():
        run_cv_cfg(cfg)


def run_cv_cfg(cfg: DictConfig) -> None:
    """Do cv on a model pipeline with K fold split."""
    print_section_separator("Q3 Detect Harmful Brain Activity - CV")
    X: XData | None
    import coloredlogs

    coloredlogs.install()

    # Set seed
    set_torch_seed()

    # Check for missing keys in the config file
    setup_config(cfg)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # Set up Weights & Biases group name
    wandb_group_name = randomname.get_name()

    model_pipeline = setup_pipeline(cfg, is_train=True)

    processed_data_path = Path(cfg.processed_path)
    processed_data_path.mkdir(parents=True, exist_ok=True)
    # Cache arguments for x_sys
    cache_args = {
        "output_data_type": "numpy_array",
        "storage_type": ".pkl",
        "storage_path": f"{processed_data_path}",
    }

    # Read the data if required and split in X, y

    x_cache_exists = model_pipeline.x_sys._cache_exists(model_pipeline.x_sys.get_hash(), cache_args)  # noqa: SLF001
    y_cache_exists = model_pipeline.y_sys._cache_exists(model_pipeline.y_sys.get_hash(), cache_args)  # noqa: SLF001

    X, y = load_training_data(
        metadata_path=cfg.metadata_path,
        eeg_path=cfg.eeg_path,
        spectrogram_path=cfg.spectrogram_path,
        cache_path=cfg.cache_path,
        x_cache_exists=x_cache_exists,
        y_cache_exists=y_cache_exists,
    )

    if y is None:
        raise ValueError("No labels loaded to train with")

    if model_pipeline.x_sys is not None:
        X = model_pipeline.x_sys.transform(X, cache_args=cache_args)

    if model_pipeline.y_sys is not None:
        processed_y = model_pipeline.y_sys.transform(y)

    if X is not None:
        splitter_data = X.meta
    else:
        X, _ = setup_data(cfg.metadata_path, cfg.eeg_path, cfg.spectrogram_path)
        splitter_data = X.meta

    scorer = instantiate(cfg.scorer)

    if cfg.wandb.enabled:
        setup_wandb(cfg, "cv", output_dir, name=wandb_group_name, group=wandb_group_name)

    scores: list[float] = []
    accuracies: list[float] = []
    f1s: list[float] = []

    for i, (train_indices, test_indices) in enumerate(instantiate(cfg.splitter).split(splitter_data, y)):
        score, accuracy, f1 = run_fold(i, X, y, train_indices, test_indices, cfg, scorer, output_dir, processed_y=processed_y)
        scores.append(score)
        accuracies.append(accuracy)
        f1s.append(f1)
        for fold, threshold in [
            (0, 0.42),
            (1, 0.41),
            (2, 0.42),
            (3, 0.41),
        ]:
            if i == fold and np.mean(scores) > threshold:
                logger.info(f"Early stopping at fold {fold} with threshold {threshold}")
                break

    avg_score = np.average(np.array(scores))
    avg_accuracy = np.average(np.array(accuracies))
    avg_f1 = np.average(np.array(f1s))

    print_section_separator("CV - Results")
    logger.info(f"Average Accuracy: {avg_accuracy}")
    logger.info(f"Average F1: {avg_f1}")
    logger.info(f"Score: {avg_score}")
    wandb.log({"Score": avg_score, "Accuracy": avg_accuracy, "F1": avg_f1})
    logger.info("Finishing wandb run")
    wandb.finish()


def run_fold(
    i: int,
    X: XData,
    y: np.ndarray[Any, Any],
    train_indices: np.ndarray[Any, Any],
    test_indices: np.ndarray[Any, Any],
    cfg: DictConfig,
    scorer: Scorer,
    output_dir: Path,
    processed_y: np.ndarray[Any, Any] | None = None,
) -> tuple[float, float, float]:
    """Run a single fold of the cross validation.

    :param i: The fold number.
    :param X: The input data.
    :param y: The labels.
    :param train_indices: The indices of the training data.
    :param test_indices: The indices of the test data.
    :param cfg: The config file.
    :param scorer: The scorer to use.
    :param output_dir: The output directory for the prediction plots.
    :param processed_y: The processed labels.
    :return: The score of the fold.
    """
    # Print section separator
    print_section_separator(f"CV - Fold {i}")
    logger.info(f"Train/Test size: {len(train_indices)}/{len(test_indices)}")

    logger.info("Creating clean pipeline for this fold")
    model_pipeline = setup_pipeline(cfg, is_train=True)

    # Fit the pipeline and get predictions
    predictions = X

    train_args = {
        "MainTrainer": {
            "train_indices": train_indices,
            "test_indices": test_indices,
            "save_model": cfg.save_folds,
            "fold": i,
        },
    }

    if model_pipeline.train_sys is not None:
        predictions, _ = model_pipeline.train_sys.train(X, processed_y, **train_args)

    if model_pipeline.pred_sys is not None:
        predictions = model_pipeline.pred_sys.transform(predictions)

    if predictions is None or isinstance(predictions, XData):
        raise ValueError("Predictions are not in correct format to get a score")

    # Make sure the predictions is the same length as the test indices
    if len(predictions) != len(test_indices):
        raise ValueError("Predictions and test indices are not the same length")

    score = scorer(y[test_indices], predictions, metadata=X.meta.iloc[test_indices, :])

    # Add i to fold path using os.path.join
    output_dir = os.path.join(output_dir, str(i))
    accuracy, f1 = scorer.visualize_preds(y[test_indices], predictions, output_folder=output_dir)
    logger.info(f"Score, fold {i}: {score}")
    logger.info(f"Accuracy, fold {i}: {accuracy}")
    logger.info(f"F1, fold {i}: {f1}")

    wandb.log({f"Score_{i}": score, f"Accuracy_{i}": accuracy, f"F1_{i}": f1})
    return score, accuracy, f1


if __name__ == "__main__":
    run_cv()
