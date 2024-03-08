"""Train.py is the main script for training the model and will take in the raw data and output a trained model."""
import os
import warnings
from contextlib import nullcontext
from pathlib import Path

import hydra
import numpy as np
import wandb
from distributed import Client
from epochalyst.logging.section_separator import print_section_separator
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from src.config.train_config import TrainConfig
from src.logging_utils.logger import logger
from src.utils.script.lock import Lock
from src.utils.seed_torch import set_torch_seed
from src.utils.setup import setup_config, setup_data, setup_label_data, setup_pipeline, setup_wandb

warnings.filterwarnings("ignore", category=UserWarning)
# Makes hydra give full error messages
os.environ["HYDRA_FULL_ERROR"] = "1"
# Set up the config store, necessary for type checking of config yaml
cs = ConfigStore.instance()
cs.store(name="base_train", node=TrainConfig)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def run_train(cfg: DictConfig) -> None:
    """Train a model pipeline with a train-test split. Entry point for Hydra which loads the config file."""
    # Run the train config with a dask client, and optionally a lock
    optional_lock = Lock if not cfg.allow_multiple_instances else nullcontext
    with optional_lock(), Client() as client:
        logger.info(f"Client: {client}")
        run_train_cfg(cfg)


def run_train_cfg(cfg: DictConfig) -> None:  # TODO(Jeffrey): Use TrainConfig instead of DictConfig
    """Train a model pipeline with a train-test split."""
    print_section_separator("Q3 Detect Harmful Brain Activity - Training")
    set_torch_seed()

    import coloredlogs

    coloredlogs.install()

    # Check for missing keys in the config file
    setup_config(cfg)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    if cfg.wandb.enabled:
        setup_wandb(cfg, "train", output_dir)

    # Preload the pipeline and save it to HTML
    print_section_separator("Setup pipeline")
    model_pipeline = setup_pipeline(cfg, is_train=True)

    # Cache arguments for x_sys
    cache_args = {
        "output_data_type": "numpy_array",
        "storage_type": ".pkl",
        "storage_path": "data/processed",
    }

    # Read the data if required and split it in X, y
    if model_pipeline.x_sys._cache_exists(model_pipeline.x_sys.get_hash(), cache_args) and not model_pipeline.y_sys._cache_exists(model_pipeline.y_sys.get_hash(), cache_args):  # noqa: SLF001
        # Only read y data
        logger.info("x_sys has an existing cache, only loading in labels")
        X = None
        y = setup_label_data(cfg.raw_path)
        indices = np.arange(len(y))
    else:
        X, y = setup_data(raw_path=cfg.raw_path)
        indices = np.arange(len(X.meta))
    if y is None:
        raise ValueError("No labels loaded to train with")
    # Split indices into train and test
    if cfg.test_size == 0:
        train_indices, test_indices = list(indices), []
    else:
        train_indices, test_indices = train_test_split(indices, test_size=cfg.test_size, random_state=42)
    logger.info(f"Train/Test size: {len(train_indices)}/{len(test_indices)}")

    # Generate the parameters for training
    # fit_params = generate_train_params(cfg, model_pipeline, train_indices=train_indices, test_indices=test_indices)

    print_section_separator("Train model pipeline")
    train_args = {
        "x_sys": {
            "cache_args": cache_args,
        },
    }
    predictions, _ = model_pipeline.train(X, y, **train_args)

    if len(test_indices) > 0:
        print_section_separator("Scoring")
        scorer = instantiate(cfg.scorer)
        score = scorer(y[test_indices], predictions[test_indices])
        logger.info(f"Score: {score}")

        if wandb.run:
            wandb.log({"Score": score})

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    run_train()
