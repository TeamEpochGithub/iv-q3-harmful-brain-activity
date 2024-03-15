"""The main script for Cross Validation. Takes in the raw data, does CV and logs the results."""
import os
import warnings
from contextlib import nullcontext
from pathlib import Path

import hydra
import numpy as np
import randomname
from epochalyst.logging.section_separator import print_section_separator
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig

import wandb
from src.config.cross_validation_config import CVConfig
from src.logging_utils.logger import logger
from src.typing.typing import XData
from src.utils.script.lock import Lock
from src.utils.script.reset_wandb_env import reset_wandb_env
from src.utils.seed_torch import set_torch_seed
from src.utils.setup import setup_config, setup_data, setup_label_data, setup_pipeline, setup_wandb

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
    raw_path = Path(cfg.raw_path)
    cache_path = Path(cfg.cache_path)
    if model_pipeline.x_sys._cache_exists(model_pipeline.x_sys.get_hash(), cache_args) and not model_pipeline.y_sys._cache_exists(model_pipeline.y_sys.get_hash(), cache_args):  # noqa: SLF001
        # Only read y data
        logger.info("x_sys has an existing cache, only loading in labels")
        X = None
        y = setup_label_data(raw_path)
    else:
        X, y = setup_data(raw_path, cache_path)
    if y is None:
        raise ValueError("No labels loaded to train with")

    if model_pipeline.x_sys is not None:
        X = model_pipeline.x_sys.transform(X, cache_args=cache_args)

    if model_pipeline.y_sys is not None:
        processed_y = model_pipeline.y_sys.transform(y)

    indices = np.arange(len(y))

    # TODO(Jasper): Replace with actual splitter
    from sklearn.model_selection import KFold

    skf = KFold(3)

    scorer = instantiate(cfg.scorer)

    for i, (train_indices, test_indices) in enumerate(skf.split(np.zeros(len(indices)), indices)):
        # https://github.com/wandb/wandb/issues/5119
        # This is a workaround for the issue where sweeps override the run id annoyingly
        reset_wandb_env()

        # Print section separator
        print_section_separator(f"CV - Fold {i}")
        logger.info(f"Train/Test size: {len(train_indices)}/{len(test_indices)}")

        if cfg.wandb.enabled:
            setup_wandb(cfg, "cv", output_dir, name=f"{wandb_group_name}_{i}", group=wandb_group_name)

        logger.info("Creating clean pipeline for this fold")
        model_pipeline = setup_pipeline(cfg, is_train=True)

        # Fit the pipeline and get predictions
        predictions = X

        train_args = {
            "MainTrainer": {
                "train_indices": train_indices,
                "test_indices": test_indices,
                "save_model": False,
            },
        }

        if model_pipeline.train_sys is not None:
            predictions, _ = model_pipeline.train_sys.train(X, processed_y, **train_args)

        if model_pipeline.pred_sys is not None:
            predictions = model_pipeline.pred_sys.transform(predictions)

        if predictions is None or isinstance(predictions, XData):
            raise ValueError("Predictions are not in correct format to get a score")

        score = scorer(y[test_indices], predictions[test_indices])
        logger.info(f"Score: {score}")
        wandb.log({"Score": score})

        logger.info("Finishing wandb run")
        wandb.finish()


if __name__ == "__main__":
    run_cv()
