"""Common functions used at the start of the main scripts train.py, cv.py, and submit.py."""
import concurrent.futures
import itertools
import os
import pickle
import re
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow.parquet as pq
import torch
import wandb
from epochalyst.pipeline.model.model import ModelPipeline
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.logging_utils.logger import logger
from src.typing.typing import XData
from src.utils.replace_list_with_dict import replace_list_with_dict


def setup_config(cfg: DictConfig) -> None:
    """Verify that config has no missing values and log it to yaml.

    :param cfg: The config object. Created with Hydra or OmegaConf.
    """
    # Check for missing keys in the config file
    missing = OmegaConf.missing_keys(cfg)

    # If both model and ensemble are specified, raise an error
    if cfg.get("model") and cfg.get("ensemble"):
        raise ValueError("Both model and ensemble specified in config.")

    # If neither model nor ensemble are specified, raise an error
    if not cfg.get("model") and not cfg.get("ensemble"):
        raise ValueError("Neither model nor ensemble specified in config.")

    # If model and ensemble are in missing raise an error
    if "model" in missing and "ensemble" in missing:
        raise ValueError("Both model and ensemble are missing from config.")

    # If any other keys except model and ensemble are missing, raise an error
    if len(missing) > 1:
        raise ValueError(f"Missing keys in config: {missing}")


def setup_pipeline(pipeline_cfg: DictConfig, is_train: bool | None) -> ModelPipeline:
    """Instantiate the pipeline and log it to HTML.

    :param pipeline_cfg: The model pipeline config. Root node should be a ModelPipeline
    :param output_dir: The directory to save the pipeline to.
    :param is_train: Whether the pipeline is for training or not.
    """
    logger.info("Instantiating the pipeline")

    test_size = pipeline_cfg.get("test_size", -1)

    if "model" in pipeline_cfg:
        model_cfg = pipeline_cfg.model

        # Add test size to the config
        model_cfg_dict = OmegaConf.to_container(model_cfg, resolve=True)
        model_cfg_dict = update_model_cfg_test_size(model_cfg_dict, test_size, is_train=is_train)

        cfg = OmegaConf.create(model_cfg_dict)

    elif "ensemble" in pipeline_cfg:
        ensemble_cfg = pipeline_cfg.ensemble

        ensemble_cfg_dict = OmegaConf.to_container(ensemble_cfg, resolve=True)
        if isinstance(ensemble_cfg_dict, dict):
            for model in ensemble_cfg_dict.get("models", []):
                ensemble_cfg_dict["models"][model] = update_model_cfg_test_size(ensemble_cfg_dict["models"][model], test_size, is_train=is_train)

        cfg = OmegaConf.create(ensemble_cfg_dict)

    model_pipeline = instantiate(cfg)

    logger.debug(f"Pipeline: \n{model_pipeline}")

    return model_pipeline


def update_model_cfg_test_size(
    model_cfg_dict: dict[str | bytes | int | Enum | float | bool, Any] | list[Any] | str | None,
    test_size: int = -1,
    *,
    is_train: bool | None,
) -> dict[str | bytes | int | Enum | float | bool, Any] | list[Any] | str | None:
    """Update the test size in the model config.

    :param cfg: The model config.
    :param test_size: The test size.

    :return: The updated model config.
    """
    if isinstance(model_cfg_dict, dict):
        for model_block in model_cfg_dict.get("model_loop_pipeline", {}).get("model_blocks_pipeline", {}).get("model_blocks", []):
            model_block["test_size"] = test_size
        for pretrain_block in model_cfg_dict.get("model_loop_pipeline", {}).get("pretrain_pipeline", {}).get("pretrain_steps", []):
            pretrain_block["test_size"] = test_size

        if not is_train:
            model_cfg_dict.get("feature_pipeline", {})["processed_path"] = "data/test"
            model_cfg_dict.get("model_loop_pipeline", {}).get("pretrain_pipeline", {})["pretrain_path"] = "data/test"
    return model_cfg_dict


def setup_data(
    raw_path: str,
) -> tuple[XData, pd.DataFrame | None]:
    """Read the metadata and return the data and target in the proper format.

    :param metadata_path: Path to the metadata.
    :param eeg_path: Path to the EEG data.
    :param spectrogram_path: Path to the spectrogram data.
    """
    # Turn raw path into separate paths
    raw_path = raw_path if raw_path[-1] == "/" else raw_path + "/"
    metadata_path = raw_path + "train.csv"
    eeg_path = raw_path + "train_eegs"
    spectrogram_path = raw_path + "train_spectrograms"

    # Check that metadata_path is not None
    if metadata_path is None:
        raise ValueError("metadata_path should not be None")

    # Check that at least one of the paths is not None
    if eeg_path is None and spectrogram_path is None:
        raise ValueError("At least one of the paths should not be None")

    # Read the metadata
    metadata = pd.read_csv(metadata_path)
    # Now split the metadata into the 3 parts: ids, offsets, and labels
    ids = metadata[["patient_id", "eeg_id", "spectrogram_id"]]

    if "eeg_label_offset_seconds" in metadata.columns and "spectrogram_label_offset_seconds" in metadata.columns:
        # If the offsets exist in metadata, use them
        offsets = metadata[["eeg_label_offset_seconds", "spectrogram_label_offset_seconds"]]
    else:
        # Ifthe offsets do not exist fill them with zeros
        offsets = pd.DataFrame(np.zeros((metadata.shape[0], 2)), columns=["eeg_label_offset_seconds", "spectrogram_label_offset_seconds"])
    label_columns = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]

    if all(column in metadata.columns for column in label_columns):
        labels = metadata[label_columns]
    else:
        labels = None

    # Get one of the paths that is not None
    path = eeg_path if eeg_path is not None else spectrogram_path

    cache_loc = "train" if "train" in path else "test"

    # Get the cache path
    cache_path = f"data/processed/{cache_loc}"
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    if eeg_path is not None:
        # Initialize the dictionary to store the EEG data
        all_eegs = load_all_eegs(eeg_path, cache_path, ids)
    else:
        logger.info("No EEG data to read, skipping...")
        all_eegs = None

    if spectrogram_path is not None:
        all_spectrograms = load_all_spectrograms(spectrogram_path, cache_path, ids)
    else:
        logger.info("No spectrogram data to read, skipping...")
        all_spectrograms = None

    X_meta = pd.concat([ids, offsets], axis=1)

    return XData(eeg=all_eegs, kaggle_spec=all_spectrograms, eeg_spec=None, meta=X_meta), labels

def setup_label_data(
        
)

def load_eeg(eeg_path: str, eeg_id: int) -> tuple[int, pd.DataFrame]:
    """Load the EEG data from the parquet file.

    :param eeg_path: The path to the EEG data.
    :param eeg_id: The EEG id.
    """
    return eeg_id, pq.read_table(f"{eeg_path}/{eeg_id}.parquet").to_pandas()


def load_spectrogram(spectrogram_path: str, spectrogram_id: int) -> tuple[int, npt.NDArray[np.float32]]:
    """Load the spectrogram data from the parquet file.

    :param spectrogram_path: The path to the spectrogram data.
    :param spectrogram_id: The spectrogram id.
    """
    data = pd.read_parquet(f"{spectrogram_path}/{spectrogram_id}.parquet")
    LL = data.filter(regex="^LL")
    LP = data.filter(regex="^LP")
    RP = data.filter(regex="^RP")
    RL = data.filter(regex="^RL")

    spectrogram = np.stack(
        [
            LL.to_numpy().T,
            LP.to_numpy().T,
            RP.to_numpy().T,
            RL.to_numpy().T,
        ],
    )

    return spectrogram_id, spectrogram


def load_all_eegs(eeg_path: str, cache_path: str, ids: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Read the EEG data and return it as a dictionary.

    :param eeg_path: Path to the EEG data.
    :param eeg_ids: The EEG ids.
    """
    all_eegs = {}
    # Read the EEG data
    logger.info("Reading the EEG data")
    if os.path.exists(cache_path + "/eeg_cache.pkl"):
        logger.info("Found pickle cache for EEG data at: " + cache_path + "/eeg_cache.pkl")
        with open(cache_path + "/eeg_cache.pkl", "rb") as f:
            all_eegs = pickle.load(f)  # noqa: S301
        logger.info("Loaded pickle cache for EEG data")
    else:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            all_eegs = dict(executor.map(load_eeg, itertools.repeat(eeg_path), ids["eeg_id"].unique()))
            executor.shutdown()
        logger.info("Finished reading the EEG data")
        logger.info("Saving pickle cache for EEG data")
        with open(cache_path + "/eeg_cache.pkl", "wb") as f:
            pickle.dump(all_eegs, f)
        logger.info("Saved pickle cache for EEG data to: " + cache_path + "/eeg_cache.pkl")

    return all_eegs


def load_all_spectrograms(spectrogram_path: str, cache_path: str, ids: pd.DataFrame) -> dict[int, torch.Tensor]:
    """Read the spectrogram data and return it as a dictionary.

    :param spectrogram_path: Path to the spectrogram data.
    :param spectrogram_ids: The spectrogram ids.
    """
    all_spec = {}
    # Read the spectrogram data
    logger.info("Reading the spectrogram data")
    if os.path.exists(cache_path + "/spectrogram_cache.pkl"):
        logger.info("Found pickle cache for spectrogram data at: " + cache_path + "/spectrogram_cache.pkl")
        with open(cache_path + "/spectrogram_cache.pkl", "rb") as f:
            all_spec = pickle.load(f)  # noqa: S301
        logger.info("Loaded pickle cache for spectrogram data")
    else:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            all_spec = dict(executor.map(load_spectrogram, itertools.repeat(spectrogram_path), ids["spectrogram_id"].unique()))
            executor.shutdown()
        for spectrogram_id in all_spec:
            all_spec[spectrogram_id] = torch.tensor(all_spec[spectrogram_id])
        logger.info("Finished reading the spectrogram data")
        logger.info("Saving pickle cache for spectrogram data")
        with open(cache_path + "/spectrogram_cache.pkl", "wb") as f:
            pickle.dump(all_spec, f)
        logger.info("Saved pickle cache for spectrogram data to: " + cache_path + "/spectrogram_cache.pkl")

    return all_spec


def setup_wandb(
    cfg: DictConfig,
    job_type: str,
    output_dir: Path,
    name: str | None = None,
    group: str | None = None,
) -> wandb.sdk.wandb_run.Run | wandb.sdk.lib.RunDisabled | None:
    """Initialize Weights & Biases and log the config and code.

    :param cfg: The config object. Created with Hydra or OmegaConf.
    :param job_type: The type of job, e.g. Training, CV, etc.
    :param output_dir: The directory to the Hydra outputs.
    :param name: The name of the run.
    :param group: The namer of the group of the run.
    """
    logger.debug("Initializing Weights & Biases")

    config = OmegaConf.to_container(cfg, resolve=True)
    run = wandb.init(
        config=replace_list_with_dict(config),  # type: ignore[arg-type]
        project="detect-harmful-brain-activity",
        entity="team-epoch-iv",
        name=name,
        group=group,
        job_type=job_type,
        tags=cfg.wandb.tags,
        notes=cfg.wandb.notes,
        settings=wandb.Settings(start_method="thread", code_dir="."),
        dir=output_dir,
        reinit=True,
    )

    if isinstance(run, wandb.sdk.lib.RunDisabled) or run is None:  # Can't be True after wandb.init, but this casts wandb.run to be non-None, which is necessary for MyPy
        raise RuntimeError("Failed to initialize Weights & Biases")

    if cfg.wandb.log_config:
        logger.debug("Uploading config files to Weights & Biases")

        # Get the config file name
        if job_type == "sweep":
            job_type = "cv"
        curr_config = "conf/" + job_type + ".yaml"

        # Get the model file name
        if "model" in cfg:
            model_name = OmegaConf.load(curr_config).defaults[2].model
            model_path = f"conf/model/{model_name}.yaml"
        elif "ensemble" in cfg:
            model_name = OmegaConf.load(curr_config).defaults[2].ensemble
            model_path = f"conf/ensemble/{model_name}.yaml"

        # Store the config as an artefact of W&B
        artifact = wandb.Artifact(job_type + "_config", type="config")
        config_path = output_dir / ".hydra/config.yaml"
        artifact.add_file(str(config_path), "config.yaml")
        artifact.add_file(curr_config)
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

    if cfg.wandb.log_code.enabled:
        logger.debug("Uploading code files to Weights & Biases")

        run.log_code(
            root=".",
            exclude_fn=cast(Callable[[str, str], bool], lambda abs_path, root: re.match(cfg.wandb.log_code.exclude, Path(abs_path).relative_to(root).as_posix()) is not None),
        )

    logger.info("Done initializing Weights & Biases")
    return run
