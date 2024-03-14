"""Sweep.py is used to run a sweep on a model pipeline with K fold split. Entry point for Hydra which loads the config file."""
import multiprocessing
import os
import warnings
from contextlib import nullcontext
from multiprocessing import Queue
from pathlib import Path
from typing import NamedTuple

import hydra
import numpy as np
import randomname
import wandb
from distributed import Client
from epochalyst.logging.section_separator import print_section_separator
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.config.cross_validation_config import CVConfig
from src.logging_utils.logger import logger
from src.utils.script.lock import Lock
from src.utils.script.reset_wandb_env import reset_wandb_env
from src.utils.seed_torch import set_torch_seed
from src.utils.setup import setup_config, setup_data, setup_label_data, setup_pipeline, setup_wandb


class Worker(NamedTuple):
    """A worker.

    :param queue: The queue to get the data from
    :param process: The process
    """

    queue: Queue  # type: ignore[type-arg]
    process: multiprocessing.Process


class WorkerInitData(NamedTuple):
    """The data to initialize a worker.

    :param cfg: The configuration
    :param output_dir: The output directory
    :param wandb_group_name: The wandb group name
    :param i: The fold number
    :param train_indices: The train indices
    :param test_indices: The test indices
    :param X: The X data
    :param y: The y data
    """

    cfg: DictConfig
    output_dir: Path
    wandb_group_name: str
    i: int
    train_indices: list[int]
    test_indices: list[int]


class WorkerDoneData(NamedTuple):
    """The data a worker returns.

    :param sweep_score: The sweep score
    """

    sweep_score: float


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
    with optional_lock(), Client() as client:
        logger.info(f"Client: {client}")
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

    # RuntimeError("Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method")
    multiprocessing.set_start_method("spawn", force=True)

    # Read the data if required and split in X, y

    # Read the label data 
    y = setup_label_data(cfg.raw_path)
    if y is None:
        raise ValueError("No labels loaded to train with")

    # Set up Weights & Biases group name
    wandb_group_name = randomname.get_name()

    # Spin up workers before calling wandb.init()
    # Workers will be blocked on a queue waiting to start
    # TODO(Jasper): Return to splitter n_splits
    n_splits = 3  # cfg.splitter.n_splits
    sweep_q: Queue[WorkerDoneData] = multiprocessing.Queue()
    workers = []
    for _ in range(n_splits):
        q: Queue[WorkerInitData] = multiprocessing.Queue()
        p = multiprocessing.Process(target=try_fold_run, kwargs={"sweep_q": sweep_q, "worker_q": q})
        p.start()
        workers.append(Worker(queue=q, process=p))

    # Initialize wandb
    sweep_run = setup_wandb(cfg, "sweep", output_dir, name=wandb_group_name, group=wandb_group_name)

    metrics = []
    failed = False  # If any worker fails, stop the run

    # TODO(Jasper): Replace with actual splitter
    from sklearn.model_selection import KFold

    skf = KFold(n_splits)
    indices = np.arange(len(y))

    for num, (train_indices, test_indices) in enumerate(skf.split(np.zeros(len(indices)), indices)):
        set_torch_seed()
        worker = workers[num]

        # If failed, stop the run
        if failed:
            logger.debug(f"Stopping worker {num}")
            worker.process.terminate()
            continue

        # Start worker
        worker.queue.put(
            WorkerInitData(
                cfg=cfg,
                output_dir=output_dir,
                wandb_group_name=wandb_group_name,
                i=num,
                train_indices=train_indices,
                test_indices=test_indices,
            ),
        )
        # Get metric from worker
        result = sweep_q.get()
        # Wait for worker to finish
        worker.process.join()
        # Log metric to sweep_run
        metrics.append(result.sweep_score)

        # If failed, stop the run by setting failed to True
        if result.sweep_score == -0.1:
            logger.error("Worker failed")
            failed = True
            continue

        # If score is too low, stop the run by setting failed to True
        if result.sweep_score < 0.25:
            logger.debug("Worker score too low, stopping run")
            failed = True
            continue

    if sweep_run is not None:
        sweep_run.log({"sweep_score": sum(metrics) / len(metrics)})
    wandb.join()


def try_fold_run(sweep_q: Queue, worker_q: Queue) -> None:  # type: ignore[type-arg]
    """Run a fold, and catch exceptions.

    :param sweep_q: The queue to put the result in
    :param worker_q: The queue to get the data from
    """
    try:
        fold_run(sweep_q, worker_q)
    except Exception as e:  # noqa: BLE001
        logger.error(e)
        wandb.join()
        sweep_q.put(WorkerDoneData(sweep_score=-0.1))


def fold_run(sweep_q: Queue, worker_q: Queue) -> None:  # type: ignore[type-arg]
    """Run a fold.

    :param sweep_q: The queue to put the result in
    :param worker_q: The queue to get the data from
    """
    # Get the data from the queue
    worker_data = worker_q.get()
    cfg = worker_data.cfg
    output_dir = worker_data.output_dir
    wandb_group_name = worker_data.wandb_group_name
    i = worker_data.i
    train_indices = worker_data.train_indices
    test_indices = worker_data.test_indices

    score = _one_fold(cfg, output_dir, i, wandb_group_name, train_indices, test_indices)

    sweep_q.put(WorkerDoneData(sweep_score=score))


def _one_fold(cfg: DictConfig, output_dir: Path, fold: int, wandb_group_name: str, train_indices: list[int], test_indices: list[int]) -> float:
    """Run one fold of cv.

    :param cfg: The configuration
    :param output_dir: The output directory
    :param fold: The fold number
    :param wandb_group_name: The wandb group name
    :param train_indices: The train indices
    :param test_indices: The test indices
    :param X: The X data
    :param y: The y data

    :return: The score
    """
    # https://github.com/wandb/wandb/issues/5119
    # This is a workaround for the issue where sweeps override the run id annoyingly
    reset_wandb_env()

    # Print section separator
    print_section_separator(f"CV - Fold {fold}")
    logger.info(f"Train/Test size: {len(train_indices)}/{len(test_indices)}")

    if cfg.wandb.enabled:
        wandb_fold_run = setup_wandb(cfg, "cv", output_dir, name=f"{wandb_group_name}_{fold}", group=wandb_group_name)

    logger.info("Creating clean pipeline for this fold")
    model_pipeline = setup_pipeline(cfg, is_train=True)

    cache_args = {
        "output_data_type": "numpy_array",
        "storage_type": ".pkl",
        "storage_path": "data/processed",
    }

    x_cache_exists = model_pipeline.x_sys._cache_exists(model_pipeline.x_sys.get_hash(), cache_args) # noqa: SLF001
    y_cache_exists = model_pipeline.y_sys._cache_exists(model_pipeline.y_sys.get_hash(), cache_args) # noqa: SLF001

    if x_cache_exists and not y_cache_exists:
        # Only read y data
        logger.info("x_sys has an existing cache, only loading in labels")
        X = None
        y = setup_label_data(cfg.raw_path)
    else:
        X, y = setup_data(raw_path=cfg.raw_path)
    if y is None:
        raise ValueError("No labels loaded to train with")

    train_args = {
        "x_sys": {
            "cache_args": cache_args,
        },
        "train_sys": {
            "MainTrainer": {
                "train_indices": train_indices,
                "test_indices": test_indices,
                "save_model": False,
            },
        },
    }

    # Fit the pipeline and get predictions
    predictions, _ = model_pipeline.train(X, y, **train_args)
    scorer = instantiate(cfg.scorer)
    score = scorer(y[test_indices], predictions[test_indices])
    logger.info(f"Score: {score}")
    if wandb_fold_run is not None:
        wandb_fold_run.log({"Score": score})

    logger.info("Finishing wandb run")
    wandb.join()

    # Memory reduction
    del X
    del y
    del predictions

    return score


if __name__ == "__main__":
    run_cv()
