"""Module for example training block."""
from epochalyst.pipeline.model.training.torch_trainer import TorchTrainer

from src.modules.logging.logger import Logger


class MainTrainer(TorchTrainer, Logger):
    """An example training block."""
