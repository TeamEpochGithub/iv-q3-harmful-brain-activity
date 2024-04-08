"""Base ensemble module, this can copied and changed for other ensembling methods."""
from epochalyst.pipeline.ensemble import EnsemblePipeline

from src.modules.logging.logger import Logger
from epochalyst.pipeline.model.training.training import TrainingPipeline


class BaseEnsemble(EnsemblePipeline):
    """Base ensemble block."""

    # def concat(self, original_data, data_to_concat, weight):
    #     if original_data is None:
    #         if data_to_concat is None:
    #             return None
    #         return data_to_concat * weight
    #
    #     return original_data + data_to_concat * weight


class PostEnsemble(TrainingPipeline, Logger):
    """Ensembling with post processing blocks"""
