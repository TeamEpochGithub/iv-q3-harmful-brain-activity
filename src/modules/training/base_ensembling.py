"""Base ensemble module, this can copied and changed for other ensembling methods."""
from dataclasses import dataclass

from epochalyst.pipeline.ensemble import EnsemblePipeline
from epochalyst.pipeline.model.training.training import TrainingPipeline

from src.modules.logging.logger import Logger


@dataclass
class BaseEnsemble(EnsemblePipeline):
    """Base ensemble block."""

    count: int = 0
    weights: list[float] = None

    def __post_init__(self):
        super().__post_init__()
        self.weights = [w / sum(self.weights) for w in self.weights]

    def concat(self, original_data, data_to_concat, weight):
        if original_data is None:
            if data_to_concat is None:
                return None
            return data_to_concat * weight
        data = original_data + data_to_concat * self.weights[self.count]
        self.count += 1
        return data
        # return original_data + data_to_concat * self.weights[count]


class PostEnsemble(EnsemblePipeline, TrainingPipeline, Logger):
    """Ensembling with post processing blocks"""
