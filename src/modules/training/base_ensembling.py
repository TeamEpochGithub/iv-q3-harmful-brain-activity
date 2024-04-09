"""Base ensemble module, this can copied and changed for other ensembling methods."""
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from epochalyst.pipeline.ensemble import EnsemblePipeline
from epochalyst.pipeline.model.training.training import TrainingPipeline

from src.modules.logging.logger import Logger


@dataclass
class BaseEnsemble(EnsemblePipeline):
    """Base ensemble block.

    :param count: Count of step
    :param weights: Weights of each step
    """

    count: int = 0
    weights: list[float] = field(default=[])

    def __post_init__(self) -> None:
        """Post initialization function of BaseEnsemble."""
        super().__post_init__()
        self.weights = [w / sum(self.weights) for w in self.weights]

    def concat(self, original_data: npt.NDArray[np.float64], data_to_concat: npt.NDArray[np.float64], weight: float) -> npt.NDArray[np.float64]:
        """Concat X data by weight to original_data.

        :param original_data: Existing data
        :param data_to_concat: New data to add
        :param weight: Weight of new data
        :return: Concated data
        """
        if original_data is None:
            if data_to_concat is None:
                return None
            data = data_to_concat * self.weights[self.count]
            self.count += 1
            return data

        weight = weight + 1
        self.count += 1
        return original_data + data_to_concat * self.weights[self.count - 1]

    def concat_labels(self, original_data: npt.NDArray[np.float64], data_to_concat: npt.NDArray[np.float64], weight: float) -> npt.NDArray[np.float64]:
        """Concat Y data by weight to original_data.

        :param original_data: Existing data
        :param data_to_concat: New data to add
        :param weight: Weight of new data
        :return: Concated data
        """
        if original_data is None:
            if data_to_concat is None:
                return None
            return data_to_concat * weight
        return original_data + data_to_concat * weight


class PostEnsemble(EnsemblePipeline, TrainingPipeline, Logger):
    """Ensembling with post processing blocks."""
