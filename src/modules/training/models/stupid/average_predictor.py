from epochalyst.pipeline.model.training.training_block import TrainingBlock
from typing import Any

import numpy as np


class AveragePredictor(TrainingBlock):

    def __init__(self) -> None:
        super().__init__()
        self.avg = None

    def custom_train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]:
        # get 1 sample per eeg
        indices = x.meta.groupby("eeg_id").head(1).index
        x_dummy = np.zeros(y.shape[0])
        y = y[indices]
        self.avg = y.mean(axis=0)
        self.avg /= self.avg.sum()
        return self.custom_predict(x_dummy), y

    def custom_predict(self, x: Any, **pred_args: Any) -> Any:
        return np.tile(self.avg, (x.shape[0], 1))