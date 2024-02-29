"""Example training block for the pipeline."""
import dask.array as da
from agogos.trainer import Trainer

from src.logging_utils.logger import logger


class ExampleTrainingBlock(Trainer):
    """Example training block, there are no init blocks in the base Trainer class.

    Both train and predict methods should be overridden in the child class.
    """

    def train(self, x: da.Array, y: da.Array) -> tuple[da.Array, da.Array]:
        """Train the block.

        :param x: The input data.
        :param y: The target variable.
        """
        logger.info("Training the block")
        return x * 2, y

    def predict(self, x: da.Array) -> da.Array:
        """Predict the target variable.

        :param x: The input data.
        :return: The predictions.
        """
        logger.info("Predicting the target variable")
        return x * 2
