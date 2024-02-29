from agogos.trainer import Trainer
import numpy as np 


class ExampleTrainingBlock(Trainer):
    """This is an example training block, there are no init blocks in the base Trainer class.
    Both train and predict methods should be overridden in the child class."""

    def train(self, x: np.ndarray, y: np.ndarray):
        """Train the block.

        :param x: The input data.
        :param y: The target variable."""
        print("Training the block")
        return x * 2, y
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the target variable.

        :param x: The input data.
        :return: The predictions."""
        print("Predicting the target variable")
        return x * 2