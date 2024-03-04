"""A verbose transformation block that logs to the terminal and to W&B."""
from abc import abstractmethod
from typing import Any

from epochalyst.pipeline.model.transformation.transformation_block import TransformationBlock

from src.modules.logging.logger import Logger


class VerboseTransformationBlock(TransformationBlock, Logger):
    """A verbose transformation block that logs to the terminal and to W&B.

    To use this block, inherit and implement the following methods:
    - custom_transform(x: Any, **kwargs: Any) -> Any
    """

    @abstractmethod
    def custom_transform(self, data: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Apply a custom transformation to the data.

        :param data: The data to transform
        :param kwargs: Any additional arguments
        :return: The transformed data
        """
        return super().custom_transform(data, **kwargs)
