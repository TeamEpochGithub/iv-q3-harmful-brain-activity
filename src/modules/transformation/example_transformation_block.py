"""Example transformation block for the transformation pipeline."""

import dask.array as da

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock


class ExampleTransformationBlock(VerboseTransformationBlock):
    """An example transformation block for the transformation pipeline."""

    def custom_transform(self, data: da.Array) -> da.Array:
        """Apply a custom transformation to the data.

        :param data: The data to transform
        :param kwargs: Any additional arguments
        :return: The transformed data
        """
        return data

