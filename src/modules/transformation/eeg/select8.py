"""Select only 8 of the EEG channels, drop the rest and EKG.

These are the electrodes that are used in the 'magic forumula' (the bipolar double banana montage).
It does not do the subtraction of the channels, only the selection.
"""
from typing import Any

from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.typing import XData

ELECTRODES = ["Fp1", "T3", "C3", "O1", "Fp2", "C4", "T4", "O2"]


class Select8(VerboseTransformationBlock):
    """Select only 8 of the EEG channels, drop the rest and EKG.

    These are the elektrodes that are used in the 'magic forumula' (the bipolar double banana montage).
    It does not do the subtraction of the channels, only the selection.
    """

    def custom_transform(self, data: XData, **transform_args: Any) -> XData:
        """Apply the transformation on the entire dataset.

        :param data: The data to transform.
        :return: The transformed data.
        """
        eeg = data.eeg
        if eeg is None:
            raise ValueError("No EEG data to transform")
        for key in tqdm(eeg, desc="Selecting 8 EEG channels"):
            eeg[key] = eeg[key][ELECTRODES]
        return data
