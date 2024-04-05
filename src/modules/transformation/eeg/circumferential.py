from dataclasses import dataclass
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.typing import XData

CIRCUMFERENTIAL = {
    "C1": ("Fp1", "F7"),
    "C2": ("F7", "T3"),
    "C3": ("T3", "T5"),
    "C4": ("T5", "O1"),
    "C5": ("O1", "O2"),
    "C6": ("O2", "T6"),
    "C7": ("T6", "T4"),
    "C8": ("T4", "F8"),
    "C9": ("F8", "Fp2"),
    "C10": ("Fp2", "Fp1"),
    "C0": ("Fz", "Pz"),
}

CIRCUMFERENTIAL_PARASAGITTAL = {

}


@dataclass
class Circumferential(VerboseTransformationBlock):
    """Convert the EEGs to circumferential configuration.

    Only EKG (optional) and the circumferential map are left after applying this block.

    :param use_full_map: Whether to use the full bipolar map or the half map
    :param keep_ekg: Whether to keep the EKG channel
    """

    # use_parasagittal: bool = False
    keep_ekg: bool = False

    def __post_init__(self) -> None:
        """Set the bipolar map to use."""
        super().__post_init__()
        self.bipolar_map = CIRCUMFERENTIAL

    def custom_transform(self, data: XData, **kwargs: Any) -> XData:
        """Apply the transformation.

        :param data: The X data to transform, as tuple (eeg, spec, meta)
        :return: The transformed data
        """
        eeg = data.eeg
        if eeg is None:
            raise ValueError("No EEG data to transform")
        for key in tqdm(eeg, desc="Converting EEG to circumferential"):
            eeg[key] = self._convert_to_bipolar(eeg[key])
        return data

    def _convert_to_bipolar(self, eeg: pd.DataFrame) -> pd.DataFrame:
        """Convert the EEG to bipolar configuration.

        :param eeg: The EEG data to convert
        :return: The converted EEG data
        """
        for chain, (e1, e2) in self.bipolar_map.items():
            eeg[chain] = eeg[e1] - eeg[e2]

        # select only the bipolar channels (and EKG if keep_ekg is True)
        selection = list(self.bipolar_map.keys())
        if self.keep_ekg:
            selection.append("EKG")
        return eeg[selection]
