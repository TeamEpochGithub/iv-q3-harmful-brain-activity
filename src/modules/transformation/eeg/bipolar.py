"""Convert the EEGs to bipolar configuration.

Only EKG (optional) and the Bipolar map are left after applying this block.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.modules.transformation.verbose_transformation_block import VerboseTransformationBlock
from src.typing.typing import XData

BIPOLAR_MAP_FULL = {
    # Left Temporal chain (Fp1, F7, T3, T5, O1
    "LT1": ("Fp1", "F7"),
    "LT2": ("F7", "T3"),
    "LT3": ("T3", "T5"),
    "LT4": ("T5", "O1"),
    # Right Temporal chain (Fp2, F8, T4, T6, O2)
    "RT1": ("Fp2", "F8"),
    "RT2": ("F8", "T4"),
    "RT3": ("T4", "T6"),
    "RT4": ("T6", "O2"),
    # Left Parasitagittal chain (Fp1, F3, C3, P3, O1)
    "LP1": ("Fp1", "F3"),
    "LP2": ("F3", "C3"),
    "LP3": ("C3", "P3"),
    "LP4": ("P3", "O1"),
    # Right Parasitagittal chain (Fp2, F4, C4, P4, O2)
    "RP1": ("Fp2", "F4"),
    "RP2": ("F4", "C4"),
    "RP3": ("C4", "P4"),
    "RP4": ("P4", "O2"),
    # Central chain (Fz, Cz, Pz)
    "C1": ("Fz", "Cz"),
    "C2": ("Cz", "Pz"),
}

# this version skips electrodes, and has only two values per chain, and one for the central chain
BIPOLAR_MAP_HALF = {
    # Left Temporal chain (Fp1, F7, T3, T5, O1)
    "LT1": ("Fp1", "T3"),
    "LT2": ("T3", "O1"),
    # Right Temporal chain (Fp2, F8, T4, T6, O2)
    "RT1": ("Fp2", "T4"),
    "RT2": ("T4", "O2"),
    # Left Parasitagittal chain (Fp1, F3, C3, P3, O1)
    "LP1": ("Fp1", "C3"),
    "LP2": ("C3", "O1"),
    # Right Parasitagittal chain (Fp2, F4, C4, P4, O2)
    "RP1": ("Fp2", "C4"),
    "RP2": ("C4", "O2"),
    # Central chain (Fz, Cz, Pz)
    "C1": ("Fz", "Pz"),
}


@dataclass
class BipolarEEG(VerboseTransformationBlock):
    """Convert the EEGs to bipolar configuration.

    Only EKG (optional) and the Bipolar map are left after applying this block.

    :param use_full_map: Whether to use the full bipolar map or the half map
    :param keep_ekg: Whether to keep the EKG channel
    """

    use_full_map: bool = False
    keep_ekg: bool = False

    def __post_init__(self) -> None:
        """Set the bipolar map to use."""
        super().__post_init__()
        self.bipolar_map = BIPOLAR_MAP_FULL if self.use_full_map else BIPOLAR_MAP_HALF

    def custom_transform(self, data: XData, **kwargs: Any) -> XData:
        """Apply the transformation.

        :param data: The X data to transform, as tuple (eeg, spec, meta)
        :return: The transformed data
        """
        eeg = data.eeg
        if eeg is None:
            raise ValueError("No EEG data to transform")
        for key in tqdm(eeg, desc="Converting EEG to bipolar"):
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
