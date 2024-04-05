"""Circumferential module for different montages."""
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
    # Parasitagittal chain
    "LP1": ("Fp1", "C3"),
    "LP2": ("C3", "O1"),
    "RP1": ("Fp2", "C4"),
    "RP2": ("C4", "O2"),
}

CZ_REFERENCE = {
    # Left Temporal
    "Cz_Fp1": ("Cz", "Fp1"),
    "Cz_F7": ("Cz", "F7"),
    "Cz_T3": ("Cz", "T3"),
    "Cz_T5": ("Cz", "T5"),
    "Cz_O1": ("Cz", "O1"),
    # Right Temporal
    "Cz_Fp2": ("Cz", "Fp2"),
    "Cz_F8": ("Cz", "F8"),
    "Cz_T4": ("Cz", "T4"),
    "Cz_T6": ("Cz", "T6"),
    "Cz_O2": ("Cz", "O2"),
    # Left Parasagittal
    "Cz_F3": ("Cz", "F3"),
    "Cz_C3": ("Cz", "C3"),
    "Cz_P3": ("Cz", "P3"),
    # Right Parasagittal
    "Cz_F4": ("Cz", "F4"),
    "Cz_C4": ("Cz", "C4"),
    "Cz_P4": ("Cz", "P4"),
    # Central
    "Cz_Fz": ("Cz", "Fz"),
    "Cz_Pz": ("Cz", "Pz"),
}

CZ_REFERENCE_REDUCED = {  # 12 channels
    # Left Temporal
    "Cz_Fp1": ("Cz", "Fp1"),
    "Cz_T3": ("Cz", "T3"),
    "Cz_O1": ("Cz", "O1"),
    # Right Temporal
    "Cz_Fp2": ("Cz", "Fp2"),
    "Cz_T4": ("Cz", "T4"),
    "Cz_O2": ("Cz", "O2"),
    # Left Parasagittal
    "Cz_F3": ("Cz", "F3"),
    "Cz_P3": ("Cz", "P3"),
    # Right Parasagittal
    "Cz_F4": ("Cz", "F4"),
    "Cz_P4": ("Cz", "P4"),
    # Central
    "Cz_Fz": ("Cz", "Fz"),
    "Cz_Pz": ("Cz", "Pz"),
}


@dataclass
class Circumferential(VerboseTransformationBlock):
    """Convert the EEGs to circumferential configuration.

    Only EKG (optional) and the circumferential map are left after applying this block.

    :param use_full_map: Whether to use the full bipolar map or the half map
    :param keep_ekg: Whether to keep the EKG channel
    """

    str_map: str = "CIRCUMFERENTIAL"
    keep_ekg: bool = False

    def __post_init__(self) -> None:
        """Set the bipolar map to use."""
        super().__post_init__()

        self.map = CIRCUMFERENTIAL
        match self.str_map:
            case "CIRCUMFERENTIAL":
                self.map = CIRCUMFERENTIAL
            case "CIRCUMFERENTIAL_PARASAGITTAL":
                self.map = CIRCUMFERENTIAL_PARASAGITTAL
            case "CZ_REFERENCE":
                self.map = CZ_REFERENCE
            case "CZ_REFERENCE_REDUCED":
                self.map = CZ_REFERENCE_REDUCED

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
        for chain, (e1, e2) in self.map.items():
            eeg[chain] = eeg[e1] - eeg[e2]

        # select only the bipolar channels (and EKG if keep_ekg is True)
        selection = list(self.map.keys())
        if self.keep_ekg:
            selection.append("EKG")
        return eeg[selection]
