"""Transform the dataset to the format that Chris used."""

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ChrisSmallGetItem:
    """Transform the dataset similiar to a format that Chris used.

    Instead of returning 3x512x512, it returns 4x256x256 or 2x256x256.

    :param use_kaggle_spec: Whether to use the Kaggle spectrogram data.
    :param use_eeg_spec: Whether to use the EEG spectrogram data.
    :param eeg_spec_augmentations: The augmentations to apply to the EEG spectrogram data.
    :param kaggle_spec_augmentations: The augmentations to apply to the Kaggle spectrogram data.
    """

    use_kaggle_spec: bool = False
    use_eeg_spec: bool = False

    eeg_spec_augmentations: list[Any] = field(default_factory=list)
    kaggle_spec_augmentations: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Check if the parameters are set correctly."""
        if not self.use_kaggle_spec and not self.use_eeg_spec:
            raise ValueError("At least one of use_kaggle_spec or use_eeg_spec must be True.")

    def __call__(
        self,
        _X_eeg: torch.Tensor,
        X_kaggle_spec: torch.Tensor,
        X_eeg_spec: torch.Tensor,
        Y: torch.Tensor,
        *,
        use_augmentations: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform the dataset to the format that Chris used.

        :param dataset: The dataset to transform.
        :return: The transformed dataset.
        """
        # Kaggle Spec: 4x128x300 -> 2x256x256
        kaggle_spec = None
        if self.use_kaggle_spec:
            if use_augmentations:
                for augmentation in self.kaggle_spec_augmentations:
                    X_kaggle_spec = augmentation(X_kaggle_spec)
            kaggle_spec = X_kaggle_spec[:, :, 22:-22]
            kaggle_spec = kaggle_spec.reshape(-1, (kaggle_spec.shape[0] * kaggle_spec.shape[1]) // 2, kaggle_spec.shape[2])

        # EEG Spec: 4x128x256 -> 2x256x256
        eeg_spec = None
        if self.use_eeg_spec:
            if use_augmentations:
                for augmentation in self.eeg_spec_augmentations:
                    X_eeg_spec = augmentation(X_eeg_spec)
            eeg_spec = X_eeg_spec.reshape(-1, (X_eeg_spec.shape[0] * X_eeg_spec.shape[1]) // 2, X_eeg_spec.shape[2])

        # Append the two spectrograms: 2x 2x256x256 -> 4x256x256
        if kaggle_spec is not None and eeg_spec is not None:
            X_result = torch.cat([kaggle_spec, eeg_spec], dim=0)
        # Otherwise just use two dimensions: 2x256x256 -> 2x256x256
        elif kaggle_spec is not None:
            X_result = kaggle_spec
        elif eeg_spec is not None:
            X_result = eeg_spec
        else:
            raise ValueError("Both kaggle_spec and eeg_spec are None.")

        return X_result, Y

    @property
    def use_eeg(self) -> bool:
        """Return whether to use the EEG data."""
        return False
