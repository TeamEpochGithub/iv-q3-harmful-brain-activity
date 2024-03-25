import torch
from src.modules.training.datasets.get_item.chris import ChrisGetItem

from dataclasses import dataclass

@dataclass
class ChrisandEEGGetItem:

    def __post_init__(self) -> None:
        self.chris = ChrisGetItem(use_kaggle_spec=True, use_eeg_spec=True)

    def __call__(self, X_eeg: torch.Tensor, X_kaggle_spec: torch.Tensor, X_eeg_spec: torch.Tensor, Y: torch.Tensor):
        chris_result = self.chris(X_eeg, X_kaggle_spec, X_eeg_spec, Y)
        return [X_eeg.transpose(1,0), chris_result[0]], Y