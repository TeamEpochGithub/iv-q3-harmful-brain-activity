"""CutMix augmentation for 1D signals."""
import torch


class CutMix1D(torch.nn.Module):
    """CutMix augmentation for 1D signals."""

    def __init__(self, p: float = 0.5, low: float = 0, high: float = 1) -> None:
        """Initialize the augmentation."""
        super().__init__()
        self.p = p
        self.low = low
        self.high = high

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the augmentation to the input signal."""
        indices = torch.arange(x.shape[0], device=x.device, dtype=torch.int)
        shuffled_indices = torch.randperm(indices.shape[0])

        low_len = int(self.low * x.shape[-1])
        high_len = int(self.high * x.shape[-1])
        cutoff_indices = torch.randint(low_len, high_len, (x.shape[-1],), device=x.device, dtype=torch.int)
        cutoff_rates = cutoff_indices.float() / x.shape[-1]

        augmented_x = x.clone()
        augmented_y = y.clone().float()
        for i in range(x.shape[0]):
            if torch.rand(1) < self.p:
                augmented_x[i, :, cutoff_indices[i] :] = x[shuffled_indices[i], :, cutoff_indices[i] :]
                augmented_y[i] = y[i] * cutoff_rates[i] + y[shuffled_indices[i]] * (1 - cutoff_rates[i])
        return augmented_x, augmented_y
