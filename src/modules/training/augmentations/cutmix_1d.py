from kornia.augmentation import RandomCutMixV2
import torch

class CutMix1D(torch.nn.Module):
    """CutMix augmentation for 1D signals."""
    
    def __init__(self, p=0.5, cut_size=(0, 1)):
        """Initialize the augmentation."""
        super().__init__()
        self.cutmix = RandomCutMixV2(p = p, data_keys=["input", "class"], cut_size=cut_size)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply the augmentation to the input signal."""
        x = x.unsqueeze(-2)
        x = torch.cat((x, x), dim=-2)
        dummy_labels = torch.arange(x.size(0))
        augmented_x, augmentation_info = self.cutmix(x, dummy_labels)
        # Take only the first height dimension, ie. the original  sequence
        augmented_x = augmented_x[:,:,0,:]
        augmentation_info = augmentation_info[0]
        # multiply the last column of augment info by 2
        augmentation_info[:, -1] *= 2
        y = y.float()
        for i in range(augmentation_info.shape[0]):
            y[i] = y[i] * (1 - augmentation_info[i, -1]) + y[int(augmentation_info[i, 1])] * augmentation_info[i, -1]
        return augmented_x, y
