"""Converts EEG data to 5D grid format based on the electrode positions."""

import numpy as np
import torch

ELECTRODES = {
    0: "FP1",
    1: "F3",
    2: "C3",
    3: "P3",
    4: "F7",
    5: "T3",
    6: "T5",
    7: "O1",
    8: "FZ",
    9: "CZ",
    10: "PZ",
    11: "FP2",
    12: "F4",
    13: "C4",
    14: "P4",
    15: "F8",
    16: "T4",
    17: "T6",
    18: "O2",
    19: "EKG",
}

ELECTRODE_POSITIONS = {
    "FP1": (0, 3),
    "FP2": (0, 5),
    "F7": (2, 0),
    "F3": (2, 2),
    "FZ": (2, 4),
    "F4": (2, 6),
    "F8": (2, 8),
    "T3": (4, 0),
    "C3": (4, 2),
    "CZ": (4, 4),
    "C4": (4, 6),
    "T4": (4, 8),
    "T5": (6, 0),
    "P3": (6, 2),
    "PZ": (6, 4),
    "P4": (6, 6),
    "T6": (6, 8),
    "O1": (8, 3),
    "O2": (8, 5),
}


def to_3d_grid(eeg_data: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Convert EEG data from 3D (N, C, L) to 5D grid format (N, C, L, W, H) based on the electrode positions.

    :param eeg_data: EEG data in 3D format (N, C, L).

    :return: EEG data in 5D grid format (N, C, L, W, H).
    """
    n, c, l = eeg_data.shape  # noqa: E741
    grid = torch.zeros(n, 1, l, width, height).to(eeg_data.device)
    for i in range(n):
        for curr_t in range(l):
            section = eeg_data[i, :, curr_t]
            for channel, value in enumerate(section):
                electrode = ELECTRODES[channel]
                if electrode == "EKG":
                    continue
                x, y = ELECTRODE_POSITIONS[electrode]
                grid[i, 0, curr_t, x, y] = value
    return grid


def to_3d_grid_vectorized(eeg_data: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Convert EEG data from 3D (N, C, L) to 5D grid format (N, C, L, W, H) based on the electrode positions, without explicit loops.

    :param eeg_data: EEG data in 3D format (N, C, L).
    :return: EEG data in 5D grid format (N, 1, L, W, H).
    """
    n, c, l = eeg_data.shape  # noqa: E741

    total_electrodes = 20
    number_channels = c / total_electrodes
    if number_channels != int(number_channels):
        raise ValueError(f"Number of channels must be a multiple of 20, got {c} instead.")

    grid = torch.zeros(n, int(number_channels), l, width, height, device=eeg_data.device)

    for j, _ in enumerate(range(int(number_channels))):
        # Generate indices for electrodes excluding "EKG"
        electrode_indices = [i for i, elec in ELECTRODES.items() if elec != "EKG"]
        electrode_positions = [ELECTRODE_POSITIONS[ELECTRODES[i]] for i in electrode_indices]

        # Convert positions to tensor for indexing
        positions = torch.tensor(electrode_positions, dtype=torch.long, device=eeg_data.device)
        x_positions, y_positions = positions[:, 0], positions[:, 1]

        # Get the corresponding EEG data values for valid electrodes
        valid_eeg_data = eeg_data[:, np.array(electrode_indices) + (j * total_electrodes), :]

        # Assign values from EEG data to the grid based on electrode positions
        for i, (x, y) in enumerate(zip(x_positions, y_positions, strict=False)):
            grid[:, j, :, x, y] = valid_eeg_data[:, i, :]

    return grid
