"""Module with plotting functions for EEGS. Can be used to plot raw EEGs and bipolar EEGs. Torch or dataframe.

Can be imported during debugging sessions. Might later be integrated into the dashboard."""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
import numpy.typing as npt

CHAIN_ORDER = ["LT", "RT", "LP", "RP", "C"]
CHAINS = {
    "LT": ("Fp1", "F7", "T3", "T5", "O1"),
    "RT": ("Fp2", "F8", "T4", "T6", "O2"),
    "LP": ("Fp1", "F3", "C3", "P3", "O1"),
    "RP": ("Fp2", "F4", "C4", "P4", "O2"),
    "C": ("Fz", "Cz", "Pz"),
}

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


def to_bipolar(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the raw EEG signal to bipolar signal.

    :param df: The raw EEG signal
    :return: The bipolar EEG signal
    """
    df_ = pd.DataFrame()
    for key, (elektrode1, elektrode2) in BIPOLAR_MAP_FULL.items():
        df_[key] = df[elektrode1] - df[elektrode2]
    return df_


def plot_torch_eeg(eeg: torch.Tensor, layout: str, title: str = "EEG Signal") -> None:
    """Plot the EEG, given a layout."""

    if layout == "raw":
        columns = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz',
                   'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'EKG']
    elif layout == "bipolar":
        columns = ['LT1', 'LT2', 'LT3', 'LT4', 'RT1', 'RT2', 'RT3', 'RT4', 'LP1', 'LP2', 'LP3',
                   'LP4', 'RP1', 'RP2', 'RP3', 'RP4', 'C1', 'C2', 'EKG']
        if eeg.shape[1] == 19:
            columns = columns[:-1]
    elif layout == "bipolar_half":
        columns = ['LT1', 'LT2', 'RT1', 'RT2', 'LP1', 'LP2', 'RP1', 'RP2', 'C1']
    else:
        raise ValueError(f"Layout {layout} not supported")
    df = pd.DataFrame(eeg.numpy(), columns=columns)
    plot_eeg(df, title)


def plot_eeg(df: pd.DataFrame, title: str = "EEG Signal", ) -> None:
    """Plot the EEG signal. Chooses the layout based on the elekrode names.

    :param df: The EEG signal to plot
    :param title: The title of the plot
    """

    # normalize the elektrode values to 0 mean
    df_ = df - df.mean()
    if "Fp1" in df.columns:
        plot_raw_eeg(df_, title)
    elif "LT1" in df.columns:
        plot_bipolar_eeg(df_, title)

    # put the time on the x-axis, using sampling rate of 200 Hz, put 10 ticks on the x-axis
    plt.xlabel("Time (s)")
    xticks = np.linspace(0, df.shape[0], 10)
    xlabels = np.linspace(0, df.shape[0] / 200, 10)
    plt.xticks(xticks, xlabels.round(2))

    plt.yticks([])
    plt.box(False)
    plt.show()


def plot_bipolar_eeg(df: pd.DataFrame, title: str = "EEG Signal") -> None:
    """Plot an EEG with all bipolar elektrodes. Plots each graph above the other with an offset of y_offset.

    :param df: The EEG signal to plot
    :param title: The title of the plot
    """
    y_offset = 0.5 * df.max().max()

    # create a new dataframe with the offset
    df_ = pd.DataFrame()

    # for each chain, plot the elektrodes in that chain in order, add some offset padding between chains
    # annotate the chain name and the elektro names next to the graphs
    total_offset = 0
    for chain_name in CHAIN_ORDER:
        i = 0
        while chain_name + str(i + 1) in df.columns:
            df_[f"{chain_name}_{i}"] = df[chain_name + str(i + 1)] + total_offset
            total_offset -= y_offset
            i += 1
        total_offset -= 2 * y_offset
    df_.plot(title=title, figsize=(20, 10), legend=False, color="black")

    for chain_name in CHAIN_ORDER:
        i = 0
        while chain_name + str(i + 1) in df.columns:
            plt.annotate(chain_name + str(i + 1), (-400, df_[f"{chain_name}_{i}"].mean()), color="black", fontsize=15)
            i += 1


def plot_raw_eeg(df: pd.DataFrame, title: str = "EEG Signal") -> None:
    """Plot an EEG with all elektrodes. Plots each graph above the other with an offset of y_offset.

    :param df: The EEG signal to plot
    :param title: The title of the plot
    """
    y_offset = 0.5 * df.max().max()

    # create a new dataframe with the offset
    df_ = pd.DataFrame()

    # for each chain, plot the elektrodes in that chain in order, add some offset padding between chains
    # annotate the chain name and the elektro names next to the graphs
    total_offset = 0
    for chain_name in CHAIN_ORDER:
        chain = CHAINS[chain_name]
        for i, elektrode in enumerate(chain):
            df_[f"{chain_name}_{elektrode}"] = df[elektrode] + total_offset
            total_offset -= y_offset
        total_offset -= 2 * y_offset
    df_.plot(title=title, figsize=(20, 10), legend=False, color="black")
    for chain_name in CHAIN_ORDER:
        chain = CHAINS[chain_name]
        plt.annotate(chain_name, (-400, df_[f"{chain_name}_{chain[0]}"].mean()), color="black", fontsize=15)
        for elektrode in chain:
            plt.annotate(elektrode, (-200, df_[f"{chain_name}_{elektrode}"].mean()), color="black", fontsize=15)


def format_y(y: npt.NDArray[np.float32]) -> str:
    """Format the y value for the plot, can be pred or true label.

    :param y: The y value to format, in shape (6,)
    :return: The formatted y value
    """
    labels = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
    label_text = ""
    for i, v in enumerate(y):
        if v > 0:
            if label_text != "":
                label_text += ','
            label_text += f'{labels[i]}:{v:.2f}'
    return label_text


if __name__ == "__main__":
    import glob

    eegs = glob.glob('../data/raw/train_eegs/*.parquet')
    # read the first 3 eegs
    for eeg in eegs[:3]:
        df = pd.read_parquet(eeg)
        plot_eeg(df)
        df_bipolar = to_bipolar(df)
        plot_eeg(df_bipolar, "Bipolar EEG Signal")
