from torch.utils.data import Dataset
import torch
import numpy as np


class EEGDataset(Dataset):
    def __init__(self, all_eegs, metadata, y):
        self.all_eegs = all_eegs
        self.metadata = metadata
        self.column_names = None
        self.label_names = None
        self.labels = y

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        # get the eeg id from the idx in the metadata
        eeg_id = self.metadata.iloc[idx]['eeg_id']
        eeg_label_offset_seconds = int(self.metadata.iloc[idx]['eeg_label_offset_seconds'])
        eeg = self.all_eegs[eeg_id]
        eeg = eeg.iloc[eeg_label_offset_seconds*200:eeg_label_offset_seconds*200 + 50 * 200, :]
        self.column_names = eeg.columns
        # set nans in eegto 0 if there are any
        eeg = eeg.fillna(0)
        self.label_names = self.metadata.columns[-6:]
        labels = self.metadata.iloc[idx, -6:]
        labels /= sum(labels)
        eeg_arr = eeg.to_numpy(dtype=np.float32)[:,:-1]
        labels_arr = labels.to_numpy(dtype=np.float32)
        return torch.from_numpy(eeg_arr[::10]), torch.from_numpy(labels_arr)
    
if __name__ == "__main__":
    print("Hello from basic_eeg_dataset.py!")