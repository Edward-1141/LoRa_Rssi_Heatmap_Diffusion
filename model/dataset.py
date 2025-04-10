import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class LoRaHeatmapDataset(Dataset):
    """
    Dataset for simplified & refined LoRa heatmap data (no sampling, no shifting)
    """
    def __init__(self, h5_file_path):
        self.h5_file = h5py.File(h5_file_path, 'r')
        self.total_len = self.h5_file['metadata'].attrs['total_len']

    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        gt = torch.tensor(np.array(self.h5_file['ground_truth'][f'{idx}']), dtype=torch.float32)

        return gt.unsqueeze(0)

def load_data(train_file, test_file):
    train_dataset = LoRaHeatmapDataset(train_file)
    test_dataset = LoRaHeatmapDataset(test_file)
    
    return train_dataset, test_dataset
