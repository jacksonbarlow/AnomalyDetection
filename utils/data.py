# utils/data.py

### Library Imports and Setup ###
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import CONFIG

### Main Function ###
def create_dataloader(sequences, targets=None, batch_size=None, shuffle=True):
    batch_size = batch_size or CONFIG['BATCH_SIZE']
    sequences_tensor = torch.tensor(sequences, dtype=torch.float32)

    if targets is not None:
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        dataset = TensorDataset(sequences_tensor, targets_tensor)
    else:
        dummy = torch.zeros(len(sequences_tensor))
        dataset = TensorDataset(sequences_tensor, dummy)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=0)
    return dataloader