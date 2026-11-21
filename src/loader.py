import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ReadmittanceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.to_numpy(dtype=np.float32))
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loader(X, y, batch_size=100, shuffle=False):
    ds = ReadmittanceDataset(X, y)

    ds_loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
    )

    return ds_loader
