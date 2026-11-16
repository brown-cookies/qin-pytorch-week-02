import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch


def get_datasets(*, cfg):
    ...


def split_indices(*, y, val_split: float, test_split: float, seed: int):
    ...


def load_datasets(*, cfg):
    """
    Load the diabetes dataset, split it, standardize the features,
    and convert everything into PyTorch tensors.
    """
    data = get_datasets()

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "class_names": class_names,
        "scaler": scaler
    }
