import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch

from src.data.pipeline import load_data


def split_indices(*, y, val_split: float, test_split: float, seed: int):
    ...


def split(*, cfg):
    """
    Load the hospital dataset, split it, standardize the features,
    and convert everything into PyTorch tensors.
    """
    df = load_data()

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
