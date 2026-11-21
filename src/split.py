import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import torch

from src.config import Config
from src.data.pipeline import prepare_data
from src.data.features import PATIENT_ID, TARGET

NUMERIC_COLS = [
    "age",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]


def split_indices(*, groups, val_split: float, test_split: float, seed: int):
    """Patient-level split: a patient never appears in two sets."""

    indices = np.arange(len(groups))

    outer = GroupShuffleSplit(
        n_splits=1,
        test_size=test_split,
        random_state=seed
    )
    train_val_index, test_index = next(outer.split(indices, groups=groups))

    val_fraction = val_split / (1 - test_split)

    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=val_fraction,
        random_state=seed
    )
    tr, va = next(inner.split(train_val_index, groups=groups[train_val_index]))

    return {
        "train": train_val_index[tr],
        "validation": train_val_index[va],
        "test": test_index,
    }


def split_train_val_test(*, df: pd.Dataframe, cfg: Config):
    """
    Build the hospital dataset, split it by patient, standardize the
    numeric features, and convert everything into PyTorch tensors.
    """
    val_split = cfg.val_size_split
    test_split = cfg.test_size_split
    seed = cfg.seed

    groups = df[PATIENT_ID].to_numpy()
    y = df[TARGET].to_numpy()
    X_df = df.drop(columns=[PATIENT_ID, TARGET])

    split = split_indices(
        groups=groups,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
    )

    X_train = X_df.iloc[split["train"]].copy()
    X_val = X_df.iloc[split["validation"]].copy()
    X_test = X_df.iloc[split["test"]].copy()

    y_train = y[split["train"]]
    y_val = y[split["validation"]]
    y_test = y[split["test"]]

    scaler = StandardScaler()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_val[NUMERIC_COLS] = scaler.transform(X_val[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "class_names": ["no_readmit", "readmit"],
        "feature_names": list(X_df.columns),
        "scaler": scaler,
    }
