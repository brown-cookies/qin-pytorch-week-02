import numpy as np
import pytest

from src.data.features import PATIENT_ID
from src.split import split_indices, split_train_val_test


def test_data_not_empty(df):
    assert df.shape[0] > 0
    assert df.shape[1] > 1


def test_data_shape(df):
    assert df.shape == (101766, 276)


def test_data_split_deterministic(df, cfg):
    first = split_indices(
        groups=df[PATIENT_ID].to_numpy(),
        val_split=cfg.val_size_split,
        test_split=cfg.test_size_split,
        seed=cfg.seed
    )

    second = split_indices(
        groups=df[PATIENT_ID].to_numpy(),
        val_split=cfg.val_size_split,
        test_split=cfg.test_size_split,
        seed=cfg.seed
    )

    assert (first["train"] == second["train"]).all()
    assert (first["validation"] == second["validation"]).all()
    assert (first["test"] == second["test"]).all()


def test_split_covers_all_rows(df, cfg):
    d = split_train_val_test(df=df, cfg=cfg)

    train_idx = set(d["X_train"].index)
    val_idx = set(d["X_val"].index)
    test_idx = set(d["X_test"].index)

    total = len(d["X_train"]) + len(d["X_val"]) + len(d["X_test"])
    assert total == len(df)

    assert train_idx | val_idx | test_idx == set(df.index)


def test_no_train_test_overlap(df, cfg):
    d = split_train_val_test(df=df, cfg=cfg)
    X_train = d["X_train"]
    X_test = d["X_test"]
    y_train = d["y_train"]
    y_test = d["y_test"]

    train_idx = set(X_train.index)
    test_idx = set(X_test.index)

    assert train_idx.isdisjoint(test_idx)

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
