from dataclasses import dataclass, replace
from typing import Literal

import torch


@dataclass(frozen=True, slots=True)
class Config:
    is_ci: bool = False
    device: Literal["cpu", "cuda"] = "cpu"
    seed: int = 42
    test_size_split: float = 0.15
    val_size_split: float = 0.15
    batch_size: int = 512
    epochs: int = 1000
    learning_rate: float = 0.01


def get_config(*, is_ci=False, is_smoke=False) -> Config:
    cfg = Config()

    if torch.cuda.is_available():
        cfg = replace(cfg, device="cuda")

    if is_smoke:
        cfg = replace(cfg, batch_size=128, epochs=100)

    if is_ci:
        cfg = replace(cfg, is_ci=True, device="cpu")

    return cfg
