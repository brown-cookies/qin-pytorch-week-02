import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

from src.model import ReadmittanceModel


def evaluate_model(*, model: ReadmittanceModel):
    ...
