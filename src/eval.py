import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.model import ReadmittanceModel


def evaluate_model(*, model, test_loader, threshold=0.5):
    model.eval()

    predictions = []
    actual = []

    with torch.no_grad():

        for X, y in test_loader:
            z = model(X)
            probs = torch.sigmoid(z)
            preds = (probs >= threshold).float()

            predictions.extend(
                preds.numpy().reshape(-1)
            )
            actual.extend(
                y.numpy().reshape(-1)
            )

        accuracy = accuracy_score(actual, predictions)
        precision = precision_score(actual, predictions, zero_division=0)
        recall = recall_score(actual, predictions, zero_division=0)
        f1 = f1_score(actual, predictions, zero_division=0)
        confusion = confusion_matrix(actual, predictions, labels=[0, 1])

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion": confusion,
        }
