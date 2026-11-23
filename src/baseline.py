from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import Config

# TODO: Add DummyClassifier for Trivial Baseline
# TODO : add HistGradientBoostingClassifier for Gradient-Boosted Baseline


def get_trivial_baseline(X, y):
    classifier = DummyClassifier(strategy="prior")

    return classifier.fit(X, y)


def get_gradient_boosted_baseline(X, y, cfg: Config):
    classifier = HistGradientBoostingClassifier(
        learning_rate=cfg.learning_rate,
        random_state=cfg.seed
    )

    return classifier.fit(X, y)


def evaluate_baseline(model, X, y, threshold=0.5):
    scores = model.predict_proba(X)[:, 1]
    predictions = (scores >= threshold).astype(int)

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)
    confusion = confusion_matrix(y, predictions, labels=[0, 1])
    pr_auc = average_precision_score(y, scores)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": float(pr_auc),
        "confusion": confusion.tolist(),
    }
