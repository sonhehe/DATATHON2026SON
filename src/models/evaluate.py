from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, roc_auc_score


@dataclass
class ClassificationResult:
    accuracy: float
    f1: float
    roc_auc: float | None


@dataclass
class RegressionResult:
    mae: float


def evaluate_classification(y_true, y_pred, y_proba=None) -> ClassificationResult:
    auc = None
    if y_proba is not None:
        try:
            auc = float(roc_auc_score(y_true, y_proba))
        except Exception:
            auc = None
    return ClassificationResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred, average="binary")),
        roc_auc=auc,
    )


def evaluate_regression(y_true, y_pred) -> RegressionResult:
    return RegressionResult(mae=float(mean_absolute_error(y_true, y_pred)))
