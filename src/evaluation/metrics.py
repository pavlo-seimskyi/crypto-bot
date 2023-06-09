from typing import Dict

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch import Tensor


def evaluate(y_test: Tensor, y_pred_proba: Tensor) -> None:
    """
    Evaluate model performance by computing the metrics
    and plotting the confusion matrix.

    Parameters
    ----------
    y_test : Tensor
        True target variable.
    y_pred_proba : Tensor
        Prediction probabilities for the positive class (from 0.0 to 1.0).
    """
    y_test = y_test.cpu()
    y_pred_proba = y_pred_proba.cpu()
    metrics = calculate_metrics(y_test, y_pred_proba)
    print(", ".join([f"{k}: {round(v, 3)}" for k, v in metrics.items()]))
    y_pred = torch.round(y_pred_proba)
    plot_confusion_matrix(y_test, y_pred)


def calculate_metrics(
    y_test: Tensor, y_pred_proba: Tensor
) -> Dict[str, float]:
    """Calculate all metrics."""
    y_test, y_pred_proba = y_test.detach(), y_pred_proba.detach()
    y_pred = torch.round(y_pred_proba)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "avg_precision_score": average_precision_score(y_test, y_pred_proba),
    }


def plot_confusion_matrix(y_test, y_pred) -> None:
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    plt.grid(False)
    plt.title("Confusion matrix")
    plt.show()
