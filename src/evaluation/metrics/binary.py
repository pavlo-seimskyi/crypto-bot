from typing import Dict

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
)
from torch import Tensor


def evaluate_binary(y_true: Tensor, y_pred: Tensor) -> None:
    """
    Evaluate model performance by computing the metrics
    and plotting the confusion matrix.

    Parameters
    ----------
    y_true : Tensor
        True target variable.
    y_pred : Tensor
        Integer predictions for the positive class.
    """
    y_true, y_pred = y_true.cpu(), y_pred.cpu()
    metrics = calculate_metrics(y_true, y_pred)
    print(" | ".join([f"{k}: {round(v, 3)}" for k, v in metrics.items()]))
    plot_confusion_matrix(y_true, y_pred)
    return metrics


def calculate_metrics(y_true: Tensor, y_pred: Tensor) -> Dict[str, float]:
    """Calculate all metrics."""
    y_true, y_pred = y_true.detach(), y_pred.detach()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred, adjusted=True
        ),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }


def plot_confusion_matrix(y_true: Tensor, y_pred: Tensor) -> None:
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
    plt.grid(False)
    plt.title("Confusion matrix")
    plt.show()
