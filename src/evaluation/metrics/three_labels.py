from typing import Dict

import matplotlib.pyplot as plt
import torch.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
)
from torch import Tensor


def evaluate_three_labels(y_test: Tensor, y_pred_proba: Tensor) -> None:
    """
    Evaluate model performance by computing the multiclass metrics
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
    y_pred = y_pred_proba.argmax(dim=1).detach()
    plot_confusion_matrix(y_test, y_pred)


def calculate_metrics(
    y_test: Tensor, y_pred_proba: Tensor
) -> Dict[str, float]:
    """Calculate all metrics."""
    y_test, y_pred_proba = y_test.detach(), y_pred_proba.detach()
    y_pred = y_pred_proba.argmax(dim=1)
    return {
        "f1": f1_score(
            y_test, y_pred, labels=(0, 1, 2), zero_division=0, average="macro"
        ),
        "avg_precision_score": average_precision_score(
            F.one_hot(y_test, num_classes=3), y_pred_proba, average="macro"
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, y_pred, adjusted=True
        ),
    }


def plot_confusion_matrix(y_test, y_pred) -> None:
    """Plot a confusion matrix on axis."""
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=(0, 1, 2), cmap="Blues"
    )
    plt.grid(False)
    plt.title("Confusion matrix")
    plt.show()
