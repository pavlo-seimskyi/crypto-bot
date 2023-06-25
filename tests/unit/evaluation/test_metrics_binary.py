import pytest
import torch

from src.evaluation.metrics.binary import calculate_metrics


@pytest.mark.unit
def test_balanced_always_negative():
    y_pred = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # always negative
    y_true = torch.Tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])  # 60% negative
    metrics = calculate_metrics(y_true, y_pred)
    # Good example for traditional metrics
    # f1: 0.0, precision: 0.0, recall: 0.0
    expected = {
        "accuracy": 0.6,
        "balanced_accuracy": 0.0,
        "matthews_corrcoef": 0.0,
        "cohen_kappa": 0.0,
    }
    assert metrics == expected


@pytest.mark.unit
def test_balanced_always_positive():
    y_pred = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # always positive
    y_true = torch.Tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # 60% positive
    metrics = calculate_metrics(y_true, y_pred)
    # Bad example for traditional metrics
    # f1: 0.75, precision: 0.6, recall: 1.0
    expected = {
        "accuracy": 0.6,
        "balanced_accuracy": 0.0,
        "matthews_corrcoef": 0.0,
        "cohen_kappa": 0.0,
    }
    assert metrics == expected


@pytest.mark.unit
def test_imbalanced_always_positive():
    y_pred = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # always negative
    y_true = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # 90% negative
    metrics = calculate_metrics(y_true, y_pred)
    # Good example for traditional metrics
    # f1: 0.0, precision: 0.0, recall: 0.0
    expected = {
        "accuracy": 0.9,
        "balanced_accuracy": 0.0,
        "matthews_corrcoef": 0.0,
        "cohen_kappa": 0.0,
    }
    assert metrics == expected


@pytest.mark.unit
def test_imbalanced_always_negative():
    y_pred = torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # always positive
    y_true = torch.Tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # 90% positive
    metrics = calculate_metrics(y_true, y_pred)
    # Bad example for traditional metrics
    # f1: 0.95, precision: 0.9, recall: 1.0
    expected = {
        "accuracy": 0.9,
        "balanced_accuracy": 0.0,
        "matthews_corrcoef": 0.0,
        "cohen_kappa": 0.0,
    }
    assert metrics == expected
