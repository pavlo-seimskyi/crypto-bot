import pytest
import torch

from src.evaluation import Backtester


@pytest.fixture
def sample_backtester():
    torch.manual_seed(0)
    x = torch.randn(1000, 1)
    y = torch.randn(1000, 1)
    return Backtester(
        x=x,
        y=y,
        wrapper=None,
        evaluation_fn=None,
        gap_proportion=0.2,
        valid_proportion=0.3,
        n_splits=1,
        n_epochs=10,
    )


@pytest.mark.unit
def test_backtester_properties(sample_backtester):
    assert sample_backtester.train_length == 500
    assert sample_backtester.gap_length == 200
    assert sample_backtester.valid_length == 300
    assert sample_backtester.split_length == 1000


@pytest.mark.unit
def test_backtester_properties_multiple_splits(sample_backtester):
    sample_backtester.n_splits = 2
    assert sample_backtester.train_length == 384
    assert sample_backtester.gap_length == 153
    assert sample_backtester.valid_length == 230
    assert sample_backtester.split_length == 769
    assert sample_backtester.step_size == 230


@pytest.mark.unit
def test_backtester_invalid_proportions(sample_backtester):
    # Gap and validation proportions sum above 1
    sample_backtester.valid_proportion = 0.9
    sample_backtester.gap_proportion = 0.2
    with pytest.raises(AssertionError):
        sample_backtester.validate()


@pytest.mark.unit
def test_backtester_invalid_n_splits(sample_backtester):
    sample_backtester.n_splits = 0
    with pytest.raises(AssertionError):
        sample_backtester.validate()


@pytest.mark.unit
def test_backtester_invalid_n_epochs(sample_backtester):
    sample_backtester.n_epochs = 0
    with pytest.raises(AssertionError):
        sample_backtester.validate()
