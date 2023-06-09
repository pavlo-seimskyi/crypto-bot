import pytest
import torch

from src.model.datasets import SlidingWindowDataset


@pytest.mark.unit
def test_sliding_window_length():
    x = torch.arange(10).unsqueeze(1)
    seq_len = 3

    dataset = SlidingWindowDataset(x, seq_len=seq_len)

    assert len(dataset) == x.size(0) - seq_len + 1


@pytest.mark.unit
def test_sliding_window_content():
    x = torch.arange(10).unsqueeze(1)
    seq_len = 3

    dataset = SlidingWindowDataset(x, seq_len=seq_len)

    for i in range(len(dataset)):
        assert torch.equal(dataset[i], x[i : i + seq_len])


@pytest.mark.unit
def test_sliding_window_with_targets():
    x = torch.arange(10).unsqueeze(1)
    y = torch.arange(10).unsqueeze(1)
    seq_len = 3

    dataset = SlidingWindowDataset(x, y, seq_len=seq_len)

    for i in range(len(dataset)):
        assert torch.equal(dataset[i][0], x[i : i + seq_len])
        assert dataset[i][1] == y[i + seq_len - 1]


@pytest.mark.unit
def test_sliding_window_with_custom_y_position():
    x = torch.arange(10).unsqueeze(1)
    y = torch.arange(10).unsqueeze(1)
    seq_len = 3
    y_position = 1

    dataset = SlidingWindowDataset(
        x, y, seq_len=seq_len, y_position=y_position
    )

    for i in range(len(dataset)):
        assert torch.equal(dataset[i][0], x[i : i + seq_len])
        assert dataset[i][1] == y[i + y_position]


@pytest.mark.unit
def test_incompatible_x_y_shapes():
    x = torch.arange(10).unsqueeze(1)
    y = torch.arange(9).unsqueeze(1)
    seq_len = 3

    with pytest.raises(AssertionError):
        SlidingWindowDataset(x, y, seq_len=seq_len)


@pytest.mark.unit
def test_invalid_y_position():
    x = torch.arange(10).unsqueeze(1)
    y = torch.arange(10).unsqueeze(1)
    seq_len = 3
    y_position = 3

    with pytest.raises(AssertionError):
        SlidingWindowDataset(x, y, seq_len=seq_len, y_position=y_position)
