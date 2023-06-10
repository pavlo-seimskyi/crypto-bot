import pytest
import torch

from src.model.preprocessors import StandardScaler


def almost_equal(x1, x2, atol=1e-6):
    """Check if two tensors are almost equal."""
    return torch.isclose(x1, x2, atol=atol).all()


@pytest.mark.unit
def test_fit():
    scaler = StandardScaler(dim=0)
    tensor = torch.randn((50, 10))
    scaler.fit(tensor)

    assert scaler.mean is not None
    assert scaler.std is not None


@pytest.mark.unit
def test_transform_unfitted():
    tensor = torch.randn((10, 10))
    scaler = StandardScaler(dim=0)
    # Raise exception when `transform` is called before `fit`
    with pytest.raises(AssertionError):
        scaler.transform(tensor)


@pytest.mark.unit
def test_transform_dim_0():
    # Will scale across features
    scaler = StandardScaler(dim=0)
    torch.manual_seed(42)
    tensor = torch.randn((50, 10))

    scaler.fit(tensor)

    scaled_tensor = scaler.transform(tensor)

    # Mean should be close to 0 and std should be close to 1
    assert almost_equal(torch.mean(scaled_tensor, dim=0), torch.zeros(10))
    assert almost_equal(torch.std(scaled_tensor, dim=0), torch.ones(10))


@pytest.mark.unit
def test_transform_dim_1():
    # Will scale across features
    scaler = StandardScaler(dim=1)
    torch.manual_seed(42)
    tensor = torch.randn((50, 10))

    scaler.fit(tensor)

    scaled_tensor = scaler.transform(tensor)

    # Mean should be close to 0 and std should be close to 1
    assert almost_equal(torch.mean(scaled_tensor, dim=1), torch.zeros(50))
    assert almost_equal(torch.std(scaled_tensor, dim=1), torch.ones(50))


@pytest.mark.unit
def test_transform_dim_2():
    dim = 2
    # Will scale across 3rd dimension
    scaler = StandardScaler(dim=dim)
    torch.manual_seed(42)
    tensor = torch.randn((50, 10, 20))

    scaler.fit(tensor)
    scaled_tensor = scaler.transform(tensor)

    # Mean should be close to 0 and std should be close to 1
    assert almost_equal(
        torch.mean(scaled_tensor, dim=dim), torch.zeros((50, 10))
    )
    assert almost_equal(
        torch.std(scaled_tensor, dim=dim), torch.ones((50, 10))
    )


@pytest.mark.unit
def test_fit_transform():
    scaler = StandardScaler(dim=0)
    torch.manual_seed(42)
    tensor = torch.randn((50, 10))

    scaled_tensor = scaler.fit_transform(tensor)

    assert scaler.mean is not None
    assert scaler.std is not None
    assert almost_equal(torch.mean(scaled_tensor, dim=0), torch.zeros(10))
    assert almost_equal(torch.std(scaled_tensor, dim=0), torch.ones(10))
