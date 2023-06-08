import pytest
import pandas as pd
import numpy as np
from src.features.feature_generators.technical_indicators import SMA
from talipp.indicators import SMA as SMA_talipp


@pytest.fixture
def sample_data():
    return pd.DataFrame({"price": [1.0, 3.0, 2.0, 3.0, 5.0]})


@pytest.fixture
def sma_instance():
    return SMA(input_col="price", period=3)


def test_initialize(sample_data, sma_instance):
    sma_instance.initialize(sample_data)
    assert isinstance(sma_instance.talipp_instance, SMA_talipp)


def test_output_values(sample_data, sma_instance):
    sma_instance.initialize(sample_data)

    expected_output_values = [
        np.nan,
        np.nan,
        2.0,
        2.6666666666666665,
        3.333333333333333,
    ]

    assert sma_instance.output_values == expected_output_values


def test_add_value(sample_data, sma_instance):
    sma_instance.initialize(sample_data)

    new_value = pd.Series({"price": 10.0})
    sma_instance.add_value(new_value)

    expected_output_values = [
        np.nan,
        np.nan,
        2.0,
        2.6666666666666665,
        3.333333333333333,
        6.0,
    ]

    assert sma_instance.output_values == expected_output_values


def test_name(sma_instance):
    assert sma_instance.name == "SMA__price__3"
