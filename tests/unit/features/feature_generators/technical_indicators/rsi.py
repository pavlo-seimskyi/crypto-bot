import pytest
import pandas as pd
import numpy as np
from src.features.feature_generators.technical_indicators import RSI
from talipp.indicators import RSI as RSI_talipp


@pytest.fixture
def sample_data():
    return pd.DataFrame({"price": [1.0, 2.0, 1.0, 2.0, 4.0]})


@pytest.fixture
def rsi_instance():
    return RSI(input_col="price", period=3)


def test_initialize(sample_data, rsi_instance):
    rsi_instance.initialize(sample_data)
    assert isinstance(rsi_instance.talipp_instance, RSI_talipp)


def test_output_values(sample_data, rsi_instance):
    rsi_instance.initialize(sample_data)

    expected_output_values = [
        np.nan,
        np.nan,
        np.nan,
        66.66666666666666,
        83.33333333333333,
    ]

    assert rsi_instance.output_values == expected_output_values


def test_add_value(sample_data, rsi_instance):
    rsi_instance.initialize(sample_data)

    new_value = pd.Series({"price": 10.0})
    rsi_instance.add_value(new_value)

    expected_output_values = [
        np.nan,
        np.nan,
        np.nan,
        66.66666666666666,
        83.33333333333333,
        94.87179487179488,
    ]

    assert rsi_instance.output_values == expected_output_values


def test_name(rsi_instance):
    assert rsi_instance.name == "RSI__price__3"
