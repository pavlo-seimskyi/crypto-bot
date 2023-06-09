import numpy as np
import pandas as pd
import pytest
from talipp.indicators import EMA as EMA_talipp

from src.features.feature_generators.technical_indicators import EMA


@pytest.fixture
def sample_data():
    return pd.DataFrame({"price": [1.0, 3.0, 2.0, 3.0, 5.0]})


@pytest.fixture
def ema_instance():
    return EMA(input_col="price", period=3)


@pytest.mark.unit
def test_initialize(sample_data, ema_instance):
    ema_instance.initialize(sample_data)
    assert isinstance(ema_instance.talipp_instance, EMA_talipp)


@pytest.mark.unit
def test_output_values(sample_data, ema_instance):
    ema_instance.initialize(sample_data)

    expected_output_values = [
        np.nan,
        np.nan,
        2.0,
        2.5,
        3.75,
    ]

    assert ema_instance.output_values == expected_output_values


@pytest.mark.unit
def test_add_value(sample_data, ema_instance):
    ema_instance.initialize(sample_data)

    new_value = pd.Series({"price": 10.0})
    ema_instance.add_value(new_value)

    expected_output_values = [np.nan, np.nan, 2.0, 2.5, 3.75, 6.875]

    assert ema_instance.output_values == expected_output_values


@pytest.mark.unit
def test_name(ema_instance):
    assert ema_instance.name == "EMA__price__3"
