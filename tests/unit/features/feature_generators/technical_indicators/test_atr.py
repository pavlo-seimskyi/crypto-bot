import numpy as np
import pandas as pd
import pytest
from talipp.indicators import ATR as ATR_talipp

from src.features.feature_generators.technical_indicators import ATR


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "high": [2.0, 5.0, 1.5, 4.0, 7.0],
            "low": [0.5, 1.0, 0.5, 1.0, 1.0],
            "close": [1.0, 2.0, 1.0, 2.0, 4.0],
        }
    )


@pytest.fixture
def atr_instance():
    return ATR(high_col="high", low_col="low", close_col="close", period=2)


@pytest.mark.unit
def test_initialize(sample_data, atr_instance):
    atr_instance.initialize(sample_data)
    assert isinstance(atr_instance.talipp_instance, ATR_talipp)


@pytest.mark.unit
def test_output_values(sample_data, atr_instance):
    atr_instance.initialize(sample_data)

    expected_output_values = [np.nan, 2.75, 2.125, 2.5625, 4.28125]

    assert atr_instance.output_values == expected_output_values


@pytest.mark.unit
def test_add_value(sample_data, atr_instance):
    atr_instance.initialize(sample_data)

    new_value = pd.Series({"high": 10.0, "low": 5.0, "close": 9.0})
    atr_instance.add_value(new_value)

    expected_output_values = [np.nan, 2.75, 2.125, 2.5625, 4.28125, 5.140625]

    assert atr_instance.output_values == expected_output_values


@pytest.mark.unit
def test_name(atr_instance):
    assert atr_instance.name == "ATR__close__2"
