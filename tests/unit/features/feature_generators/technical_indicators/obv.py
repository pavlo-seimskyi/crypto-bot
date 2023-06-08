import pytest
import pandas as pd
import numpy as np
from src.features.feature_generators.technical_indicators import OBV
from talipp.indicators import OBV as OBV_talipp


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "open": [1.0, 3.0, 2.0, 3.0, 5.0],
            "close": [3.0, 2.0, 3.0, 5.0, 4.0],
            "high": [3.0, 3.0, 3.0, 5.0, 5.0],
            "low": [1.0, 2.0, 2.0, 3.0, 4.0],
            "volume": [10.0, 9.0, 11.0, 13.0, 14.0],
        }
    )


@pytest.fixture
def obv_instance():
    return OBV(
        open_col="open",
        close_col="close",
        high_col="high",
        low_col="low",
        volume_col="volume",
    )


def test_initialize(sample_data, obv_instance):
    obv_instance.initialize(sample_data)
    assert isinstance(obv_instance.talipp_instance, OBV_talipp)


def test_output_values(sample_data, obv_instance):
    obv_instance.initialize(sample_data)

    expected_output_values = [10.0, 1.0, 12.0, 25.0, 11.0]

    assert obv_instance.output_values == expected_output_values


def test_add_value(sample_data, obv_instance):
    obv_instance.initialize(sample_data)

    new_value = pd.Series(
        {
            'open': 4.,
            "close": 7.,
            "high": 7.,
            "low": 4.,
            "volume": 20.,
        }
    )
    obv_instance.add_value(new_value)

    expected_output_values = [10.0, 1.0, 12.0, 25.0, 11.0, 31.0]

    assert obv_instance.output_values == expected_output_values


def test_name(obv_instance):
    assert obv_instance.name == "OBV__close"
