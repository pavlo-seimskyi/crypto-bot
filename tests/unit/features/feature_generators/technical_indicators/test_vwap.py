import pandas as pd
import pytest
from talipp.indicators import VWAP as VWAP_talipp

from src.features.feature_generators.technical_indicators import VWAP


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "high": [4.0, 3.0, 4.0, 6.0, 5.5],
            "low": [2.0, 1.0, 2.0, 4.5, 1.5],
            "close": [3.0, 1.5, 3.5, 5.0, 4.0],
            "volume": [10.0, 9.0, 11.0, 13.0, 14.0],
        }
    )


@pytest.fixture
def vwap_instance():
    return VWAP(
        high_col="high",
        low_col="low",
        close_col="close",
        volume_col="volume",
    )


@pytest.mark.unit
def test_initialize(sample_data, vwap_instance):
    vwap_instance.initialize(sample_data)
    assert isinstance(vwap_instance.talipp_instance, VWAP_talipp)


@pytest.mark.unit
def test_output_values(sample_data, vwap_instance):
    vwap_instance.initialize(sample_data)

    expected_output_values = [
        3.0,
        2.4473684210526314,
        2.711111111111111,
        3.453488372093023,
        3.505847953216374
    ]

    assert vwap_instance.output_values == expected_output_values


@pytest.mark.unit
def test_add_value(sample_data, vwap_instance):
    vwap_instance.initialize(sample_data)

    new_value = pd.Series(
        {
            "high": 11.0,
            "low": 5.0,
            "close": 9.0,
            "volume": 20.0,
        }
    )
    vwap_instance.add_value(new_value)

    expected_output_values = [
        3.0,
        2.4473684210526314,
        2.711111111111111,
        3.453488372093023,
        3.505847953216374,
        4.759740259740259
    ]

    assert vwap_instance.output_values == expected_output_values


@pytest.mark.unit
def test_name(vwap_instance):
    assert vwap_instance.name == "VWAP__close"
