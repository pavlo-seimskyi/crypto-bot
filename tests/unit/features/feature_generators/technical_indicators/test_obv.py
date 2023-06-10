import pandas as pd
import pytest
from talipp.indicators import OBV as OBV_talipp

from src.features.feature_generators.technical_indicators import OBV


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "close": [3.0, 2.0, 3.0, 5.0, 4.0],
            "volume": [10.0, 9.0, 11.0, 13.0, 14.0],
        }
    )


@pytest.fixture
def obv_instance():
    return OBV(
        close_col="close",
        volume_col="volume",
    )


@pytest.mark.unit
def test_initialize(sample_data, obv_instance):
    obv_instance.initialize(sample_data)
    assert isinstance(obv_instance.talipp_instance, OBV_talipp)


@pytest.mark.unit
def test_output_values(sample_data, obv_instance):
    obv_instance.initialize(sample_data)

    expected_output_values = [10.0, 1.0, 12.0, 25.0, 11.0]

    assert obv_instance.output_values == expected_output_values


@pytest.mark.unit
def test_add_value(sample_data, obv_instance):
    obv_instance.initialize(sample_data)

    new_value = pd.Series(
        {
            "close": 7.0,
            "volume": 20.0,
        }
    )
    obv_instance.add_value(new_value)

    expected_output_values = [10.0, 1.0, 12.0, 25.0, 11.0, 31.0]

    assert obv_instance.output_values == expected_output_values


@pytest.mark.unit
def test_name(obv_instance):
    assert obv_instance.name == "OBV__close"
