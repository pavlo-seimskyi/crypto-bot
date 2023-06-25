import numpy as np
import pandas as pd
import pytest
from talipp.indicators import BB as BB_talipp

from src.features.feature_generators.technical_indicators import BBP


@pytest.fixture
def sample_data():
    return pd.DataFrame({"price": [1.0, 2.0, 3.5, 1.0, 5.0]})


@pytest.fixture
def bbp_instance():
    return BBP(input_col="price", period=3, std_dev_multiplier=2)


@pytest.mark.unit
def test_initialize(sample_data, bbp_instance):
    bbp_instance.initialize(sample_data)
    assert isinstance(bbp_instance.talipp_instance, BB_talipp)


@pytest.mark.unit
def test_initialize_range_0(bbp_instance):
    # BBP should be 0.5 when the upper and lower bands are equal
    sample_data = pd.DataFrame({"price": [1.0, 1.0, 1.0, 1.0, 1.0]})

    bbp_instance.initialize(sample_data)

    expected_output_values = [np.nan, np.nan, 0.5, 0.5, 0.5]

    assert bbp_instance.output_values == expected_output_values


@pytest.mark.unit
def test_output_values(sample_data, bbp_instance):
    bbp_instance.initialize(sample_data)

    expected_output_values = [
        np.nan,
        np.nan,
        0.8244428422615251,
        0.2161125130211656,
        0.7777919497518578,
    ]

    assert bbp_instance.output_values == expected_output_values


@pytest.mark.unit
def test_add_value(sample_data, bbp_instance):
    bbp_instance.initialize(sample_data)

    new_value = pd.Series({"price": 100.0})
    bbp_instance.add_value(new_value)

    expected_output_values = [
        np.nan,
        np.nan,
        0.8244428422615251,
        0.2161125130211656,
        0.7777919497518578,
        0.8533281495061139,
    ]

    assert bbp_instance.output_values == expected_output_values


@pytest.mark.unit
def test_name(bbp_instance):
    assert bbp_instance.name == "BBP__price__period_3__std_mul_2"
