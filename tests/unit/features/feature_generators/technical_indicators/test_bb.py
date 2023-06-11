import numpy as np
import pandas as pd
import pytest
from talipp.indicators import BB as BB_talipp

from src.features.feature_generators.technical_indicators import BB


@pytest.fixture
def sample_data():
    return pd.DataFrame({"price": [1.0, 2.0, 1.0, 2.0, 4.0]})


@pytest.fixture
def bb_instance():
    return BB(input_col="price", period=2, std_dev_multiplier=2)


@pytest.mark.unit
def test_initialize(sample_data, bb_instance):
    bb_instance.initialize(sample_data)
    assert isinstance(bb_instance.talipp_instance, BB_talipp)


@pytest.mark.unit
def test_output_values(sample_data, bb_instance):
    bb_instance.initialize(sample_data)

    expected_output_values = {
        "lower": [np.nan, 0.5, 0.5, 0.5, 1.0],
        "middle": [np.nan, 1.5, 1.5, 1.5, 3.0],
        "upper": [np.nan, 2.5, 2.5, 2.5, 5.0],
    }

    assert bb_instance.output_values == expected_output_values


@pytest.mark.unit
def test_add_value(sample_data, bb_instance):
    bb_instance.initialize(sample_data)

    new_value = pd.Series({"price": 4.0})
    bb_instance.add_value(new_value)

    expected_output_values = {
        "lower": [np.nan, 0.5, 0.5, 0.5, 1.0, 4.0],
        "middle": [np.nan, 1.5, 1.5, 1.5, 3.0, 4.0],
        "upper": [np.nan, 2.5, 2.5, 2.5, 5.0, 4.0],
    }

    assert bb_instance.output_values == expected_output_values


@pytest.mark.unit
def test_name(bb_instance):
    assert bb_instance.name == "BB__price__period_2__std_mul_2"
