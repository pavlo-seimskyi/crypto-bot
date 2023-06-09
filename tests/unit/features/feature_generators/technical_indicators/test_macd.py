import numpy as np
import pandas as pd
import pytest
from talipp.indicators import MACD as MACD_talipp

from src.features.feature_generators.technical_indicators import MACD


@pytest.fixture
def sample_data():
    return pd.DataFrame({"price": [1.0, 3.0, 2.0, 3.0, 5.0, 4.0, 3.0, 1.0]})


@pytest.fixture
def macd_instance():
    return MACD(
        input_col="price", fast_period=2, slow_period=5, signal_period=4
    )


def test_initialize(sample_data, macd_instance):
    macd_instance.initialize(sample_data)
    assert isinstance(macd_instance.talipp_instance, MACD_talipp)


def test_output_values(sample_data, macd_instance):
    macd_instance.initialize(sample_data)

    expected_output_values = {
        "line": [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            1.4222222222222225,
            0.8740740740740742,
            0.2246913580246912,
            -0.636213991769548,
        ],
        "signal": [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0.47119341563786,
        ],
        "histogram": [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            -1.107407407407408,
        ],
    }

    assert macd_instance.output_values == expected_output_values


def test_add_value(sample_data, macd_instance):
    macd_instance.initialize(sample_data)

    new_value = pd.Series({"price": 10.0})
    macd_instance.add_value(new_value)

    expected_output_values = {
        "line": [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            1.4222222222222225,
            0.8740740740740742,
            0.2246913580246912,
            -0.636213991769548,
            2.3138545953360756,
        ],
        "signal": [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0.47119341563786,
            1.2082578875171461,
        ],
        "histogram": [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            -1.107407407407408,
            1.1055967078189295,
        ],
    }

    assert macd_instance.output_values == expected_output_values


def test_name(macd_instance):
    assert macd_instance.name == "MACD__price__fast_2__slow_5__signal_4"
