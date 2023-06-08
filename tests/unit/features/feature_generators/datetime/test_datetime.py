from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from src.features.feature_generators.datetime import DateTime


@pytest.fixture
def sample_data():
    return [
        1598798820000,  # 2020-08-30 14:47:00
        1610789040000,  # 2021-01-16 09:24:00
        1650080340000,  # 2022-04-16 03:39:00
        1667627160000,  # 2022-11-05 05:46:00
        1673979900000,  # 2023-01-17 18:25:00
    ]


@pytest.fixture
def sample_data_later():
    return [1681928160000]  # 2023-04-19 18:16:00


def test_initialize(sample_data):
    timestamp_col = "timestamp"
    data = pd.DataFrame({timestamp_col: sample_data})

    dt = DateTime(timestamp_col)
    dt.initialize(data)

    expected_series = pd.Series(
        {
            0: pd.Timestamp("2020-08-30 14:47:00+0000", tz="UTC"),
            1: pd.Timestamp("2021-01-16 09:24:00+0000", tz="UTC"),
            2: pd.Timestamp("2022-04-16 03:39:00+0000", tz="UTC"),
            3: pd.Timestamp("2022-11-05 05:46:00+0000", tz="UTC"),
            4: pd.Timestamp("2023-01-17 18:25:00+0000", tz="UTC"),
        },
    )

    pd.testing.assert_series_equal(dt.time_series, expected_series)


def test_add_value(sample_data, sample_data_later):
    timestamp_col = "timestamp"
    data = pd.DataFrame({timestamp_col: sample_data})
    data_later = pd.Series({timestamp_col: sample_data_later})

    dt = DateTime(timestamp_col)
    dt.initialize(data)
    dt.add_value(data_later)

    expected_series = pd.Series(
        {
            0: pd.Timestamp("2020-08-30 14:47:00+0000", tz="UTC"),
            1: pd.Timestamp("2021-01-16 09:24:00+0000", tz="UTC"),
            2: pd.Timestamp("2022-04-16 03:39:00+0000", tz="UTC"),
            3: pd.Timestamp("2022-11-05 05:46:00+0000", tz="UTC"),
            4: pd.Timestamp("2023-01-17 18:25:00+0000", tz="UTC"),
            5: pd.Timestamp("2023-04-19 18:16:00+0000", tz="UTC"),
        },
    )

    pd.testing.assert_series_equal(dt.time_series, expected_series)


def test_output_values(sample_data):
    timestamp_col = "timestamp"
    data = pd.DataFrame({timestamp_col: sample_data})

    dt = DateTime(timestamp_col)
    dt.initialize(data)
    output_values = dt.output_values

    expected_output_values = {
        "day_of_week__5": [0.0, 1.0, 1.0, 1.0, 0.0],
        "day_of_week__6": [1.0, 0.0, 0.0, 0.0, 0.0],
        "month__4": [0.0, 0.0, 1.0, 0.0, 0.0],
        "month__8": [1.0, 0.0, 0.0, 0.0, 0.0],
        "month__11": [0.0, 0.0, 0.0, 1.0, 0.0],
        "quarter__2": [0.0, 0.0, 1.0, 0.0, 0.0],
        "quarter__3": [1.0, 0.0, 0.0, 0.0, 0.0],
        "quarter__4": [0.0, 0.0, 0.0, 1.0, 0.0],
        "time_of_day__evening": [0.0, 0.0, 0.0, 0.0, 1.0],
        "time_of_day__morning": [0.0, 1.0, 0.0, 1.0, 0.0],
        "time_of_day__night": [0.0, 0.0, 1.0, 0.0, 0.0],
    }

    assert output_values == expected_output_values


def test_to_one_hot_one_unique():
    data = [0, 0, 0, 0, 0]
    expected_result = {
        # All 1s
        "prefix__0": [1.0, 1.0, 1.0, 1.0, 1.0],
    }

    result = DateTime.to_one_hot(data, "prefix_")

    assert result == expected_result


def test_to_one_hot_more_than_one_unique():
    data = [1, 2, 1, 3, 1, 2]
    expected_result = {
        # First column is dropped
        "prefix__2": [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "prefix__3": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    }

    result = DateTime.to_one_hot(data, "prefix_")

    assert result == expected_result


def test_time_of_day():
    assert DateTime.time_of_day(6) == "morning"
    assert DateTime.time_of_day(12) == "afternoon"
    assert DateTime.time_of_day(18) == "evening"
    assert DateTime.time_of_day(0) == "night"
