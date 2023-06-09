from typing import Dict, List

import pandas as pd

from src.dataloaders.abstract import DataLoader
from src.features.feature_generators.abstract import FeatureGenerator


class DateTime(FeatureGenerator):
    def __init__(self, timestamp_col: str) -> None:
        super().__init__()
        self.time_series = None
        self.timestamp_col = timestamp_col

    def initialize(self, data):
        self.time_series = self.to_dt_series(data[self.timestamp_col])
        self.time_series.name = None

    @staticmethod
    def to_dt_series(data) -> pd.Series:
        return pd.Series(data).apply(DataLoader.timestamp_to_datetime)

    def add_value(self, data, purging: bool = False):
        self.time_series = pd.concat(
            (self.time_series, self.to_dt_series(data[self.timestamp_col])),
            ignore_index=True,
        )
        if purging is True:
            self.time_series = self.time_series.iloc[1:]

    @property
    def output_values(self):
        assert self.time_series is not None, "Initialize first."
        ts = self.time_series
        time_of_day = ts.dt.hour.apply(DateTime.time_of_day)
        return {
            **self.to_one_hot(ts.dt.dayofweek, "day_of_week_"),
            **self.to_one_hot(ts.dt.month, "month_"),
            **self.to_one_hot(ts.dt.quarter, "quarter_"),
            **self.to_one_hot(time_of_day, "time_of_day_"),
        }

    @staticmethod
    def time_of_day(hour: int) -> str:
        if hour >= 5 and hour < 12:
            return "morning"
        elif hour >= 12 and hour < 17:
            return "afternoon"
        elif hour >= 17 and hour < 21:
            return "evening"
        else:
            return "night"

    @staticmethod
    def to_one_hot(data, prefix: str = None) -> Dict[str, List[float]]:
        series = pd.Series(data)
        drop_first = series.nunique() > 1
        one_hot_encoded = pd.get_dummies(
            series, prefix=prefix, drop_first=drop_first
        ).astype(float)
        return {
            col: one_hot_encoded[col].tolist()
            for col in one_hot_encoded.columns
        }

    @property
    def name(self):
        return "DateTime"
