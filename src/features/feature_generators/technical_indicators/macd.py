from typing import Any, List
import numpy as np
from talipp.indicators import MACD as MACD_talipp

from src.features.feature_generators.abstract import FeatureGenerator


class MACD(FeatureGenerator):
    def __init__(
        self,
        input_col: str,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> None:
        super().__init__()
        self.talipp_instance = None
        self.input_col = input_col
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def initialize(self, data):
        self.talipp_instance = MACD_talipp(
            input_values=data[self.input_col],
            fast_period=self.fast_period,
            slow_period=self.slow_period,
            signal_period=self.signal_period,
        )

    def add_value(self, data, purging: bool = False):
        self.talipp_instance.add_input_value(data[self.input_col])
        if purging is True:
            self.talipp_instance.purge_oldest(1)

    @property
    def output_values(self):
        assert self.talipp_instance is not None, "Initialize first."
        nans = (self.slow_period - 1) * [np.nan]
        values = self.talipp_instance.output_values
        return {
            "line": nans + self.none_to_nan([v.macd for v in values]),
            "signal": nans + self.none_to_nan([v.signal for v in values]),
            "histogram": nans
            + self.none_to_nan([v.histogram for v in values]),
        }

    @staticmethod
    def none_to_nan(input_list: List[Any]) -> List[Any]:
        return [np.nan if x is None else x for x in input_list]

    @property
    def name(self):
        return (
            f"MACD__{self.input_col}__"
            f"fast_{self.fast_period}__"
            f"slow_{self.slow_period}__"
            f"signal_{self.signal_period}"
        )
