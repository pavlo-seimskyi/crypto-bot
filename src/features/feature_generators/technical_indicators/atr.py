import numpy as np
from talipp.indicators import ATR as ATR_talipp
from talipp.ohlcv import OHLCVFactory

from src.features.feature_generators.abstract import FeatureGenerator


class ATR(FeatureGenerator):
    """
    Average True Range.
    """

    def __init__(
        self, high_col: str, low_col: str, close_col: str, period: int = 14
    ) -> None:
        super().__init__()
        self.talipp_instance = None
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.period = period

    def initialize(self, data):
        input_values = OHLCVFactory.from_dict(
            {
                "high": data[self.high_col],
                "low": data[self.low_col],
                "close": data[self.close_col],
            }
        )
        self.talipp_instance = ATR_talipp(
            input_values=input_values,
            period=self.period,
        )

    def add_value(self, data, purging: bool = False):
        new_value = OHLCVFactory.from_dict(
            {
                "high": [data[self.high_col]],
                "low": [data[self.low_col]],
                "close": [data[self.close_col]],
            }
        )
        self.talipp_instance.add_input_value(new_value)
        if purging is True:
            self.talipp_instance.purge_oldest(1)

    @property
    def output_values(self):
        assert self.talipp_instance is not None, "Initialize first."
        nans = (self.period - 1) * [np.nan]
        return nans + self.talipp_instance.output_values

    @property
    def name(self):
        return f"ATR__{self.close_col}__{self.period}"
