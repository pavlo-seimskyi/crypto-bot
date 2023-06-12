import numpy as np
from talipp.indicators import VWAP as VWAP_talipp
from talipp.ohlcv import OHLCVFactory

from src.features.feature_generators.abstract import FeatureGenerator


class VWAP(FeatureGenerator):
    """
    Volume Weighted Average Price
    """

    def __init__(
        self,
        high_col: str,
        low_col: str,
        close_col: str,
        volume_col: str,
    ) -> None:
        super().__init__()
        self.talipp_instance = None
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col

    def initialize(self, data):
        values = OHLCVFactory.from_dict(
            {
                "high": data[self.high_col],
                "low": data[self.low_col],
                "close": data[self.close_col],
                "volume": data[self.volume_col],
            }
        )
        self.talipp_instance = VWAP_talipp(input_values=values)

    def add_value(self, data, purging: bool = False):
        new_value = OHLCVFactory.from_dict(
            {
                "high": [data[self.high_col]],
                "low": [data[self.low_col]],
                "close": [data[self.close_col]],
                "volume": [data[self.volume_col]],
            }
        )
        self.talipp_instance.add_input_value(new_value)
        if purging is True:
            self.talipp_instance.purge_oldest(1)

    @property
    def output_values(self):
        assert self.talipp_instance is not None, "Initialize first."
        # nans = (self.period - 1) * [np.nan]
        return self.talipp_instance.output_values

    @property
    def name(self):
        return f"VWAP__{self.close_col}"
