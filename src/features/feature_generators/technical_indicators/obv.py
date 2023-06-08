from talipp.indicators import OBV as OBV_talipp
from talipp.ohlcv import OHLCVFactory

from src.features.feature_generators.abstract import FeatureGenerator


class OBV(FeatureGenerator):
    def __init__(
        self,
        open_col: str,
        close_col: str,
        high_col: str,
        low_col: str,
        volume_col: str,
    ) -> None:
        super().__init__()
        self.talipp_instance = None
        self.open_col = open_col
        self.close_col = close_col
        self.high_col = high_col
        self.low_col = low_col
        self.volume_col = volume_col

    def initialize(self, data):
        ohlcvs_factory = OHLCVFactory.from_dict(
            {
                "open": data[self.open_col],
                "close": data[self.close_col],
                "high": data[self.high_col],
                "low": data[self.low_col],
                "volume": data[self.volume_col],
            }
        )
        self.talipp_instance = OBV_talipp(ohlcvs_factory)

    def add_value(self, data, purging: bool = False):
        new_value = OHLCVFactory.from_dict(
            {
                "open": [data[self.open_col]],
                "close": [data[self.close_col]],
                "high": [data[self.high_col]],
                "low": [data[self.low_col]],
                "volume": [data[self.volume_col]],
            }
        )
        self.talipp_instance.add_input_value(new_value)
        if purging is True:
            self.talipp_instance.purge_oldest(1)

    @property
    def output_values(self):
        assert self.talipp_instance is not None, "Initialize first."
        return self.talipp_instance.output_values

    @property
    def name(self):
        return f"OBV__{self.close_col}"
