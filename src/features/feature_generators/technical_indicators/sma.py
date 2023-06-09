import numpy as np
from talipp.indicators import SMA as SMA_talipp

from src.features.feature_generators.abstract import FeatureGenerator


class SMA(FeatureGenerator):
    def __init__(self, input_col: str, period: int) -> None:
        super().__init__()
        self.talipp_instance = None
        self.input_col = input_col
        self.period = period

    def initialize(self, data):
        self.talipp_instance = SMA_talipp(
            input_values=data[self.input_col],
            period=self.period,
        )

    def add_value(self, data, purging: bool = False):
        self.talipp_instance.add_input_value(data[self.input_col])
        if purging is True:
            self.talipp_instance.purge_oldest(1)

    @property
    def output_values(self):
        assert self.talipp_instance is not None, "Initialize first."
        nans = (self.period - 1) * [np.nan]
        return nans + self.talipp_instance.output_values

    @property
    def name(self):
        return f"SMA__{self.input_col}__{self.period}"
