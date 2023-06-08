import numpy as np
from talipp.indicators import RSI as RSI_talipp

from src.features.feature_generators.abstract import FeatureGenerator


class RSI(FeatureGenerator):
    def __init__(self, input_col: str, period: int = 14) -> None:
        super().__init__()
        self.talipp_instance = None
        self.input_col = input_col
        self.period = period

    def initialize(self, data):
        self.talipp_instance = RSI_talipp(
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
        nans = self.period * [np.nan]
        return nans + self.talipp_instance.output_values

    @property
    def name(self):
        return f"RSI__{self.input_col}__{self.period}"
