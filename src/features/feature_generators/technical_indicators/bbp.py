import numpy as np
from talipp.indicators import BB as BB_talipp

from src.features.feature_generators.abstract import FeatureGenerator


class BBP(FeatureGenerator):
    """
    Bollinger Bands Percentile.
    """
    
    def __init__(
        self, input_col: str, period: int = 14, std_dev_multiplier: int = 2
    ) -> None:
        super().__init__()
        self.talipp_instance = None
        self.input_col = input_col
        self.period = period
        self.std_dev_multiplier = std_dev_multiplier

    def initialize(self, data):
        self.talipp_instance = BB_talipp(
            input_values=data[self.input_col],
            period=self.period,
            std_dev_multiplier=self.std_dev_multiplier,
        )

    def add_value(self, data, purging: bool = False):
        self.talipp_instance.add_input_value(data[self.input_col])
        if purging is True:
            self.talipp_instance.purge_oldest(1)

    @property
    def output_values(self):
        assert self.talipp_instance is not None, "Initialize first."
        nans = (self.period - 1) * [np.nan]
        values = self.talipp_instance.output_values
        lower = np.array(nans + [v.lb for v in values])
        upper = np.array(nans + [v.ub for v in values])
        prices = np.array(self.talipp_instance.input_values)
        bb_percentile = (prices - lower) / (upper - lower)
        return bb_percentile.tolist()
        return {
            "lower": nans + [v.lb for v in values],
            "middle": nans + [v.cb for v in values],
            "upper": nans + [v.ub for v in values],
        }

    @property
    def name(self):
        return f"BB__{self.input_col}"
