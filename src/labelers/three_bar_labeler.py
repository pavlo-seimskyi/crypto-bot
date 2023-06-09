from typing import List

import numpy as np

from src.labelers.abstract import Labeler


class ThreeBarLabeler(Labeler):
    def __init__(
        self, price_col: str, period: int, threshold_pct: float
    ) -> None:
        """
        Label depending on relative changes in future prices.

        Parameters
        ----------
        price_col : str
            Column/key in the data object that contains the prices.
        period : int
            How many steps ahead is the future price.
        threshold: float
            Threshold (+/-) for the price change to surpass.
            1.0 = 1%.

        Returns
        -------
        List of buy/sell/hold signals, filled with NaNs at the end.
            0. = Sell signal. Price will drop below the threshold.
            1. = Hold signal. Price will not surpass the threshold
            2. = Buy signal. Price will rise above the threshold.
        """
        super().__init__()
        self.price_col = price_col
        self.period = period
        self.threshold_pct = threshold_pct
        self.validate()

    def transform(self, data):
        prices = np.array(data[self.price_col])
        future_prices = np.roll(a=prices, shift=-self.period)
        # Get % change in prices
        pct_diff = 100 * future_prices / prices - 100
        # Label depending on threshold
        conditions = [
            pct_diff < -self.threshold_pct,  # Below negative threshold
            ( # Within threshold range
                (pct_diff >= -self.threshold_pct) & 
                (pct_diff <= self.threshold_pct)
            ),  
            pct_diff > self.threshold_pct  # Above positive threshold
        ]
        choices = [0.0, 1.0, 2.0]
        labels = np.select(conditions, choices, default=np.nan).tolist()
        # Add NaNs to the end
        return labels[:-self.period] + self.period * [np.nan]

    def validate(self):
        assert self.period > 0, f"Period must be positive, got {self.period}."
        assert (
            self.threshold_pct > 0
        ), f"Threshold must be positive, got {self.threshold_pct}."
