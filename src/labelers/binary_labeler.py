from typing import List
import numpy as np

from src.labelers.abstract import Labeler


class BinaryLabeler(Labeler):
    def __init__(self, price_col: str, period: int) -> None:
        """
        Label depending on whether the future prices go up.

        Parameters
        ----------
        price_col : str
            Column/key in the data object that contains the prices.
        period : int
            How many steps ahead is the future price.

        Returns
        -------
        List of `1` if the price went up, `0` if it didn't.
        Filled with NaNs at the end.
        """
        super().__init__()
        self.price_col = price_col
        self.period = period
        self.validate()

    def transform(self, data):
        prices = data[self.price_col]
        future_prices = np.roll(a=prices, shift=-self.period)
        labels = (future_prices > prices).astype(float).tolist()
        return labels[:-self.period] + self.period * [np.nan]

    def validate(self):
        assert self.period > 0, f"Period must be positive, got {self.period}."
