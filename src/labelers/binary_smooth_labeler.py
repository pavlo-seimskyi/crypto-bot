import numpy as np

from src.labelers.abstract import Labeler


class BinarySmoothLabeler(Labeler):
    def __init__(self, price_col: str, period: int) -> None:
        """
        Label depending on whether the avg. future prices
        within the defined `period` are above the current prices.

        Parameters
        ----------
        price_col : str
            Column/key in the data object that contains the prices.
        period : int
            How many steps ahead is the future price.

        Returns
        -------
        List of `1` if the avg. price went up, `0` if it didn't.
        Filled with NaNs at the end.
        """
        super().__init__()
        self.price_col = price_col
        self.period = period
        self.validate()

    def transform(self, data):
        prices = data[self.price_col]
        # calculate the sliding window view
        window_view = np.lib.stride_tricks.sliding_window_view(
            prices, self.period + 1
        )
        # calculate the average of the next 'period' values for each window
        avg_ahead = np.mean(window_view[:, 1:], axis=1)
        # check whether each average is greater than the corresponding value
        labels = (avg_ahead > window_view[:, 0]).astype(float)
        # append NaNs for the last 'period' values
        return labels.tolist() + [np.nan] * self.period

    def validate(self):
        assert self.period > 0, f"Period must be positive, got {self.period}."
