from torch import Tensor


class Normalizer:
    """
    scikit-learn-like standard scaler that accepts Tensor data.
    Brings mean to 0 and standard deviation to 1.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: Tensor) -> None:
        self.mean = x.mean()
        self.std = x.std()

    def transform(self, x: Tensor) -> Tensor:
        assert (
            self.mean is not None and self.std is not None
        ), "Fit the normalizer first."
        return (x - self.mean) / self.std

    def fit_transform(self, x: Tensor) -> Tensor:
        self.fit(x)
        return self.transform(x)
