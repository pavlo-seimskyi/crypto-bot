import torch
from torch import Tensor


class StandardScaler:
    """
    scikit-learn-like standard scaler that accepts Tensor data.
    Brings mean to 0 and standard deviation to 1.
    """

    def __init__(self, dim: int = 0):
        self.mean = None
        self.std = None
        self.dim = dim

    def fit(self, x: Tensor) -> None:
        self.mean = x.mean(dim=self.dim).unsqueeze(self.dim)
        self.std = x.std(dim=self.dim).unsqueeze(self.dim)

    def transform(self, x: Tensor) -> Tensor:
        assert (
            self.mean is not None and self.std is not None
        ), "Fit the scaler first."
        return torch.where(
            self.std != 0,
            torch.div(x - self.mean, self.std),
            0.0,  # return 0 if std is 0
        )

    def fit_transform(self, x: Tensor) -> Tensor:
        self.fit(x)
        return self.transform(x)
