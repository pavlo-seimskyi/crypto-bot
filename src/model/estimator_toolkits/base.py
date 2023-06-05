from abc import ABC, abstractmethod

from torch import Tensor


class BaseEstimator(ABC):
    """Abstract class for estimators."""

    @abstractmethod
    def fit(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def predict(self) -> Tensor:
        raise NotImplementedError()
