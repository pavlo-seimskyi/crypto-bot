from abc import ABC, abstractmethod

from torch import Tensor


class BaseEstimator(ABC):
    """
    Abstract class for estimator toolkits.

    This class is an abstraction layer to train and predict
    with model architectures from `src.model.architectures`
    with just `fit` and `predict` methods.
    """

    @abstractmethod
    def fit(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def predict(self) -> Tensor:
        raise NotImplementedError()
