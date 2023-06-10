from abc import ABC, abstractmethod

from torch import Tensor


class Preprocessor(ABC):
    """
    Abstract class for preprocessors
    to be used within a model wrapper.
    """

    @abstractmethod
    def fit(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def transform(self) -> Tensor:
        raise NotImplementedError()

    @abstractmethod
    def fit_transform(self) -> Tensor:
        raise NotImplementedError()
