from abc import ABC, abstractmethod

from torch import Tensor


class ModelWrapper(ABC):
    """
    Abstract class for model wrappers.

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
