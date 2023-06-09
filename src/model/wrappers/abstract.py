from abc import ABC, abstractmethod

from torch import Tensor
from torch.utils.data import DataLoader


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

    @abstractmethod
    def build_dataloader(self) -> DataLoader:
        raise NotImplementedError()
