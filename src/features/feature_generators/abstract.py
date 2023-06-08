from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Dict, List, Union

from pandas import DataFrame, Series


class FeatureGenerator(ABC):
    """Abstract class for feature generators."""

    @abstractmethod
    def initialize(self, data: Union[DataFrame, Dict[str, List[Any]]]) -> None:
        """Initialize the generator and its output values."""
        raise NotImplementedError()

    @abstractmethod
    def add_value(self, data: Union[Series, Dict[str, Any]]) -> None:
        """Add a new value to the feature generator."""
        raise NotImplementedError()

    @abstractproperty
    def output_values(self) -> Dict[str, Any]:
        raise NotImplementedError()

    @abstractproperty
    def name(self) -> str:
        raise NotImplementedError()
