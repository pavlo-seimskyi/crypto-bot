from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Dict, List


class FeatureGenerator(ABC):
    """Abstract class for feature generators."""

    @abstractmethod
    def initialize(self, data: Any) -> None:
        """Initialize the generator and its output values."""
        raise NotImplementedError()

    @abstractmethod
    def add_value(self, data: Any, purging: bool) -> None:
        """Add a new value to the feature generator."""
        raise NotImplementedError()

    @abstractproperty
    def output_values(self) -> Dict[str, List[float]]:
        raise NotImplementedError()

    @abstractproperty
    def name(self) -> str:
        raise NotImplementedError()
