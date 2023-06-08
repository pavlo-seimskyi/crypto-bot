from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Dict


class FeatureGenerator(ABC):
    """Abstract class for feature generators."""

    @abstractmethod
    def initialize(self):
        raise NotImplementedError()

    @abstractmethod
    def add_value(self):
        raise NotImplementedError()

    @abstractproperty
    def output_values(self) -> Dict[str, Any]:
        raise NotImplementedError()

    @abstractproperty
    def name(self):
        raise NotImplementedError()
