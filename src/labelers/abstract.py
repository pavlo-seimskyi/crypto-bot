from abc import ABC, abstractmethod
from typing import List


class Labeler(ABC):
    """
    Abstract class for labelers.
    """

    @abstractmethod
    def transform(self) -> List[float]:
        raise NotImplementedError()
