from abc import ABC, abstractmethod


class NewsClient(ABC):
    """Abstract class for NewsClients."""

    @abstractmethod
    def get_data(self, **kwargs):
        raise NotImplementedError()
