from abc import ABC, abstractmethod


class NewsClient(ABC):
    """Abstract class for ExchangeClients."""

    @abstractmethod
    def get_data(self, **kwargs):
        raise NotImplementedError()
