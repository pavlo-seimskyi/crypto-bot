from abc import ABC, abstractmethod


class ExchangeClient(ABC):
    """Abstract class for ExchangeClients."""

    @abstractmethod
    def get_historic_prices(self, **kwargs):
        raise NotImplementedError()
