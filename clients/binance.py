from binance.client import Client


class BinanceClient:
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize BinanceClient instance.

        Parameters
        ----------
        api_key : str
            API key for Binance.
        api_secret : str
            API secret for Binance.
        """
        self.client = Client(api_key=api_key, api_secret=api_secret)

    def get_historic_prices(
        self,
        symbol: str,
        interval: str,
        start: int,
        end: int,
    ):
        """
        Get historical exchange rates for a currency pair from Binance.

        Parameters
        ----------
        symbol : str
            Currency pair symbol, such as `ETHUSDT`.
        interval : str
            Time interval, one of `binance.enums`.
        start_timestamp : int
            Start timestamp.
        end_timestamp : int
            End timestamp.

        Returns
        -------
        list
            Historical exchange rates for the given currency pair.
        """
        return self.client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start,
            end_str=end,
        )
