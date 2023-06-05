from binance import enums
from binance.client import Client

from clients.exchange.abstract import ExchangeClient


class BinanceClient(ExchangeClient):
    def __init__(
        self,
        api_key: str,
        api_secret: str,
    ):
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
        self.dtypes = {
            "open_timestamp": int,
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float,
            "close_timestamp": int,
            "quote_asset_volume": float,
            "number_of_trades": int,
            "taker_buy_base_asset_volume": float,
            "taker_buy_quote_asset_volume": float,
        }
        self.interval_mapping = {
            enums.KLINE_INTERVAL_1MINUTE: "1T",
            enums.KLINE_INTERVAL_3MINUTE: "3T",
            enums.KLINE_INTERVAL_5MINUTE: "5T",
            enums.KLINE_INTERVAL_15MINUTE: "15T",
            enums.KLINE_INTERVAL_30MINUTE: "30T",
            enums.KLINE_INTERVAL_1HOUR: "1H",
            enums.KLINE_INTERVAL_2HOUR: "2H",
            enums.KLINE_INTERVAL_4HOUR: "4H",
            enums.KLINE_INTERVAL_6HOUR: "6H",
            enums.KLINE_INTERVAL_8HOUR: "8H",
            enums.KLINE_INTERVAL_12HOUR: "12H",
            enums.KLINE_INTERVAL_1DAY: "1d",
            enums.KLINE_INTERVAL_3DAY: "3d",
            enums.KLINE_INTERVAL_1WEEK: "1w",
            enums.KLINE_INTERVAL_1MONTH: "1M",
        }

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
