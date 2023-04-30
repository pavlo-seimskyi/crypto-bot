from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from binance import enums

from clients import ENV
from clients.binance import BinanceClient
from src.dataloaders.abstract import DataLoader


@dataclass
class CandleStickDataLoader(DataLoader):
    interval: str
    assets: List[str]
    fiat: str
    api_key: str = ENV["BINANCE_API_KEY"]
    api_secret: str = ENV["BINANCE_API_SECRET"]

    def __post_init__(self):
        self.client = BinanceClient(self.api_key, self.api_secret)
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
        self.validate()

    def load_data(self, start: int, end: int) -> pd.DataFrame:
        """Load candlestick data from Binance.

        Parameters
        ----------
        start : int
            Starting timestamp in milliseconds, like `1672531200000`.
        end : int
            Ending timestamp. Not inclusive.

        Returns
        -------
        Pandas data frame
        """
        data = pd.DataFrame()
        end -= 1  # `<` instead of `<=`
        for symbol in self.symbols:
            input_data = self.client.get_historic_prices(
                symbol=symbol,
                interval=self.interval,
                start=start,
                end=end,
            )
            data_for_symbol = self.read_raw_input_data(input_data)
            data_for_symbol["symbol"] = symbol
            data = pd.concat((data, data_for_symbol), ignore_index=True)
        data = self.pivot_price_data(data)
        data = self.process_missing_intervals(data)
        return data

    @property
    def symbols(self) -> List[str]:
        return [asset + self.fiat for asset in self.assets]

    def read_raw_input_data(self, input_data: List[List[str]]) -> pd.DataFrame:
        """Create a data frame from raw input data like:
        [
            [
                '1672531200000'  Open time
                '16541.77        Open
                '16545.70'       High
                '16508.39'       Low
                '16529.67'       Close
                '4364.83'        Volume
                '1672534799999'  Close time
                '72146293.58'    Quote asset volume
                '149854'         Number of trades
                '2179.94'        Taker buy base asset volume
                '36032352.87'    Taker buy quote asset volume
                '0'              Ignore
            ],
            ...
        ]
        """
        # Drop the last `Ignore` column
        input_data = np.array(input_data)[:, :-1]
        data = pd.DataFrame(input_data, columns=self.dtypes.keys())
        data = data.astype(self.dtypes)
        return data

    def pivot_price_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the dataframe so that each symbol gets its own column.

        Example
        -------
        Before:
            +----------------+----------+---------+
            | open_timestamp |     open |  symbol |
            +----------------+----------+---------+
            |           1111 | 17331.90 | BTCUSDT |
            |           1111 |  1553.43 | ETHUSDT |
            +----------------+----------+---------+
        After:
            +----------------+--------------+-------------+
            | open_timestamp | BTCUSDT_open | ETHUSDT_open|
            +----------------+--------------+-------------+
            |           1111 |      17331.9 |     1553.43 |
            +----------------+--------------+-------------+
        """
        cols = [
            col
            for col in self.dtypes.keys()
            if col not in ("open_timestamp", "close_timestamp")
        ]
        return pd.concat(
            {
                f"{asset}_{col}": data.pivot(
                    index="open_timestamp", columns="symbol", values=col
                )[asset]
                for col in cols
                for asset in data["symbol"].unique()
            },
            axis=1,
        ).reset_index()

    def process_missing_intervals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.extract_time(data)
        data = self.extend_missing_intervals(data)
        # Mark missing intervals
        data["service_down"] = np.where(data.isnull().any(axis=1), True, False)
        # Fill missing data to last available
        return data.fillna(method="ffill")

    def extract_time(self, data: pd.DataFrame) -> pd.DataFrame:
        time = pd.to_datetime(data["open_timestamp"].apply(self.timestamp_to_date))
        data.insert(loc=0, column="time", value=time)
        return data

    def extend_missing_intervals(self, data: pd.DataFrame) -> pd.DataFrame:
        full_range_df = pd.DataFrame(
            {
                "time": pd.date_range(
                    start=data["time"].min(),
                    end=data["time"].max(),
                    freq=self.interval_mapping[self.interval],
                )
            }
        )
        return full_range_df.merge(data, on="time", how="left")

    def validate(self):
        assert (
            self.interval in self.interval_mapping.keys()
        ), f"Invalid interval '{self.interval}'."
        assert len(self.symbols) > 0, "No symbols to get data for."
