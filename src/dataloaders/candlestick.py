from dataclasses import dataclass
from datetime import datetime
import os
from typing import List, Union

import pytz
import constants
import pandas as pd
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
    cache_dir: str = os.path.join(constants.BASE_PATH, "data", "binance")

    def __post_init__(self):
        self.client = BinanceClient(self.api_key, self.api_secret)
        self.dtypes = {
            "Open time": int,
            "Open": float,
            "High": float,
            "Low": float,
            "Close": float,
            "Volume": float,
            "Close time": int,
            "Quote asset volume": float,
            "Number of trades": int,
            "Taker buy base asset volume": float,
            "Taker buy quote asset volume": float,
            "Ignore": int,
        }
        self.data = pd.DataFrame()
        os.makedirs(self.cache_dir, exist_ok=True)

    @property
    def symbols(self):
        return [asset + self.fiat for asset in self.assets]

    def load(self, start: int, end: int):
        day_partitions = self.partition_timestamps_into_days(start, end)
        for date_start, date_end in day_partitions:
            date_end -= 1
            date_df = self.load_data_for_date(date_start, date_end)
            self.save_partition(date_df, date_start)
            self.data = pd.concat((self.data, date_df), ignore_index=True)
        return self.data

    def load_data_for_date(self, date_start: int, date_end: int):
        date_df = None
        for symbol in self.symbols:
            input_data = self.client.get_historic_prices(
                symbol=symbol,
                interval=self.interval,
                start=date_start,
                end=date_end,
            )
            symbol_df = self.read_raw_input_data(input_data, symbol)
            date_df = self.add_new_symbol_to_date_df(date_df, symbol_df)
        return date_df

    def read_raw_input_data(
        self, input_data: List[List[Union[int, float]]], symbol: str
    ) -> pd.DataFrame:
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
        data = pd.DataFrame(input_data, columns=self.dtypes.keys())
        data = data.astype(self.dtypes)
        data = data.drop("Ignore", axis=1)
        col_mapping = {
            col: f"{symbol}_{col}"
            for col in data.columns
            if not col in ("Open time", "Close time")
        }
        data = data.rename(columns=col_mapping)
        return data

    def add_new_symbol_to_date_df(self, date_df, symbol_df):
        if date_df is None:
            return symbol_df
        else:
            symbol_df = symbol_df.drop("Close time", axis=1)
            return pd.merge(date_df, symbol_df, on="Open time", how="outer")

    @property
    def saved_partitions(self):
        return [
            name for name in os.listdir(self.cache_dir) if name.startswith("date")
        ]
    
    def save_partition(self, data, timestamp):
        date = datetime.fromtimestamp(timestamp / 1000, tz=pytz.utc).strftime("%Y-%m-%d")
        os.makedirs(os.path.join(self.cache_dir, f"date={date}"), exist_ok=True)
        data.to_parquet(os.path.join(self.cache_dir, f"date={date}", "data.parquet"))
