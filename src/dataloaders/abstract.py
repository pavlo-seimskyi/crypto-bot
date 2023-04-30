from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import *

import pytz


@dataclass
class DataLoader(ABC):
    """Abstract class for DataLoaders."""

    datetime_fmt: str = field(default="%Y-%m-%d %H:%M:%S", init=False)

    @abstractmethod
    def load_data():
        raise NotImplementedError()

    def date_to_timestamp(self, date: str) -> int:
        dt = datetime.strptime(date, self.datetime_fmt)
        dt = dt.replace(tzinfo=pytz.utc)
        return int(dt.timestamp() // 1) * 1000

    def timestamp_to_date(self, timestamp: int) -> str:
        timestamp /= 1000
        dt = datetime.fromtimestamp(timestamp, tz=pytz.utc)
        return dt.strftime(self.datetime_fmt)

    @property
    def now(self):
        return int(datetime.now(pytz.utc).timestamp() // 1) * 1000
