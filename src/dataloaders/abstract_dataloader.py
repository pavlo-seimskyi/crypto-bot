from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pytz


@dataclass
class DataLoader(ABC):
    """Abstract class for DataLoaders."""

    datetime_fmt: Optional[str] = field(default="%Y-%m-%d %H:%M:%S", init=False)
    # datetime_fmt: Optional[str] = "%Y-%m-%d %H:%M:%S"

    @abstractmethod
    def load_data():
        raise NotImplementedError()

    def str_to_datetime(self, date: str) -> datetime:
        dt = datetime.strptime(date, self.datetime_fmt)
        return dt.replace(tzinfo=pytz.utc)

    def datetime_to_str(self, dt: datetime) -> str:
        return dt.strftime(self.datetime_fmt)

    def timestamp_to_datetime(self, timestamp: int) -> datetime:
        timestamp /= 1000
        return datetime.fromtimestamp(timestamp, tz=pytz.utc)

    def datetime_to_timestamp(self, dt: datetime) -> int:
        return int(dt.timestamp() // 1) * 1000

    def str_to_timestamp(self, date: str) -> int:
        dt = self.str_to_datetime(date)
        return self.datetime_to_timestamp(dt)

    def timestamp_to_str(self, timestamp: int) -> str:
        dt = self.timestamp_to_datetime(timestamp)
        return self.datetime_to_str(dt)

    @property
    def now(self):
        return int(datetime.now(pytz.utc).timestamp() // 1) * 1000
