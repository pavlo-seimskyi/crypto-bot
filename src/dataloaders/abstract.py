from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import *
from datetime import datetime
import pytz


@dataclass
class DataLoader(ABC):
    """Abstract class for DataLoaders."""

    datetime_fmt: str = field(default="%Y-%m-%d %H:%M:%S", init=False)

    # @abstractmethod
    def load():
        raise NotImplementedError()

    def date_to_timestamp(self, date: str) -> int:
        dt = datetime.strptime(date, self.datetime_fmt)
        dt = dt.replace(tzinfo=pytz.utc)
        return int(dt.timestamp() // 1) * 1000

    def timestamp_to_date(self, timestamp: int) -> str:
        timestamp /= 1000
        dt = datetime.fromtimestamp(timestamp, tz=pytz.utc)
        return dt.strftime(self.datetime_fmt)

    def partition_timestamps_into_days(
        self, start: int, end: int
    ) -> List[Tuple[int, int]]:
        # hours * minutes * seconds * milliseconds
        milliseconds_per_day = 24 * 60 * 60 * 1000
        daily_chunks = []

        current_timestamp = start
        next_midnight = (start // milliseconds_per_day + 1) * milliseconds_per_day

        # Add the first chunk, which might be incomplete
        if next_midnight <= end:
            daily_chunks.append((current_timestamp, next_midnight))
            current_timestamp = next_midnight

        # Add the complete chunks
        while current_timestamp + milliseconds_per_day <= end:
            next_day = current_timestamp + milliseconds_per_day
            daily_chunks.append((current_timestamp, next_day))
            current_timestamp = next_day

        # Add the last chunk, which might be incomplete
        if current_timestamp < end:
            daily_chunks.append((current_timestamp, end))

        return daily_chunks

    @property
    def now(self):
        return int(datetime.now(pytz.utc).timestamp() // 1) * 1000
