import http.client
import json
import urllib.parse
from dataclasses import dataclass
from typing import List

import pandas as pd

from clients import ENV
from src.dataloaders.abstract_dataloader import DataLoader


@dataclass
class MediaStackNewsScraper(DataLoader):
    keywords: List[str]
    access_key: str = ENV["MEDIASTACK_ACCESS_KEY"]

    def __post_init__(self):
        self.connection = http.client.HTTPConnection("api.mediastack.com")
        # Daily granularity of data loading
        self.datetime_fmt = "%Y-%m-%d"
        self.validate()

    def load_data(self, start: int, end: int) -> pd.DataFrame:
        """Load news data from MediaStack.

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
        data = []
        for date in self.get_dates(start, end):
            data_for_date = self.get_data_for_date(date)
            data.extend(data_for_date)
        return data

    def get_dates(self, start: int, end: int) -> List[str]:
        dates = pd.date_range(
            start=self.timestamp_to_str(start),
            end=self.timestamp_to_str(end),
            freq="D",
        )
        return [d.strftime(self.datetime_fmt) for d in dates]

    def get_data_for_date(self, date: str) -> List[dict]:
        """
        Parameters
        ----------
        date : str
            Date in the format `YYYY-MM-DD`.

        Returns
        -------
        List of news article headlines.

        Example
        -------
        Each article looks like this:
            {
                "author": "Jeff Newmond",
                "title": "Vaultoro Unveils...",
                "description": "Vaultoro has launched a new product...",
                "url": "https://www.businessmole.com/...",
                "source": "Businessmole",
                "image": None,
                "category": "business",
                "language": "en",
                "country": "us",
                "published_at": "2023-05-03T11:35:55+00:00",
            }

        """
        params = urllib.parse.urlencode(
            {
                "access_key": self.access_key,
                "keywords": ",".join(self.keywords),
                "date": date,
                "limit": 100,
                "languages": "en",
                "sort": "popularity",
            }
        )
        self.connection.request("GET", f"/v1/news?{params}")
        response = self.connection.getresponse()
        bytes_data = response.read()
        return json.loads(bytes_data)["data"]

    def validate(self):
        pass
