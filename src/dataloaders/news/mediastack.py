from typing import List

import pandas as pd

from clients.news.abstract import NewsClient
from src.dataloaders.abstract import DataLoader


class MediaStackNewsScraper(DataLoader):
    def __init__(
        self,
        keywords: List[str],
        news_client: NewsClient,
        language: str = "en",
        limit: int = 100,
        sort_by: str = "popularity",
        datetime_fmt="%Y-%m-%d",
    ):
        super().__init__(datetime_fmt=datetime_fmt)
        self.news_client = news_client
        self.keywords = keywords
        self.language = language
        self.limit = limit
        self.sort_by = sort_by

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
            data_for_date = self.news_client.get_data(
                date=date,
                keywords=self.keywords,
                limit=self.limit,
                language=self.language,
                sort_by=self.sort_by,
            )
            data.extend(data_for_date)
        return data

    def get_dates(self, start: int, end: int) -> List[str]:
        dates = pd.date_range(
            start=self.timestamp_to_str(start),
            end=self.timestamp_to_str(end),
            freq="D",
        )
        return [d.strftime(self.datetime_fmt) for d in dates]
