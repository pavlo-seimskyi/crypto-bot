import http.client
import json
import urllib.parse
from typing import List

from clients.news.abstract import NewsClient


class MediaStackNewsClient(NewsClient):
    def __init__(self, access_key: str):
        self.access_key = access_key
        self.connection = http.client.HTTPConnection("api.mediastack.com")

    def get_data(
        self, 
        date: str,
        keywords: List[str],
        limit: int,
        language: str,
        sort_by: str,
    ) -> List[dict]:
        """
        Parameters
        ----------
        date : str
            Date in the format `YYYY-MM-DD`.
        keywords : List[str]
            List of keywords to search for.
        limit : int
            Number of articles to return. Max is 100.
        language : str
            Language of the articles.
        sort : str
            How to sort the results. Can be `popularity` or `published_desc`.

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
                "keywords": ",".join(keywords),
                "date": date,
                "limit": limit,
                "languages": language,
                "sort": sort_by,
            }
        )
        self.connection.request("GET", f"/v1/news?{params}")
        response = self.connection.getresponse()
        bytes_data = response.read()
        return json.loads(bytes_data)["data"]
