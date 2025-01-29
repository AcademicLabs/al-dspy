import os
from enum import Enum
from typing import Callable, Union, List, Optional, Any

import requests

import dspy
from al_commons.log.logger import LOGGER
from dspy.retrieve.common import CollectedResult


class SearchType(str, Enum):
    search = "search"
    images = "images"
    news = "news"


class BingSearch(dspy.Retrieve):
    def __init__(
        self,
        bing_search_api_key=None,
        k=3,
        is_valid_source: Optional[Callable] = None,
        mkt="en-US",
        language="en",
        search_type: Optional[SearchType] = SearchType.search,
        **kwargs,
    ):
        """
        Params:
            mkt, language, **kwargs: Bing search API parameters.
            - Reference:
                - https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/query-parameters
                - https://learn.microsoft.com/en-us/bing/search-apis/bing-image-search/reference/query-parameters
                - https://learn.microsoft.com/en-us/bing/search-apis/bing-news-search/reference/query-parameters
        """
        super().__init__(k=k)
        if not bing_search_api_key and not os.environ.get("BING_SEARCH_API_KEY"):
            raise RuntimeError(
                "You must supply bing_search_subscription_key or set environment variable BING_SEARCH_API_KEY"
            )
        elif bing_search_api_key:
            self.bing_api_key = bing_search_api_key
        else:
            self.bing_api_key = os.environ["BING_SEARCH_API_KEY"]
        match search_type:
            case SearchType.search:
                self.endpoint = "https://api.bing.microsoft.com/v7.0/search"
            case SearchType.news:
                self.endpoint = "https://api.bing.microsoft.com/v7.0/news/search"
            case SearchType.images:
                self.endpoint = "https://api.bing.microsoft.com/v7.0/images/search"
        self.search_type = search_type
        self.params = {"mkt": mkt, "setLang": language, "count": k, **kwargs}
        self.usage = 0

        # If not None, is_valid_source shall be a function that takes a URL and returns a boolean.
        if is_valid_source:
            self.is_valid_source = is_valid_source
        else:
            self.is_valid_source = lambda x: True

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {"BingSearch": usage}

    # noinspection PyMethodOverriding
    def forward(
        self, query_or_queries: Union[str, List[str]], exclude_urls: List[str] = None
    ) -> List[CollectedResult] | []:
        """Search with Bing for self.k top passages for query or queries

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): A list of urls to exclude from the search results.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)

        collected_results: List[CollectedResult] = []

        headers = {"Ocp-Apim-Subscription-Key": self.bing_api_key}

        for query in queries:
            try:
                results = requests.get(
                    self.endpoint, headers=headers, params={**self.params, "q": query}
                ).json()

                collected_results.extend(self.collect_results(results))

            except Exception as e:
                LOGGER.error(f"Error occurs when searching query {query}: {e}")

        return collected_results


    def collect_results(self, json_results: Any) -> List[CollectedResult]:
        match self.search_type:
            case SearchType.search:
                results = json_results["webPages"]["value"]
                return [
                    CollectedResult(
                        url=d["url"],
                        title=d["name"],
                        description=",".join(
                            [tag.get("content") for tag in d.get("searchTags", [])]
                        ),
                        snippets=[d["snippet"]],
                        pub_date=d["datePublished"],  # YYYY-MM-DDTHH:MM:SS
                    )
                    for d in results
                ]
            case SearchType.news:
                results = json_results["value"]
                return [
                    CollectedResult(
                        url=d["url"],
                        title=d["name"],
                        description=d["description"],
                        snippets=[d["description"]],
                        pub_date=d["datePublished"],  # YYYY-MM-DDTHH:MM:SS
                    )
                    for d in results
                ]

            case SearchType.images:
                results = json_results["value"]
                return [
                    CollectedResult(
                        url=d["contentUrl"],
                        title=d["name"],
                        description=d["description"],
                        snippets=[d["description"]],
                        pub_date=d["datePublished"],  # YYYY-MM-DDTHH:MM:SS
                    )
                    for d in results
                ]
