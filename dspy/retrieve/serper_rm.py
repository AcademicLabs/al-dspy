import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Union

import requests

from dspy import Retrieve

from typing import List, Optional
from pydantic import BaseModel, Field


class SearchType(str, Enum):
    search = "search"
    images = "images"
    videos = "videos"
    places = "places"
    maps = "maps"
    reviews = "reviews"
    news = "news"
    shopping = "shopping"
    image_search = "image_search"  # "Lens" can just be part of the label
    scholar = "scholar"
    patents = "patents"
    autocomplete = "autocomplete"
    webpage = "webpage"


class SerperSearchParameters(BaseModel):
    q: Optional[str] = Field(default=None, description="The search query string.")
    gl: Optional[str] = Field(
        default="US",
        description="The geolocation to use for the search (e.g., 'US', 'UK').",
    )
    hl: Optional[str] = Field(
        default="en",
        description="The language to use for the search (e.g., 'en', 'es').",
    )
    autocorrect: Optional[bool] = Field(
        default=True, description="Whether to automatically correct typos in the query."
    )
    page: Optional[int] = Field(
        default=1, description="The number of pages to retrieve."
    )
    num: Optional[int] = Field(
        default=10,
        description="The number of search results to return per page (must be multiple of 10).",
    )
    type: SearchType = Field(
        default=SearchType.search,
        description="The type of the search (e.g., search, images, videos, etc.).",
    )


class KnowledgeGraph(BaseModel):
    title: str
    type: str
    website: Optional[str] = None
    imageUrl: Optional[str] = None
    description: Optional[str] = None
    descriptionSource: Optional[str] = None
    descriptionLink: Optional[str] = None
    attributes: Optional[dict] = None


class Sitelink(BaseModel):
    title: str
    link: str


class Organic(BaseModel):
    title: str
    link: str
    snippet: str
    position: int
    sitelinks: Optional[List[Sitelink]] = None
    date: Optional[str] = None
    attributes: Optional[dict] = None


class PeopleAlsoAsk(BaseModel):
    question: Optional[str] = None
    snippet: Optional[str] = None
    title: Optional[str] = None
    link: Optional[str] = None


class RelatedSearch(BaseModel):
    query: str


class SerperSearchResult(BaseModel):
    searchParameters: SerperSearchParameters
    organic: List[Organic]
    peopleAlsoAsk: Optional[List[PeopleAlsoAsk]] = None
    relatedSearches: Optional[List[RelatedSearch]] = None
    knowledgeGraph: Optional[KnowledgeGraph] = None


@dataclass
class CollectedResult:
    snippets: List[str]
    title: str
    url: str
    description: str


class SerperRM(Retrieve):
    """Retrieve information from custom queries using Serper.dev."""

    results: Optional[List[SerperSearchResult]] = None
    usage: int = 0
    query_params: Optional[SerperSearchParameters] = None
    serper_search_api_key: Optional[str] = None
    base_url: Optional[str] = None

    def __init__(
        self,
        k=3,
        query_params: Optional[SerperSearchParameters] = None,
    ):
        """
        Args:
            query_params (dict or list of dict): parameters in dictionary or list of dictionaries that has a max size of 100 that will be used to query.
                Commonly used fields are as follows (see more information in https://serper.dev/playground):
                    q str: query that will be used with google search
                    type str: type that will be used for browsing google. Types are search, images, video, maps, places, etc.
                    gl str: Country that will be focused on for the search
                    location str: Country where the search will originate from. All locates can be found here: https://api.serper.dev/locations.
                    autocorrect bool: Enable autocorrect on the queries while searching, if query is misspelled, will be updated.
                    results int: Max number of results per page.
                    page int: Max number of pages per call.
                    tbs str: date time range, automatically set to any time by default.
                    qdr:h str: Date time range for the past hour.
                    qdr:d str: Date time range for the past 24 hours.
                    qdr:w str: Date time range for past week.
                    qdr:m str: Date time range for past month.
                    qdr:y str: Date time range for past year.
        """
        super().__init__(k=k)
        self.usage = 0

        if query_params is None:
            self.query_params = SerperSearchParameters(num=k, autocorrect=True, page=1)
        else:
            self.query_params = query_params
            self.query_params.num = k
        if not os.environ.get("SERPER_API_KEY"):
            raise RuntimeError(
                "You must supply a serper_search_api_key param or set environment variable SERPER_API_KEY"
            )
        self.serper_search_api_key = os.environ["SERPER_API_KEY"]
        self.base_url = "https://google.serper.dev"

    def _serper_runner(
        self, query_params: SerperSearchParameters
    ) -> SerperSearchResult:
        search_url = f"{self.base_url}/{query_params.type.value}"

        headers = {
            "X-API-KEY": self.serper_search_api_key,
            "Content-Type": "application/json",
        }

        response = requests.request(
            "POST", search_url, headers=headers, data=query_params.model_dump_json()
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Error occurred while running the search process. Error: {response.reason}, Status code: {response.status_code}"
            )

        try:
            search_result = SerperSearchResult.model_validate_json(response.text)
        except Exception as e:
            raise RuntimeError(
                f"Error occurred while parsing the serper response. Error is {e}"
            )

        return search_result

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0
        return {"SerperRM": usage}

    # noinspection PyMethodOverriding
    def forward(
        self,
        query_or_queries: str | List[str],
        exclude_urls: Optional[List[str]] = None,
    ) -> List[CollectedResult] | []:
        """
        Calls the API and searches for the provided query or queries.

        Args:
            self: The SerperRM instance.
            query_or_queries (Union[str, List[str]]): The query or list of queries to search for.
            exclude_urls (List[str]): Dummy parameter to match the interface. Does not have any effect.

        Returns:
            List[dict]: A list of dictionaries, each containing the keys 'description', 'snippets' (list of strings), 'title', and 'url'.
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )

        self.usage += len(queries)
        self.results = []
        for query in queries:
            if query == "Queries:":
                continue
            query_params: SerperSearchParameters = self.query_params

            # All available parameters can be found in the playground: https://serper.dev/playground
            # Sets the value for query to be the query that is being parsed.
            query_params.q = query

            result = self._serper_runner(query_params)
            self.results.append(result)

        # Array of dictionaries that will be used by Storm to create the jsons
        collected_results: List[CollectedResult] = []

        for result in self.results:
            # An array of dictionaries that contains the snippets, title of the document and url that will be used.
            organic_results = result.organic
            knowledge_graph = result.knowledgeGraph
            for organic in organic_results:
                snippets = [organic.snippet]
                collected_results.append(
                    CollectedResult(
                        snippets=snippets,
                        title=organic.title,
                        url=organic.link,
                        description=(
                            knowledge_graph.description
                            if knowledge_graph is not None
                            else ""
                        ),
                    )
                )
        return collected_results
