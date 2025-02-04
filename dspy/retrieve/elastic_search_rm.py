from typing import Dict, Any, Optional, List

from elasticsearch_dsl.query import MultiMatch
from pydantic import BaseModel, Field

from dspy import Retrieve

from elasticsearch_dsl import Search, MultiSearch
from elasticsearch_dsl.response import Hit, Response


class ESSearchResult(BaseModel):
    index: str
    id: str
    score: float
    source: Dict[str, Any]


class ESSearchParameters(BaseModel):
    index: str
    query: str | List[str]
    size: Optional[int] = 10
    fields: Optional[List[str]] = Field(default_factory=lambda: ["title", "abstract"])


class ElasticSearchRM(Retrieve):
    """Retrieve data from one or more ElasticSearch indexes."""

    query_parameters: ESSearchParameters
    results = Optional[List[ESSearchResult]]

    def __init__(self, es_client, search_parameters: Optional[ESSearchParameters], k=3):
        super().__init__(k=k)
        self.es_client = es_client
        self.searchParameters = search_parameters
        if search_parameters is None:
            self.search_parameters = ESSearchParameters(index="", query="", size=k)
        else:
            self.search_parameters = search_parameters

    def _es_search(self, search_parameters: ESSearchParameters) -> Response:
        ms = MultiSearch(index=search_parameters.index)
        query = " ".join(search_parameters.query)
        s = Search(using=self.es_client, index=search_parameters.index)
        q = MultiMatch(query=query, fields=["title", "abstract"])
        s = s.query(q)
        ms = ms.add(s)
        response: Response = ms.execute()
        return response[0]

    @staticmethod
    def _process_results(result: Response) -> str:

        search_results = f"Results: {result}"

        return search_results

    # noinspection PyMethodOverriding
    def forward(
        self, query_or_queries: str | List[str], index: Optional[str] = None
    ) -> str:
        query_params: ESSearchParameters = self.search_parameters

        query_params.query = query_or_queries
        if index:
            query_params.index = index

        results = self._es_search(query_params)

        return self._process_results(results)
