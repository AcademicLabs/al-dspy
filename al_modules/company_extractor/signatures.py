from typing import List, Optional

from pydantic import BaseModel, Field

from al_commons.webpage_helpers.text_extractors.text_extractors import (
    default_trafilatura_text_extractor,
)
from al_commons.webpage_helpers.webpage_downloaders.httpx_downloaders import (
    httpx_downloader,
)
from dspy import Signature, InputField, OutputField, ReAct
from dspy.retrieve.serper_rm import SerperRM, SerperSearchParameters, CollectedResult


class GeoLocation(BaseModel):
    city: Optional[str] = Field(None, description="The city of the given string")
    state: Optional[str] = Field(
        None,
        description="The state from the given string. Parse in alpha_2 code",
    )
    country_code: Optional[str] = Field(
        None,
        description="The country of the given string. When country_code is not present, disambiguate the country based on city and/or state. Parse in alpha_2 code",
    )


class Company(BaseModel):
    name: Optional[str] = Field(None, description="The name of the company.")
    geo_location: Optional[GeoLocation] = Field(
        None, description="The head-quarter location of the company."
    )
    website: Optional[str] = Field(None, description="The website URL of the company.")
    linkedin: Optional[str] = Field(
        None, description="The LinkedIn URL of the company."
    )
    # ceo: Optional[str] = Field(None, description="The name of the CEO of the company.")


class DocumentToCompanies(Signature):
    """Extract all relevant company names and additional information from a document.
    Use web search to ground the company names and scrape the webpages to extract additional information if necessary.
    company name and website are mandatory fields.
    """

    document: str = InputField(
        prefix="document",
        desc="The document containing the company names in its text.",
    )
    companies: Optional[List[Company]] = OutputField(
        desc="The list of companies extracted from the document and grounded via web search when necessary to retrieve the requested company data.",
        format=List[Company],
    )


rm = SerperRM(k=10, query_params=SerperSearchParameters())


def search(query_or_queries: str | List[str]) -> List[str]:
    """Search for companies based on the given query or list of queries to retrieve the necessary indormation."""
    results: List[CollectedResult] | [] = rm(query_or_queries)
    return [_format_result(result) for result in results]


def _format_result(result: CollectedResult) -> str:
    return f"Topic: {result.title}, Description: {result.description},URL: {result.url} Snippets: {result.snippets}"


def scrape(url: str) -> str:
    """Scrape the webpage at the given URL to extract company information from, if necessary."""
    webpage = httpx_downloader(url)
    text = default_trafilatura_text_extractor(webpage)
    return f"Scraped content from {url}:\n {text}"


company_extractor = ReAct(DocumentToCompanies, tools=[search, scrape], max_iters=20)
