from typing import List, Optional

from dsp import Example
from pydantic import BaseModel, Field

import dspy
from al_commons.webpage_helpers.text_extractors.text_extractors import (
    default_trafilatura_text_extractor,
)
from al_commons.webpage_helpers.webpage_downloaders.httpx_downloaders import (
    httpx_downloader,
)
from dspy import Signature, InputField, OutputField, ReAct, Prediction
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
    ceo: Optional[str] = Field(None, description="The name of the CEO of the company.")


class DocumentToCompanies(Signature):
    """Extract all relevant company names and their website, Linkedin company url, head-quarter location (city, state, country), and CEO.
    Use web search to ground the information and scrape the webpages to extract and ground that information if necessary.
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


class Assess(dspy.Signature):
    """Assess the correctness of the extracted companies from the document.
    The gold example contains the correct company names and additional information that was found manually.
    for each correct key/value pair in the extracted company, the correctness is increased by 1.
    for each correct company name and website url, the correctness is increased by 2.
    Company names can of course slightly differ due to abbreviations or legal suffixes, this is acceptable.
    The correctness is then divided by the total number of key/value pairs to get the normalized final correctness score (float between 0 and 1).
    """

    gold = InputField(
        desc="The gold example containing the correct company names and additional information."
    )
    pred = InputField(
        desc="The prediction containing the extracted company names and additional information."
    )
    correctness: float = OutputField(
        desc="The correctness of the extracted companies from the document.",
        format=float,
    )


# metric for correctness
def metric(gold: Example, pred: Prediction, trace: bool = False):
    """Evaluate the example to extract companies from the document."""
    gold_companies = gold.companies
    pred_companies = pred.companies
    correctness = dspy.Predict(Assess)(
        gold=gold_companies, pred=pred_companies
    ).correctness
    if trace:
        return correctness > 0.9
    return correctness


company_extractor = ReAct(DocumentToCompanies, tools=[search], max_iters=20)
