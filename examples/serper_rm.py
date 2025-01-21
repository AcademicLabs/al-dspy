from functools import partial
from typing import List

from trafilatura.settings import Extractor

from al_commons.website_helpers.text_extractors import (
    TextExtractorFn,
    trafilatura_text_extractor,
)
from al_commons.website_helpers.text_splitters import (
    TextSplitterFn,
    recursive_character_text_splitter_fn,
)
from al_commons.website_helpers.webpage_downloaders import (
    WebPageDownloaderFn,
    httpx_downloader,
)
from al_commons.website_helpers.webpage_helper import HTMLHelper
from dspy.retrieve.serper_rm import SerperRM, SerperSearchParameters, CollectedResult

text_extractor: TextExtractorFn = partial(
    trafilatura_text_extractor,
    options=Extractor(tables=False, output_format="txt", comments=False),
)
text_splitter: TextSplitterFn = partial(
    recursive_character_text_splitter_fn, snippet_chunk_size=500, chunk_overlap=100
)
html_downloader: WebPageDownloaderFn = partial(httpx_downloader, timeout=10)

website_helper = HTMLHelper(
    text_extractor=text_extractor,
    text_splitter=text_splitter,
    html_downloader=html_downloader,
    min_char_count=150,
    max_thread_num=10,
)

query_params = SerperSearchParameters()
query_params.autocorrect = True
query_params.page = 1
query_params.num = 10

rm = SerperRM(k=3, query_params=query_params, website_helper=website_helper)


def serper_rm(topics: List[str]):
    results: List[CollectedResult] | [] = rm(topics)

    for result in results:
        print(f"Topic: {result.title}")
        print(f"Description: {result.description}")
        print(f"URL: {result.url}")
        print(f"Snippets: {result.snippets}")


if __name__ == "__main__":
    # topics = ["Python", "Data Science", "Machine Learning"]
    topics = ["Python"]
    serper_rm(topics)
