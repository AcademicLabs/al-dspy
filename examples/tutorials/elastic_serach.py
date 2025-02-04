import os
from typing import List

from al_indexer.es.es_client import ESClient
from dspy import (
    InputField,
    OutputField,
    ReAct,
    LM,
    configure,
    Signature,
    inspect_history,
)
from dspy.retrieve.elastic_search_rm import ElasticSearchRM


class TopicSnippets(Signature):
    """
    Retrieve relevant snippets for a given topic.
    Use the search tool to retrieve the data from which the snippets should be extracted.
    """

    topic: str = InputField(
        desc="The topic of interest.",
        format=str,
    )

    snippets: List[str] = OutputField(
        desc="The snippets retrieved for the topic.",
        format=List[str],
    )


es_client = ESClient()

rm = ElasticSearchRM(es_client=es_client, search_parameters=None, k=5)


def search(keywords: List[str]) -> str:
    """Retrieve data for the given topic."""
    search_results = rm(keywords)
    return str(search_results)


snippet_extractor = ReAct(TopicSnippets, tools=[search], max_iters=5)

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    lm = LM("openai/gpt-4o", api_key=api_key)
    configure(lm=lm)

    snippets = snippet_extractor(topic="Trikafta", index="publications").snippets
    print(snippets)
    history = inspect_history(n=1)
    print(history)
