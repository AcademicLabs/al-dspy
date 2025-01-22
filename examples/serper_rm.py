from typing import List
from dspy.retrieve.serper_rm import SerperRM, SerperSearchParameters, CollectedResult

rm = SerperRM(k=10, query_params=SerperSearchParameters())


def serper_rm_example(topics: List[str]):
    results: List[CollectedResult] | [] = rm(topics)

    for result in results:
        print(f"Topic: {result.title}")
        print(f"Description: {result.description}")
        print(f"URL: {result.url}")
        print(f"Snippets: {result.snippets}")


if __name__ == "__main__":
    # topics = ["Python", "Data Science", "Machine Learning"]
    topic_list = ["Python"]
    serper_rm_example(topic_list)
