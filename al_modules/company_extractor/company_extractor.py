import os

from al_commons.webpage_helpers.text_extractors.text_extractors import (
    default_trafilatura_text_extractor,
)
from al_modules.company_extractor.signatures import company_extractor
from dspy import LM, configure, inspect_history

title = "News Title: Idiopathic Inflammatory Myositis Market Predicted to Show Positive Growth at a Tremendous CAGR of 21.1% by 2034 | DelveInsight"

if __name__ == "__main__":
    with open("example_doc.html", "r") as file:
        raw_body = file.read()
        body = default_trafilatura_text_extractor(raw_body)

    example_doc: str = title + "\n News Body:" + body

    api_key = os.getenv("OPENAI_API_KEY")
    lm = LM("openai/gpt-4o", api_key=api_key)
    configure(lm=lm)

    try:
        companies = company_extractor(document=example_doc).companies
    except Exception as e:
        print(f"Error extracting companies: {str(e)}")
    for company in companies:
        print(company)

    print("\n----------\n")
    history = inspect_history(n=1)
    print(history)
