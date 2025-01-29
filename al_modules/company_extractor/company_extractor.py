import json
import os

from dspy.evaluate import Evaluate

from al_commons.webpage_helpers.text_extractors.text_extractors import (
    default_trafilatura_text_extractor,
)
from al_modules.company_extractor.signatures import company_extractor, metric
from dspy import LM, configure, Example

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    lm = LM("openai/gpt-4o", api_key=api_key)
    configure(lm=lm)

    json_path = "fixtures/training.json"
    training_set = []
    with open(json_path, "r") as file:
        json_data = json.load(file)
        for obj in json_data:
            slug = obj["slug"]
            _title = obj["title"]
            _companies = obj["companies"]
            with open(f"fixtures/{slug}.html", "r") as file:
                raw_body = file.read()
                _body = default_trafilatura_text_extractor(raw_body)

            sample_document = "News Title:" + _title + "\n News Body:" + _body
            training_set.append(
                Example(document=sample_document, companies=_companies).with_inputs(
                    "document"
                )
            )

    evaluator = Evaluate(
        devset=training_set, num_threads=8, display_progress=True, display_table=20
    )
    overall_score, result_triples, individual_scores = evaluator(
        company_extractor, metric=metric, return_all_scores=True, return_outputs=True
    )
    print(f"Overall Score: {overall_score}")
    print(f"Individual Scores: {individual_scores}")
    print(f"Result Triples: {result_triples}")

    # try:
    #     companies = company_extractor(document=example_doc).companies
    # except Exception as e:
    #     print(f"Error extracting companies: {str(e)}")
    # for company in companies:
    #     print(company)
    #
    # print("\n----------\n")
    # history = inspect_history(n=1)
    # print(history)

    # config = dict(metric_threshold=0.8)
    #
    # optimizer = BootstrapFewShot(metric=metric, **config)
    # optimized_program = optimizer.compile(company_extractor, trainset=training_set)
    #
    # optimized_program.save("optimized_program.json")
    #
    # overall_score, result_triples, individual_scores = evaluator(
    #     company_extractor, metric=metric, return_all_scores=True, return_outputs=True
    # )
    # print(f"Overall Score: {overall_score}")
    # print(f"Individual Scores: {individual_scores}")
    # print(f"Result Triples: {result_triples}")

    # config = dict(
    #     metric_threshold=0.7,
    #     auto="light",
    #     max_bootstrapped_demos=0,
    #     max_labeled_demos=0,
    #     num_threads=6,
    # )
    #
    # optimizer = MIPROv2(metric=metric, **config)
    #
    # optimized_program = optimizer.compile(
    #     company_extractor, trainset=training_set, minibatch=True
    # )

    # optimized_program.save("optimized_program_MIPROV2.json")
