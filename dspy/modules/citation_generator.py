import uuid
from typing import Dict, List, Optional, Any

import dspy
from dspy.primitives.prediction import Prediction
from dspy.retrieve.enhanced_document import (
    EnhancedDocument,
    DocumentReference,
    Citation,
    CitedText,
)


class GenerateWithCitations(dspy.Module):
    """
    A module for generating text with citations to source documents.

    This module extends the basic generation capabilities of DSPy to include
    citations to source documents with precise line and character positions.
    """

    def __init__(self, retriever=None, citation_style="brackets", max_attempts=3):
        """
        Initialize the module.

        Args:
            retriever: The retriever to use for finding supporting evidence.
            citation_style: Style to format citations (brackets, parentheses, or hyperlink).
            max_attempts: Maximum number of attempts to generate text with proper citations.
        """
        super().__init__()
        self.retriever = retriever
        self.citation_style = citation_style
        self.max_attempts = max_attempts

        # Define the signature for the LLM to generate text with citation placeholders
        self.generate_text = dspy.ChainOfThought(GenerateTextWithPlaceholders)

        # Define the signature to identify claims in the generated text
        self.identify_claims = dspy.ChainOfThought(IdentifyClaims)

        # Define the signature to find supporting evidence for each claim
        self.find_evidence = dspy.ChainOfThought(FindEvidence)

    def forward(self, context, question):
        """
        Generate text with citations in response to a question.

        Args:
            context: The context information.
            question: The question to answer.

        Returns:
            Prediction object with cited text.
        """
        # Step 1: Generate initial text with citation placeholders
        generation_result = self.generate_text(context=context, question=question)
        text = generation_result.text

        # Step 2: Identify claims in the generated text
        claims_result = self.identify_claims(text=text)
        claims = claims_result.claims

        # Initialize cited text
        cited_text = CitedText(text=text)

        # Step 3: For each claim, find supporting evidence and add citations
        for i, claim in enumerate(claims):
            # Find supporting evidence for the claim
            if self.retriever:
                evidence_list = self.retriever.find_supporting_evidence(claim, top_k=1)
                if evidence_list:
                    evidence = evidence_list[0]

                    # Create a citation
                    citation_id = f"cite_{i+1}"
                    citation = self.retriever.create_citation(
                        reference=evidence,
                        text=claim,
                        citation_id=citation_id,
                        citation_style=self.citation_style,
                    )

                    # Add the citation to the cited text
                    cited_text.add_citation(citation)
            else:
                # If no retriever, use the find_evidence signature to manually identify evidence
                evidence_result = self.find_evidence(context=context, claim=claim)
                doc_id = evidence_result.doc_id
                start_line = evidence_result.start_line
                end_line = evidence_result.end_line

                # Create a reference and citation
                reference = DocumentReference(
                    doc_id=doc_id,
                    start_line=start_line,
                    end_line=end_line,
                    text=evidence_result.text,
                )

                citation = Citation(
                    reference=reference,
                    text=claim,
                    citation_id=f"cite_{i+1}",
                    citation_style=self.citation_style,
                )

                # Add the citation to the cited text
                cited_text.add_citation(citation)

        # Generate the final text with formatted citations
        text_with_citations = cited_text.get_text_with_citations()

        # Generate a references section
        references_section = cited_text.get_references_section()

        # Return the result
        return Prediction(
            text=text,
            text_with_citations=text_with_citations,
            references=references_section,
            cited_text=cited_text,
        )


class GenerateTextWithPlaceholders(dspy.Signature):
    """Generate text with placeholders for citations."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    text = dspy.OutputField(
        desc="comprehensive answer with factual statements that can be cited"
    )


class IdentifyClaims(dspy.Signature):
    """Identify factual claims in the text that should be cited."""

    text = dspy.InputField()
    claims = dspy.OutputField(desc="list of factual claims that should be cited")


class FindEvidence(dspy.Signature):
    """Find supporting evidence for a claim in the context."""

    context = dspy.InputField(desc="may contain relevant facts")
    claim = dspy.InputField(desc="a factual claim that needs citation")
    doc_id = dspy.OutputField(desc="identifier of the document containing evidence")
    start_line = dspy.OutputField(desc="starting line number of the evidence")
    end_line = dspy.OutputField(desc="ending line number of the evidence")
    text = dspy.OutputField(desc="the text of the evidence")
