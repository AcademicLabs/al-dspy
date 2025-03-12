from typing import Dict, List, Optional, Tuple, Any

import dspy
from dspy.primitives.prediction import Prediction
from dspy.retrieve.enhanced_document import (
    EnhancedDocument,
    DocumentReference,
    Citation,
    CitedText,
)


class VerifyCitations(dspy.Module):
    """
    A module for verifying the accuracy of citations in generated text.

    This module checks that each citation accurately represents the information
    in the referenced document at the specified location.
    """

    def __init__(self, document_store=None):
        """
        Initialize the module.

        Args:
            document_store: Optional dictionary of EnhancedDocuments, keyed by doc_id.
        """
        super().__init__()
        self.document_store = document_store or {}

        # Define the signature for verifying a single citation
        self.verify_citation = dspy.ChainOfThought(VerifyCitation)

    def add_document(self, document: EnhancedDocument) -> None:
        """Add a document to the document store."""
        self.document_store[document.doc_id] = document

    def get_document(self, doc_id: str) -> Optional[EnhancedDocument]:
        """Get a document from the document store by ID."""
        return self.document_store.get(doc_id)

    def forward(self, cited_text):
        """
        Verify the accuracy of citations in the text.

        Args:
            cited_text: CitedText object containing text and citations.

        Returns:
            Prediction object with verification results.
        """
        verification_results = []

        for citation in cited_text.citations:
            # Get the reference
            reference = citation.reference
            doc_id = reference.doc_id

            # Get the document
            document = self.get_document(doc_id)
            if not document:
                verification_results.append(
                    {
                        "citation_id": citation.citation_id,
                        "text": citation.text,
                        "verified": False,
                        "error": f"Document with ID {doc_id} not found",
                    }
                )
                continue

            # Get the referenced text
            referenced_text = document.get_snippet(
                reference.start_line, reference.end_line
            )

            # Verify the citation
            result = self.verify_citation(
                claim_text=citation.text,
                source_text=referenced_text,
                doc_id=doc_id,
                start_line=reference.start_line,
                end_line=reference.end_line,
            )

            # Add verification result
            verification_results.append(
                {
                    "citation_id": citation.citation_id,
                    "text": citation.text,
                    "verified": result.is_accurate,
                    "explanation": result.explanation,
                    "confidence": result.confidence,
                }
            )

        # Calculate overall accuracy
        verified_count = sum(
            1 for result in verification_results if result.get("verified", False)
        )
        accuracy = (
            verified_count / len(verification_results) if verification_results else 1.0
        )

        return Prediction(
            verification_results=verification_results,
            overall_accuracy=accuracy,
            is_verified=accuracy
            >= 0.8,  # Consider verified if 80% of citations are accurate
        )


class VerifyCitation(dspy.Signature):
    """Verify that a citation accurately represents the source text."""

    claim_text = dspy.InputField(desc="the text of the claim being cited")
    source_text = dspy.InputField(desc="the text from the source document")
    doc_id = dspy.InputField(desc="the document identifier")
    start_line = dspy.InputField(desc="the starting line number")
    end_line = dspy.InputField(desc="the ending line number")
    is_accurate = dspy.OutputField(
        desc="boolean indicating if the citation is accurate"
    )
    confidence = dspy.OutputField(desc="confidence score between 0 and 1")
    explanation = dspy.OutputField(desc="explanation of verification result")
