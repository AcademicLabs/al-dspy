import uuid
from typing import Any, Dict, List, Optional, Union

import dspy
from dspy.primitives.prediction import Prediction
from dspy.retrieve.enhanced_document import (
    EnhancedDocument,
    DocumentReference,
    Citation,
    CitedText,
)


class EnhancedRetrieve(dspy.Retrieve):
    """
    Enhanced retriever that preserves line information and supports citations.

    This retriever extends the standard DSPy Retrieve module to work with EnhancedDocuments,
    allowing for precise citations with line and character positions.
    """

    def __init__(
        self,
        base_retriever: dspy.Retrieve,
        document_store: Dict[str, EnhancedDocument] = None,
    ):
        """
        Initialize the enhanced retriever.

        Args:
            base_retriever: The base DSPy retriever to use for retrieving documents.
            document_store: Optional dictionary of EnhancedDocuments, keyed by doc_id.
        """
        super().__init__(k=base_retriever.k)
        self.base_retriever = base_retriever
        self.document_store = document_store or {}

    def add_document(self, document: EnhancedDocument) -> None:
        """Add a document to the document store."""
        self.document_store[document.doc_id] = document

    def add_documents(self, documents: List[EnhancedDocument]) -> None:
        """Add multiple documents to the document store."""
        for document in documents:
            self.add_document(document)

    def get_document(self, doc_id: str) -> Optional[EnhancedDocument]:
        """Get a document from the document store by ID."""
        return self.document_store.get(doc_id)

    def forward(self, query: str) -> Prediction:
        """
        Retrieve documents relevant to the query.

        This method uses the base retriever to find relevant documents, then
        enhances the results with line information if the documents are in the document store.

        Args:
            query: The query to search for.

        Returns:
            Prediction object with enhanced documents.
        """
        # Use the base retriever to get results
        base_results = self.base_retriever(query)

        # Extract document IDs from results
        doc_ids = getattr(base_results, "doc_ids", [])
        documents = getattr(base_results, "docs", [])

        # Create enhanced documents if they don't exist
        enhanced_docs = []
        for i, doc in enumerate(documents):
            # Get doc_id if available, otherwise generate one
            doc_id = doc_ids[i] if i < len(doc_ids) else f"doc_{uuid.uuid4().hex[:8]}"

            # Check if we already have this document in our store
            if doc_id in self.document_store:
                enhanced_docs.append(self.document_store[doc_id])
            else:
                # Create a new enhanced document
                enhanced_doc = EnhancedDocument(
                    doc_id=doc_id, content=doc, metadata={"original_position": i}
                )
                self.document_store[doc_id] = enhanced_doc
                enhanced_docs.append(enhanced_doc)

        # Return results with enhanced documents
        result = Prediction(
            docs=documents, doc_ids=doc_ids, enhanced_docs=enhanced_docs
        )

        # Copy any other fields from the base result
        for key, value in base_results.items():
            if key not in result:
                result[key] = value

        return result

    def create_reference(
        self,
        doc_id: str,
        start_line: int,
        end_line: int,
        start_char: Optional[int] = None,
        end_char: Optional[int] = None,
    ) -> DocumentReference:
        """
        Create a reference to a specific part of a document.

        Args:
            doc_id: ID of the document.
            start_line: Starting line number.
            end_line: Ending line number.
            start_char: Optional starting character position.
            end_char: Optional ending character position.

        Returns:
            DocumentReference object.
        """
        document = self.get_document(doc_id)
        if not document:
            raise ValueError(f"Document with ID {doc_id} not found in document store")

        # Get the text for the reference
        text = document.get_snippet(start_line, end_line)

        return DocumentReference(
            doc_id=doc_id,
            start_line=start_line,
            end_line=end_line,
            start_char=start_char,
            end_char=end_char,
            text=text,
            source_name=document.source_name,
            source_url=document.source_url,
        )

    def create_citation(
        self,
        reference: DocumentReference,
        text: str,
        citation_id: Optional[str] = None,
        citation_style: str = "brackets",
    ) -> Citation:
        """
        Create a citation for a specific reference.

        Args:
            reference: The document reference to cite.
            text: The text being cited.
            citation_id: Optional unique identifier for the citation.
            citation_style: Style to format the citation.

        Returns:
            Citation object.
        """
        if citation_id is None:
            citation_id = f"cite_{uuid.uuid4().hex[:8]}"

        return Citation(
            reference=reference,
            text=text,
            citation_id=citation_id,
            citation_style=citation_style,
        )

    def find_supporting_evidence(
        self, claim: str, top_k: int = 3
    ) -> List[DocumentReference]:
        """
        Find supporting evidence for a claim in the retrieved documents.

        Args:
            claim: The claim to find evidence for.
            top_k: Maximum number of evidence pieces to return.

        Returns:
            List of DocumentReference objects supporting the claim.
        """
        claim = claim.strip()
        evidence = []

        # Search through all documents in the store
        for doc_id, document in self.document_store.items():
            # Use semantic search or simple text matching
            # Here we're using a simple string search as an example
            positions = document.find_text_position(claim)

            # If exact match not found, try to find sentences that might support the claim
            if not positions:
                lines = document._lines
                for i, line in enumerate(lines):
                    # Very simple heuristic - look for keyword overlap
                    # In a real implementation, you might use semantic similarity
                    claim_words = set(claim.lower().split())
                    line_words = set(line.text.lower().split())
                    overlap = (
                        len(claim_words.intersection(line_words)) / len(claim_words)
                        if claim_words
                        else 0
                    )

                    if overlap > 0.3:  # arbitrary threshold
                        # Found a potential match
                        start_line = max(1, i)
                        end_line = min(
                            len(lines), i + 3
                        )  # Include a few lines as context

                        evidence.append(
                            self.create_reference(
                                doc_id=doc_id, start_line=start_line, end_line=end_line
                            )
                        )
            else:
                # For each position found, create a reference
                for start_line, start_char, end_line, end_char in positions:
                    evidence.append(
                        self.create_reference(
                            doc_id=doc_id,
                            start_line=start_line,
                            end_line=end_line,
                            start_char=start_char,
                            end_char=end_char,
                        )
                    )

            # Limit to top_k pieces of evidence
            if len(evidence) >= top_k:
                break

        return evidence[:top_k]
