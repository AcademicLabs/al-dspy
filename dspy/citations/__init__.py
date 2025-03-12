"""
DSPy Citations - A system for generating LLM text with precise citations and references.

This module provides tools for generating text with citations that reference specific
lines and positions in source documents.
"""

from dspy.retrieve.enhanced_document import (
    EnhancedDocument,
    DocumentReference,
    Citation,
    CitedText,
    LineInfo,
)
from dspy.retrieve.enhanced_retriever import EnhancedRetrieve
from dspy.modules.citation_generator import GenerateWithCitations
from dspy.modules.citation_verifier import VerifyCitations
from dspy.modules.profile_summarizer import ProfileSummarizer

__all__ = [
    "EnhancedDocument",
    "DocumentReference",
    "Citation",
    "CitedText",
    "LineInfo",
    "EnhancedRetrieve",
    "GenerateWithCitations",
    "VerifyCitations",
    "ProfileSummarizer",
]
