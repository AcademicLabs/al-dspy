from typing import Dict, List, Optional, Any, Union
import uuid

import dspy
from dspy.primitives.prediction import Prediction
from dspy.retrieve.enhanced_document import (
    EnhancedDocument,
    DocumentReference,
    Citation,
    CitedText,
)
from dspy.modules.citation_generator import GenerateWithCitations
from dspy.modules.citation_verifier import VerifyCitations


class ProfileSummarizer(dspy.Module):
    """
    A module for generating summaries of user profiles with citations.

    This module retrieves relevant documents about a user, generates a summary
    with citations to specific parts of the source documents, and verifies the
    accuracy of the citations.
    """

    def __init__(self, retriever, citation_style="brackets", verify_citations=True):
        """
        Initialize the module.

        Args:
            retriever: The retriever to use for finding documents about the user.
            citation_style: Style to format citations (brackets, parentheses, or hyperlink).
            verify_citations: Whether to verify the accuracy of citations.
        """
        super().__init__()
        self.retriever = retriever
        self.citation_style = citation_style
        self.verify_citations = verify_citations

        # Define the signature for generating a user profile summary
        self.generate_summary = GenerateWithCitations(
            retriever=retriever, citation_style=citation_style
        )

        # Define the signature for generating questions about a user
        self.generate_questions = dspy.ChainOfThought(GenerateUserQuestions)

        # Define the citation verifier
        if verify_citations:
            self.citation_verifier = VerifyCitations(
                document_store=retriever.document_store
            )

    def forward(self, user_id=None, user_name=None, user_info=None):
        """
        Generate a summary of a user profile with citations.

        Args:
            user_id: ID of the user to summarize.
            user_name: Name of the user to summarize.
            user_info: Additional information about the user.

        Returns:
            Prediction object with summary and citation information.
        """
        # Step 1: Formulate a query about the user
        if user_id and user_name:
            query = f"Information about user {user_name} (ID: {user_id})"
        elif user_id:
            query = f"Information about user with ID {user_id}"
        elif user_name:
            query = f"Information about user named {user_name}"
        else:
            query = "User profile information"

        if user_info:
            query += f" {user_info}"

        # Step 2: Retrieve documents about the user
        retrieval_result = self.retriever(query)
        documents = retrieval_result.enhanced_docs

        if not documents:
            return Prediction(
                summary="No information found for this user.",
                error="No documents retrieved",
            )

        # Step 3: Generate questions about the user
        questions_result = self.generate_questions(
            user_id=user_id, user_name=user_name, user_info=user_info
        )
        questions = questions_result.questions

        # Step 4: Generate a summary for each question
        summaries = []
        combined_text = ""

        for question in questions:
            # Generate a summary with citations for the question
            summary_result = self.generate_summary(context=documents, question=question)

            # Add to the list of summaries
            summaries.append(
                {"question": question, "summary": summary_result.text_with_citations}
            )

            # Add to the combined text
            combined_text += (
                f"\n\n## {question}\n\n{summary_result.text_with_citations}"
            )

            # Verify citations if enabled
            if self.verify_citations:
                verification_result = self.citation_verifier(summary_result.cited_text)
                summaries[-1]["verification"] = {
                    "overall_accuracy": verification_result.overall_accuracy,
                    "is_verified": verification_result.is_verified,
                    "results": verification_result.verification_results,
                }

        # Generate a comprehensive profile summary
        comprehensive_summary = (
            f"# Profile Summary for {user_name or 'User'}\n\n{combined_text}"
        )

        # Step 5: Return the results
        return Prediction(
            user_id=user_id,
            user_name=user_name,
            comprehensive_summary=comprehensive_summary,
            summaries=summaries,
            retrieval_result=retrieval_result,
        )


class GenerateUserQuestions(dspy.Signature):
    """Generate questions to ask about a user profile."""

    user_id = dspy.InputField(desc="ID of the user", default=None)
    user_name = dspy.InputField(desc="Name of the user", default=None)
    user_info = dspy.InputField(
        desc="Additional information about the user", default=None
    )
    questions = dspy.OutputField(desc="list of questions to ask about the user profile")
