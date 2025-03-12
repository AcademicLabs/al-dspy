import dspy
from dspy.retrieve.enhanced_document import EnhancedDocument
from dspy.retrieve.enhanced_retriever import EnhancedRetrieve
from dspy.modules.profile_summarizer import ProfileSummarizer


def main():
    """
    Example demonstrating how to use the citation system with a user profile summarization task.
    """
    print("DSPy User Profile Summarization with Citations Example")
    print("=" * 60)

    # Step 1: Set up a basic DSPy configuration
    llm = dspy.OpenAI(model="gpt-3.5-turbo")
    dspy.settings.configure(lm=llm)

    # Step 2: Create some sample user profile documents
    user_docs = [
        EnhancedDocument(
            doc_id="user123_basic",
            content="User ID: 123\nName: John Smith\nAge: 35\nLocation: New York, NY\nOccupation: Software Engineer\nJoined: 2020-05-15",
            source_name="User Basic Information",
            source_url="https://example.com/users/123",
        ),
        EnhancedDocument(
            doc_id="user123_activity",
            content="User ID: 123\nActivity Summary:\n- Posted 150 times in the past year\n- Visited the platform on average 3 times per week\n- Most active in the Technology and Programming forums\n- Received 520 upvotes on contributions\n- Highest rated post: 'Guide to Neural Networks for Beginners'",
            source_name="User Activity Log",
            source_url="https://example.com/users/123/activity",
        ),
        EnhancedDocument(
            doc_id="user123_preferences",
            content="User ID: 123\nPreferences:\n- Prefers dark mode UI\n- Email notification frequency: Weekly\n- Interested in: Machine Learning, Python, Web Development, Data Science\n- Follows 25 other users\n- Subscribed to 8 premium courses",
            source_name="User Preferences",
            source_url="https://example.com/users/123/preferences",
        ),
        EnhancedDocument(
            doc_id="user123_contributions",
            content="User ID: 123\nContributions:\nJohn has authored the following popular content:\n1. 'Getting Started with TensorFlow' - Tutorial with 15,000 views\n2. 'Python Tips and Tricks' - Article with 8,500 views\n3. 'Web Development Best Practices' - Forum post with 230 replies\n4. 'Introduction to DSPy' - Recent tutorial with growing popularity\n\nJohn is a Level 4 contributor and has earned the 'Python Expert' and 'Helpful Mentor' badges.",
            source_name="User Contributions",
            source_url="https://example.com/users/123/contributions",
        ),
    ]

    # Step 3: Set up a simple retriever that will return all documents
    class SimpleRetriever(dspy.Retrieve):
        def __init__(self, documents):
            super().__init__(k=len(documents))
            self.documents = documents

        def forward(self, query):
            # Simple retriever that returns all documents regardless of query
            return dspy.Prediction(
                docs=[doc.content for doc in self.documents],
                doc_ids=[doc.doc_id for doc in self.documents],
            )

    base_retriever = SimpleRetriever(user_docs)

    # Step 4: Create an enhanced retriever with our document store
    document_store = {doc.doc_id: doc for doc in user_docs}
    enhanced_retriever = EnhancedRetrieve(base_retriever, document_store)

    # Step 5: Create a profile summarizer
    profile_summarizer = ProfileSummarizer(
        retriever=enhanced_retriever, citation_style="brackets", verify_citations=True
    )

    # Step 6: Generate a profile summary
    result = profile_summarizer(user_id=123, user_name="John Smith")

    # Step 7: Display the results
    print("\nGenerated Profile Summary with Citations:")
    print("=" * 60)
    print(result.comprehensive_summary)

    print("\nVerification Information:")
    print("=" * 60)
    for summary in result.summaries:
        if "verification" in summary:
            verification = summary["verification"]
            print(f"Question: {summary['question']}")
            print(f"Overall accuracy: {verification['overall_accuracy']:.2f}")
            print(f"Is verified: {verification['is_verified']}")
            print("-" * 40)


if __name__ == "__main__":
    main()
