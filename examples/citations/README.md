# DSPy Citation System

This extension for DSPy provides a comprehensive citation and reference system for LLM tasks. It allows your DSPy modules to generate text with precise citations that reference specific lines and positions in source documents.

## Features

- **Line-specific citations**: Citations can reference specific line numbers and character positions in source documents
- **Multiple citation styles**: Support for different citation styles (brackets, parentheses, hyperlinks)
- **Citation verification**: Built-in tools to verify that citations accurately represent the source material
- **Interactive references**: Citations can be rendered as interactive links in supported environments
- **Document structure preservation**: Enhanced document structure that preserves line information
- **Flexible API**: Easy to integrate with existing DSPy modules and workflows

## Core Components

### Enhanced Document Structure

- `EnhancedDocument`: Stores document content with line information
- `DocumentReference`: References a specific part of a document (lines, characters)
- `Citation`: Links text to a document reference
- `CitedText`: Text with citations

### Retrieval System

- `EnhancedRetrieve`: Extends DSPy's Retrieve module to work with enhanced documents
- Support for finding supporting evidence for claims

### Generation and Verification

- `GenerateWithCitations`: Generates text with citations to source documents
- `VerifyCitations`: Verifies that citations accurately represent the source material

### Application-Specific Modules

- `ProfileSummarizer`: Generates summaries of user profiles with citations

## Examples

### Basic Example

```python
import dspy
from dspy.retrieve.enhanced_document import EnhancedDocument
from dspy.retrieve.enhanced_retriever import EnhancedRetrieve
from dspy.modules.citation_generator import GenerateWithCitations

# Create some sample documents
docs = [
    EnhancedDocument(
        doc_id="doc1",
        content="This is the first document.\nIt contains important information.",
        source_name="Source 1",
        source_url="https://example.com/doc1"
    ),
    EnhancedDocument(
        doc_id="doc2",
        content="This is the second document.\nIt contains additional details.",
        source_name="Source 2",
        source_url="https://example.com/doc2"
    )
]

# Create an enhanced retriever
document_store = {doc.doc_id: doc for doc in docs}
retriever = EnhancedRetrieve(base_retriever, document_store)

# Create a citation generator
citation_generator = GenerateWithCitations(retriever=retriever)

# Generate text with citations
result = citation_generator(
    context=docs,
    question="What information is in the documents?"
)

# Display the result
print(result.text_with_citations)
```

### User Profile Example

See the complete example in `user_profile_example.py`

## Integration with Existing DSPy Code

To integrate the citation system with your existing DSPy code:

1. Enhance your documents with the `EnhancedDocument` class
2. Wrap your retriever with `EnhancedRetrieve`
3. Use the provided generation modules or create custom ones

## Citation Styles

The system supports multiple citation styles:

- **Brackets**: `This is a citation [doc1:1-2]`
- **Parentheses**: `This is a citation (Source 1, lines 1-2)`
- **Hyperlink**: `This is a citation [source](https://example.com/doc1#L1)`

## Customization

The citation system is designed to be highly customizable:

- Create custom citation styles
- Implement domain-specific verification logic
- Extend the document reference system

## Requirements

- DSPy
- Python 3.7+

## Future Enhancements

Possible future enhancements to the citation system:

1. **Pre-processing tools**: Tools to automatically process and enhance document collections
2. **Citation visualization**: Interactive visualization tools for citations and references
3. **Semantic search**: Improved semantic search for finding supporting evidence
4. **Citation formats**: More citation formats and styles
5. **Integrated evaluation**: Metrics and tools for evaluating citation quality 