import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union


@dataclass
class LineInfo:
    """Information about a specific line in a document."""

    line_number: int
    text: str
    char_start: int
    char_end: int


@dataclass
class EnhancedDocument:
    """Enhanced document structure that preserves line information."""

    doc_id: str
    content: str
    source_name: Optional[str] = None
    source_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _lines: List[LineInfo] = field(default_factory=list, repr=False)

    def __post_init__(self):
        """Process the document content to extract line information."""
        if not self._lines:
            self._process_lines()

    def _process_lines(self):
        """Process the document content to extract line information."""
        lines = self.content.split("\n")
        char_position = 0

        for line_number, line in enumerate(lines, 1):
            line_length = len(line)
            self._lines.append(
                LineInfo(
                    line_number=line_number,
                    text=line,
                    char_start=char_position,
                    char_end=char_position + line_length,
                )
            )
            # Add 1 for the newline character
            char_position += line_length + 1

    def get_line(self, line_number: int) -> Optional[LineInfo]:
        """Get line information for a specific line number."""
        if 1 <= line_number <= len(self._lines):
            return self._lines[line_number - 1]
        return None

    def get_lines(self, start_line: int, end_line: int) -> List[LineInfo]:
        """Get line information for a range of line numbers."""
        start = max(1, start_line) - 1
        end = min(len(self._lines), end_line)
        return self._lines[start:end]

    def get_snippet(self, start_line: int, end_line: int) -> str:
        """Get the text of a range of lines."""
        lines = self.get_lines(start_line, end_line)
        return "\n".join(line.text for line in lines)

    def find_text_position(self, text: str) -> List[Tuple[int, int, int, int]]:
        """
        Find all occurrences of text and return positions as
        (start_line, start_char, end_line, end_char) tuples.
        """
        positions = []
        regex = re.compile(re.escape(text))

        for match in regex.finditer(self.content):
            start_pos = match.start()
            end_pos = match.end()

            # Find start line and character
            start_line = None
            start_char = None
            for line in self._lines:
                if line.char_start <= start_pos < line.char_end:
                    start_line = line.line_number
                    start_char = start_pos - line.char_start
                    break

            # Find end line and character
            end_line = None
            end_char = None
            for line in self._lines:
                if line.char_start < end_pos <= line.char_end:
                    end_line = line.line_number
                    end_char = end_pos - line.char_start
                    break

            if start_line and end_line:
                positions.append((start_line, start_char, end_line, end_char))

        return positions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "metadata": self.metadata,
            "line_count": len(self._lines),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnhancedDocument":
        """Create an EnhancedDocument from a dictionary."""
        return cls(
            doc_id=data["doc_id"],
            content=data["content"],
            source_name=data.get("source_name"),
            source_url=data.get("source_url"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DocumentReference:
    """A reference to a specific part of a document."""

    doc_id: str
    start_line: int
    end_line: int
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    text: Optional[str] = None
    source_name: Optional[str] = None
    source_url: Optional[str] = None

    def format_citation(self, citation_style: str = "brackets") -> str:
        """Format the citation according to the specified style."""
        if citation_style == "brackets":
            return f"[{self.doc_id}:{self.start_line}-{self.end_line}]"
        elif citation_style == "parentheses":
            return f"({self.source_name or self.doc_id}, lines {self.start_line}-{self.end_line})"
        elif citation_style == "hyperlink":
            if self.source_url:
                link_target = f"{self.source_url}#L{self.start_line}"
                return f"[{self.source_name or 'source'}]({link_target})"
            else:
                return f"[{self.source_name or self.doc_id}]"
        else:
            return f"{self.doc_id}:{self.start_line}-{self.end_line}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "doc_id": self.doc_id,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "text": self.text,
            "source_name": self.source_name,
            "source_url": self.source_url,
        }


@dataclass
class Citation:
    """A citation in the generated text, linking to a document reference."""

    reference: DocumentReference
    text: str  # The text being cited
    citation_id: str  # Unique identifier for the citation
    citation_style: str = "brackets"

    def format(self) -> str:
        """Format the citation for inclusion in text."""
        return f"{self.text} {self.reference.format_citation(self.citation_style)}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "reference": self.reference.to_dict(),
            "text": self.text,
            "citation_id": self.citation_id,
            "citation_style": self.citation_style,
        }


@dataclass
class CitedText:
    """Text with citations."""

    text: str
    citations: List[Citation] = field(default_factory=list)
    references: Dict[str, DocumentReference] = field(default_factory=dict)

    def add_citation(self, citation: Citation) -> None:
        """Add a citation to the cited text."""
        self.citations.append(citation)
        self.references[citation.citation_id] = citation.reference

    def get_text_with_citations(self) -> str:
        """Get the text with formatted citations."""
        result = self.text
        for citation in sorted(self.citations, key=lambda c: len(c.text), reverse=True):
            result = result.replace(citation.text, citation.format())
        return result

    def get_references_section(self, style: str = "numbered") -> str:
        """Generate a references section from the citations."""
        if not self.references:
            return ""

        if style == "numbered":
            refs = [
                f"[{i+1}] {ref.source_name or ref.doc_id}"
                for i, ref in enumerate(self.references.values())
            ]
            return "References:\n" + "\n".join(refs)
        elif style == "academic":
            refs = []
            for i, ref in enumerate(self.references.values()):
                if ref.source_url:
                    refs.append(
                        f"{ref.source_name or ref.doc_id}. Retrieved from {ref.source_url}"
                    )
                else:
                    refs.append(f"{ref.source_name or ref.doc_id}.")
            return "References:\n" + "\n".join(refs)
        else:
            return ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "citations": [citation.to_dict() for citation in self.citations],
            "references": {k: v.to_dict() for k, v in self.references.items()},
        }
