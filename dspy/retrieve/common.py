from dataclasses import dataclass
from typing import List


@dataclass
class CollectedResult:
    snippets: List[str]
    title: str
    url: str
    description: str
    pub_date: str
