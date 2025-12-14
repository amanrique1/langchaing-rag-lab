from dataclasses import dataclass
from typing import Optional
from src.domain.models.chunk import Chunk

@dataclass
class SearchResult:
    """
    Represents a search result with relevance scoring and retrieval metadata.
    """
    chunk: Chunk
    retrieval_method: str
    score: float = None
    rank: Optional[int] = None

    def __lt__(self, other):
        """Enable sorting by score (descending). Higher is better."""
        return self.score > other.score

    def __eq__(self, other):
        """Check equality based on chunk_id."""
        if not isinstance(other, SearchResult):
            return False
        return self.chunk.chunk_id == other.chunk.chunk_id

    def __hash__(self):
        """Enable use in sets for deduplication."""
        return hash(self.chunk.chunk_id)