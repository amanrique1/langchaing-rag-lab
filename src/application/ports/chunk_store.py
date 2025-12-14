from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.domain.models.chunk import Chunk
from src.domain.models.search_result import SearchResult


class ChunkStore(ABC):
    """Chunk store interface for persisting and retrieving chunks."""
    
    @abstractmethod
    def save(self, chunks: list[Chunk]):
        """
        Save a list of chunks to the store.
        
        Args:
            chunks: List of Chunk objects to save
        """
        pass

    @abstractmethod
    def delete(self, chunk_id: str):
        """
        Delete a chunk by its ID.
        
        Args:
            chunk_id: ID of the chunk to delete
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        mode: str = "content"
    ) -> list[SearchResult]:
        """
        Search for chunks similar to the query.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            filter: Optional metadata filter
            mode: Search mode - 'content' to search chunk text, 'metadata' to search metadata text
            
        Returns:
            List of SearchResult objects with scores
        """
        pass

    @abstractmethod
    def get_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        """
        Retrieve chunks by their IDs.
        
        Args:
            chunk_ids: List of chunk IDs to retrieve
            
        Returns:
            List of matching chunks
        """
        pass

    @abstractmethod
    def clear(self):
        """
        Clear all chunks from the store.
        """
        pass
