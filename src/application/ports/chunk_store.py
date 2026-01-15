from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.domain.models.chunk import Chunk
from src.domain.models.search_result import SearchResult


class ChunkStore(ABC):
    """
    Chunk store interface for persisting and retrieving chunks.

    Implementations should handle dual collection strategies internally,
    always returning complete chunks from search operations.
    """

    def __init__(
        self,
        collection_name: str = None,
        embedding_model = None,
        persist_directory: str = None,
        **kwargs
    ):
        """
        Initialize the chunk store.

        Args:
            collection_name: Name of the collection/table
            embedding_model: The embedding model to use
            persist_directory: Directory where data is persisted
            **kwargs: Additional store-specific parameters
        """
        pass

    @abstractmethod
    def save(self, chunks: List[Chunk]) -> None:
        """
        Save a list of chunks to the store.

        Args:
            chunks: List of Chunk objects to save
        """
        pass

    @abstractmethod
    def delete(self, chunk_id: str, where: Optional[Dict[str, Any]] = None, where_document: Optional[Dict[str, Any]] = None) -> None:
        """
        Delete a chunk by its ID.

        Args:
            chunk_id: ID of the chunk to delete
            where: Optional metadata filter for deletion
            where_document: Optional content filter for deletion
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        mode: str = "content"
    ) -> List[SearchResult]:
        """
        Search for chunks similar to the query.

        Args:
            query: Search query string
            top_k: Number of results to return
            filter: Optional metadata filter
            mode: Search mode - 'content' to search chunk text, 'metadata' to search metadata text

        Returns:
            List of SearchResult objects with COMPLETE chunks (always includes content)
        """
        pass

    @abstractmethod
    def get_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        """
        Retrieve complete chunks by their IDs.

        This method should be implemented efficiently for each store.
        Avoid full table scans or pandas conversions when possible.

        Args:
            chunk_ids: List of chunk IDs to retrieve

        Returns:
            List of complete chunks with content and metadata
        """
        pass

    @abstractmethod
    def get_client(self) -> Any:
        """
        Return the underlying database client or connection.

        This allows other components (like memory managers) to reuse the
        same connection/client without re-initializing it.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all chunks from the store.
        """
        pass