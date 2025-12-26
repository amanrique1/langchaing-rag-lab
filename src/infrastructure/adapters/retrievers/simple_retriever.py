from typing import List, Optional, Dict, Any
from src.application.ports.retriever import Retriever
from src.application.ports.chunk_store import ChunkStore
from src.domain.models.search_result import SearchResult


class SimpleRetriever(Retriever):
    """Basic retriever that searches content collection only."""
    
    def __init__(self, chunk_store: ChunkStore):
        self.chunk_store = chunk_store
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Retrieve the top-k most similar chunks to the query.

        Args:
            query (str): The search query.
            top_k (int): The number of results to return.
            filter (Optional[Dict[str, Any]]): Optional metadata filter.
        
        Returns:
            List[SearchResult]: List of search results with scores.
        """
        return self.chunk_store.search(query, top_k, filter, mode="content")