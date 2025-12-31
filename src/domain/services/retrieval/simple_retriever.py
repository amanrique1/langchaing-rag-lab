from typing import List, Optional, Dict, Any
from src.application.ports.retriever import Retriever
from src.application.ports.chunk_store import ChunkStore
from src.application.ports.query_expander import QueryExpander
from src.domain.models.search_result import SearchResult


class SimpleRetriever(Retriever):
    """Basic retriever that searches content collection only."""
    
    def __init__(
        self, 
        chunk_store: ChunkStore, 
        query_expander: Optional[QueryExpander] = None
    ):
        """
        Initialize the SimpleRetriever.
        
        Args:
            chunk_store (ChunkStore): The chunk store to search.
            query_expander (Optional[QueryExpander]): Optional query expansion strategy.
        """
        super().__init__(query_expander)
        self.chunk_store = chunk_store
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Retrieve the top-k most similar chunks to the query.
        Uses query expansion automatically if query_expander was provided.

        Args:
            query (str): The search query.
            top_k (int): The number of results to return.
            filter (Optional[Dict[str, Any]]): Optional metadata filter.
        
        Returns:
            List[SearchResult]: List of search results with scores.
        """
        # Get queries (will include expanded query if generator exists)
        queries = self._get_expanded_queries(query)
        
        all_results = []
        # Run search for every query variation
        for q in queries:
            results = self.chunk_store.search(q, top_k, filter, mode="content")
            all_results.extend(results)
            
        # Deduplicate and return
        return self._deduplicate_results(all_results, top_k)