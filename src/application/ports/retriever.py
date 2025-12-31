from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from src.domain.models.search_result import SearchResult
from src.application.ports.query_expander import QueryExpander

class Retriever(ABC):
    """
    Base retriever interface with query expansion support.
    """
    
    def __init__(self, query_expander: Optional[QueryExpander] = None):
        """
        Initialize the base retriever.
        
        Args:
            query_expander (Optional[QueryExpander]): Optional query expansion strategy.
        """
        self.query_expander = query_expander
    
    @abstractmethod
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
        pass
    
    def _get_expanded_queries(self, query: str) -> List[str]:
        """
        Get list of queries. If query_expander exists, return both original and expanded.
        
        Args:
            query (str): The original query.
        
        Returns:
            List[str]: List of queries (original + expanded if expander is available).
        """
        if not self.query_expander:
            return [query]
        
        # Generate expanded query using the injected expander
        try:
            expanded = self.query_expander.generate(query)
            return [query, expanded]
        except Exception as e:
            print(f"Warning: Query expansion failed: {e}. Using original query only.")
            return [query]
    
    def _deduplicate_results(
        self, 
        results: List[SearchResult], 
        top_k: int
    ) -> List[SearchResult]:
        """
        Deduplicate and aggregate results by chunk_id.
        
        Args:
            results (List[SearchResult]): List of search results to deduplicate.
            top_k (int): Number of results to return.
        
        Returns:
            List[SearchResult]: Deduplicated and sorted results.
        """
        seen = {}
        for result in results:
            cid = result.chunk.chunk_id
            if cid not in seen or result.score > seen[cid].score:
                seen[cid] = result
        
        deduplicated = list(seen.values())
        deduplicated.sort(key=lambda x: x.score, reverse=True)
        return deduplicated[:top_k]