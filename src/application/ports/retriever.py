from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from src.domain.models.search_result import SearchResult


class Retriever(ABC):
    """
    Port for retrieval strategies.
    Separates retrieval logic from storage concerns.
    """
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Retrieve relevant chunks for the given query.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of SearchResult objects with scores
        """
        pass