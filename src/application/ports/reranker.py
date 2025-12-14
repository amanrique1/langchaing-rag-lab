from abc import ABC, abstractmethod
from typing import List
from src.domain.models.search_result import SearchResult


class Reranker(ABC):
    """
    Interface for reranking search results based on relevance to a query.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Rerank search results based on relevance to the query.

        Args:
            query: The search query
            results: List of search results to rerank
            top_k: Number of top results to return

        Returns:
            Reranked list of search results (top_k items)
        """
        pass
