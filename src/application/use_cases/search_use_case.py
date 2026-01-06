from typing import List, Optional
from src.application.ports.retriever import Retriever
from src.application.ports.reranker import Reranker
from src.domain.models.chunk import Chunk


class SearchUseCase:
    """
    Orchestrates search: Retrieval -> Reranking -> Extraction.

    Now cleaner with single dependency on Retriever strategy.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Optional[Reranker] = None
    ):
        """
        Args:
            retriever: The retrieval strategy to use
            reranker: Optional reranker for refining results
        """
        self.retriever = retriever
        self.reranker = reranker

    def execute(
        self,
        query: str,
        top_k: int = 5,
        num_candidates: int = 20,
        use_reranking: bool = True
    ) -> List[Chunk]:
        """
        Execute search pipeline.

        Args:
            query: Search query
            top_k: Number of final results
            num_candidates: Number of candidates before reranking
            use_reranking: Whether to apply reranker

        Returns:
            List of most relevant chunks
        """
        # 1. Retrieval (strategy handles the details)
        results = self.retriever.retrieve(query, num_candidates)

        # 2. Optional Reranking
        if use_reranking and self.reranker:
            results = self.reranker.rerank(query, results, top_k)
        else:
            results = results[:top_k]

        # 3. Extract domain objects
        return [r.chunk for r in results]