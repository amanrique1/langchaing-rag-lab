from typing import List, Dict, Optional, Any
from src.application.ports.retriever import Retriever
from src.application.ports.chunk_store import ChunkStore
from src.domain.models.search_result import SearchResult


class EnsembleRetriever(Retriever):
    """
    Retriever that combines content and metadata search using 
    Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        chunk_store: ChunkStore,
        rrf_k: int = 60,
        content_weight: float = 1.0,
        metadata_weight: float = 1.0
    ):
        """
        Initialize the EnsembleRetriever with a chunk store and optional parameters.
        
        Args:
            chunk_store (ChunkStore): The chunk store to use for retrieval.
            rrf_k (int): The number of candidates to consider for RRF.
            content_weight (float): The weight of content in the RRF score.
            metadata_weight (float): The weight of metadata in the RRF score.
        """
        self.chunk_store = chunk_store
        self.rrf_k = rrf_k
        self.content_weight = content_weight
        self.metadata_weight = metadata_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Executes ensemble retrieval: Content + Metadata -> RRF -> Hydration.
        
        Args:
            query (str): The search query.
            top_k (int): The number of results to return.
            filter (Optional[Dict[str, Any]]): Optional metadata filter.
        
        Returns:
            List[SearchResult]: List of search results with scores.
        """
        # 1. Retrieve from both collections
        # Get more candidates initially since we'll merge and dedupe
        num_candidates = top_k * 2
        
        content_results = self.chunk_store.search(
            query, num_candidates, filter, mode="content"
        )
        metadata_results = self.chunk_store.search(
            query, num_candidates, filter, mode="metadata"
        )

        # 2. Merge using RRF
        merged_results = self._reciprocal_rank_fusion(
            content_results,
            metadata_results
        )

        # 3. Hydrate content for metadata-only hits
        self._hydrate_content(merged_results)

        # 4. Sort and limit
        merged_results.sort(key=lambda x: x.score, reverse=True)
        return merged_results[:top_k]

    def _reciprocal_rank_fusion(
        self,
        content_results: List[SearchResult],
        metadata_results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Merges results using RRF algorithm.
        
        Args:
            content_results (List[SearchResult]): List of content search results.
            metadata_results (List[SearchResult]): List of metadata search results.
        
        Returns:
            List[SearchResult]: List of merged search results with scores.
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, SearchResult] = {}

        def process_list(results: List[SearchResult], weight: float, is_content: bool):
            for result in results:
                c_id = (result.chunk.chunk_id if is_content 
                       else result.chunk.metadata.get('chunk_id', result.chunk.chunk_id))
                
                score_contribution = weight / (self.rrf_k + result.rank)
                rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + score_contribution

                if c_id not in chunk_map or is_content:
                    if not is_content:
                        result.chunk.chunk_id = c_id
                    chunk_map[c_id] = result

        process_list(content_results, self.content_weight, True)
        process_list(metadata_results, self.metadata_weight, False)

        return [
            SearchResult(
                chunk=chunk_map[c_id].chunk,
                score=score,
                retrieval_method="ensemble_rrf",
                rank=None
            )
            for c_id, score in rrf_scores.items()
            if c_id in chunk_map
        ]

    def _hydrate_content(self, results: List[SearchResult]) -> None:
        """
        Fetches real content for metadata-only chunks.
        
        Args:
            results (List[SearchResult]): List of search results to hydrate.
        """
        ids_to_fetch = []
        indices_map = {}

        for i, res in enumerate(results):
            if res.chunk.metadata.get('is_metadata_doc', False):
                c_id = res.chunk.chunk_id
                ids_to_fetch.append(c_id)
                indices_map[c_id] = i

        if not ids_to_fetch:
            return

        real_chunks = self.chunk_store.get_by_ids(ids_to_fetch)
        for real_chunk in real_chunks:
            if real_chunk.chunk_id in indices_map:
                idx = indices_map[real_chunk.chunk_id]
                results[idx].chunk = real_chunk