from typing import List, Dict, Optional, Any
from src.application.ports.chunk_store import ChunkStore
from src.application.ports.query_expander import QueryExpander
from src.domain.models.search_result import SearchResult
from src.application.ports.retriever import Retriever


class EnsembleRetriever(Retriever):
    """
    Retriever that combines content and metadata search using 
    Reciprocal Rank Fusion (RRF), with optional Query Expansion.
    """

    def __init__(
        self,
        chunk_store: ChunkStore,
        rrf_k: int = 60,
        content_weight: float = 1.0,
        metadata_weight: float = 1.0,
        query_expander: Optional[QueryExpander] = None
    ):
        """
        Initialize the EnsembleRetriever.
        
        Args:
            chunk_store (ChunkStore): The chunk store to search.
            rrf_k (int): RRF constant (default: 60).
            content_weight (float): Weight for content search results.
            metadata_weight (float): Weight for metadata search results.
            query_expander (Optional[QueryExpander]): Optional query expansion strategy.
        """
        super().__init__(query_expander)
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
        Executes ensemble retrieval with automatic query expansion if generator is available.
        
        1. Expands query (if generator exists).
        2. Runs RRF Ensemble for EACH query variation.
        3. Aggregates and deduplicates results.

        Args:
            query (str): The search query.
            top_k (int): The number of results to return.
            filter (Optional[Dict[str, Any]]): Optional metadata filter.
        
        Returns:
            List[SearchResult]: List of search results with scores.
        """
        # Get list of queries (includes expanded if generator exists)
        queries = self._get_expanded_queries(query)
        
        all_results = []

        # Run the ensemble logic for each query variation
        for q in queries:
            results = self._execute_single_pass(q, top_k, filter)
            all_results.extend(results)

        # If we only ran one query, return results directly
        if len(queries) == 1:
            return all_results

        # Deduplicate and sort if multiple queries were run
        return self._deduplicate_results(all_results, top_k)

    def _execute_single_pass(
        self, 
        query: str, 
        top_k: int, 
        filter: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """
        The core logic: Content + Metadata Search -> RRF -> Hydration.
        This runs for a single query string.

        Args:
            query (str): The search query.
            top_k (int): The number of results to return.
            filter (Optional[Dict[str, Any]]): Optional metadata filter.
        
        Returns:
            List[SearchResult]: List of search results with scores.
        """
        # A. Retrieve from both collections
        num_candidates = top_k * 2
        
        content_results = self.chunk_store.search(
            query, num_candidates, filter, mode="content"
        )
        metadata_results = self.chunk_store.search(
            query, num_candidates, filter, mode="metadata"
        )

        # B. Merge using RRF
        merged_results = self._reciprocal_rank_fusion(
            content_results,
            metadata_results
        )

        # C. Hydrate content for metadata-only hits
        self._hydrate_content(merged_results)

        # D. Sort and limit
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