from typing import List, Dict, Optional
from src.domain.models.search_result import SearchResult
from src.application.ports.chunk_store import ChunkStore

class EnsembleRetrieverService:
    """
    Service that orchestrates multiple retrieval strategies and merges results
    using Reciprocal Rank Fusion (RRF).
    
    This service is responsible for querying different indices (e.g., Content vs Metadata),
    normalizing their scores, merging them, and ensuring the final result contains 
    actual content text rather than metadata placeholders.
    """

    def __init__(
        self,
        chunk_store: ChunkStore,
        rrf_k: int = 60
    ):
        """
        Initialize ensemble retriever service.

        Args:
            chunk_store (ChunkStore): The unified store containing both content 
                                      and metadata collections.
            rrf_k (int): Constant for RRF algorithm (smoothing factor). 
                         Defaults to 60.
        """
        self.chunk_store = chunk_store
        self.rrf_k = rrf_k

    def execute(
        self,
        query: str,
        top_k: int = 20,
        content_weight: float = 1.0,
        metadata_weight: float = 1.0
    ) -> List[SearchResult]:
        """
        Executes the ensemble retrieval pipeline.

        Steps:
        1. Parallel search (Content Collection vs Metadata Collection).
        2. Reciprocal Rank Fusion (Merge & Dedup).
        3. Content Hydration (Swap metadata text for real content).
        4. Final Sorting.

        Args:
            query (str): The search query.
            top_k (int): Number of candidates to retrieve from *each* source 
                         before merging.
            content_weight (float): Importance weight for content matches.
            metadata_weight (float): Importance weight for metadata matches.

        Returns:
            List[SearchResult]: Merged, hydrated, and sorted results.
        """
        # 1. Retrieve from both collections
        content_results = self.chunk_store.search(query, top_k=top_k, mode="content")
        metadata_results = self.chunk_store.search(query, top_k=top_k, mode="metadata")

        # 2. Merge using Reciprocal Rank Fusion
        merged_results = self._reciprocal_rank_fusion(
            content_results,
            metadata_results,
            content_weight,
            metadata_weight
        )

        # 3. Hydrate content for metadata-only hits
        self._hydrate_content(merged_results)

        # 4. Sort by final RRF score (descending)
        merged_results.sort(key=lambda x: x.score, reverse=True)

        return merged_results

    def _reciprocal_rank_fusion(
        self,
        content_results: List[SearchResult],
        metadata_results: List[SearchResult],
        content_weight: float,
        metadata_weight: float
    ) -> List[SearchResult]:
        """
        Merges two lists of results using the Reciprocal Rank Fusion algorithm.
        
        Formula: Score = Sum( weight / (k + rank) ) for each occurrence.
        
        Args:
            content_results (List[SearchResult]): Results from content search.
            metadata_results (List[SearchResult]): Results from metadata search.
            content_weight (float): Weight for content results.
            metadata_weight (float): Weight for metadata results.

        Returns:
            List[SearchResult]: A new list of unique results with RRF scores.
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, SearchResult] = {}

        def process_list(results: List[SearchResult], weight: float, is_content_source: bool):
            """Inner helper to process a result list."""
            for result in results:
                # Resolve ID: Metadata docs store the real chunk_id inside metadata dict
                if is_content_source:
                    c_id = result.chunk.chunk_id
                else:
                    c_id = result.chunk.metadata.get('chunk_id', result.chunk.chunk_id)

                # RRF Math: Add score contribution
                score_contribution = weight / (self.rrf_k + result.rank)
                rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + score_contribution

                # Store the Result Object.
                # Prefer the object from 'content_results' as it has the real text.
                if c_id not in chunk_map or is_content_source:
                    if not is_content_source:
                        result.chunk.chunk_id = c_id # Normalize ID
                    chunk_map[c_id] = result

        process_list(content_results, content_weight, is_content_source=True)
        process_list(metadata_results, metadata_weight, is_content_source=False)

        # Reconstruct final list
        final_results = []
        for c_id, total_score in rrf_scores.items():
            if c_id in chunk_map:
                res = chunk_map[c_id]
                final_results.append(SearchResult(
                    chunk=res.chunk,
                    score=total_score, 
                    retrieval_method="ensemble_rrf",
                    rank=None # To be determined by final sort
                ))

        return final_results

    def _hydrate_content(self, results: List[SearchResult]) -> None:
        """
        Fetches real content for chunks found via metadata-only search.

        Args:
            results (List[SearchResult]): The list of merged results (modified in-place).
        """
        ids_to_fetch = []
        indices_map = {}

        for i, res in enumerate(results):
            # Check flag often set by metadata collection ingestion
            if res.chunk.metadata.get('is_metadata_doc', False):
                c_id = res.chunk.chunk_id
                ids_to_fetch.append(c_id)
                indices_map[c_id] = i

        if not ids_to_fetch:
            return

        # Bulk fetch real content from the store
        real_chunks = self.chunk_store.get_by_ids(ids_to_fetch)

        for real_chunk in real_chunks:
            if real_chunk.chunk_id in indices_map:
                idx = indices_map[real_chunk.chunk_id]
                results[idx].chunk = real_chunk