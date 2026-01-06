import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict

from src.application.ports.chunk_store import ChunkStore
from src.domain.services.retrieval.query_expander import QueryExpander
from src.application.ports.retriever import Retriever
from src.domain.models.search_result import SearchResult

logger = logging.getLogger(__name__)


class EnsembleRetriever(Retriever):
    """
    Advanced retrieval strategy using Reciprocal Rank Fusion (RRF).

    This retriever combines results from:
    1. Content-based semantic search (vector similarity on chunk text)
    2. Metadata-based semantic search (vector similarity on metadata fields)

    **Key Optimization:**
    All ChunkStore implementations now return COMPLETE chunks from search(),
    eliminating the need for a separate hydration step. This significantly
    improves performance by reducing database round-trips.

    RRF Formula:
        RRF_score(d) = Σ (weight / (k + rank(d)))

    where:
    - d = document/chunk
    - k = RRF constant (typically 60)
    - rank(d) = position in result list
    - weight = importance multiplier for each source
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
            chunk_store (ChunkStore): The storage backend (ChromaDB, LanceDB, FileSystem).
            rrf_k (int): The RRF constant. Higher values reduce the impact of rank position.
                        Typical range: 1-100. Default: 60.
            content_weight (float): Multiplier for content-based search scores.
                                   Higher = favor semantic content matches.
            metadata_weight (float): Multiplier for metadata-based search scores.
                                    Higher = favor metadata/keyword matches.
            query_expander (Optional[QueryExpander]): Optional query expansion strategy.
        """
        super().__init__(query_expander)
        self.chunk_store = chunk_store
        self.rrf_k = rrf_k
        self.content_weight = content_weight
        self.metadata_weight = metadata_weight

        logger.info(
            f"EnsembleRetriever initialized with RRF_k={rrf_k}, "
            f"content_weight={content_weight}, metadata_weight={metadata_weight}"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Retrieve the top-k most similar chunks using ensemble retrieval.

        This method:
        1. Optionally expands the query if query_expander is configured
        2. Executes retrieval for each query (original + expanded)
        3. Merges all results using deduplication
        4. Returns top_k results sorted by score

        Args:
            query (str): The search query.
            top_k (int): The number of results to return.
            filter (Optional[Dict[str, Any]]): Optional metadata filter.

        Returns:
            List[SearchResult]: List of search results with RRF scores.
        """
        logger.info(f"Starting ensemble retrieval for query: '{query}' (top_k={top_k})")

        # Get expanded queries (includes original if no expander)
        queries = self._get_expanded_queries(query)
        logger.debug(f"Processing {len(queries)} queries (original + expanded)")

        # Collect results from all queries
        all_results = []
        for idx, q in enumerate(queries):
            logger.debug(f"Executing retrieval pass {idx + 1}/{len(queries)}: '{q}'")
            results = self._execute_single_pass(q, top_k, filter)
            all_results.extend(results)

        # Deduplicate and merge results
        logger.debug(f"Deduplicating {len(all_results)} total results")
        final_results = self._deduplicate_results(all_results, top_k)

        # Re-rank the final results
        for rank, result in enumerate(final_results, start=1):
            result.rank = rank

        logger.info(
            f"Ensemble retrieval complete: returned {len(final_results)} results "
            f"from {len(all_results)} candidates"
        )

        return final_results

    def _execute_single_pass(
        self,
        query: str,
        top_k: int,
        filter: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """
        Executes a single retrieval pass with RRF fusion.

        **Optimized Flow:**
        1. Parallel searches on content and metadata
        2. Both return COMPLETE chunks (no hydration needed!)
        3. Merge using RRF with custom weights
        4. Sort and return top_k results

        Args:
            query (str): The search query.
            top_k (int): Number of final results to return.
            filter (Optional[Dict]): Metadata filters to apply.

        Returns:
            List[SearchResult]: The top_k merged results with complete chunks.
        """
        # Fetch more candidates than needed for better fusion quality
        num_candidates = top_k * 2

        # A. Content-based semantic search
        logger.debug(f"Executing content search for: '{query}'")
        try:
            content_results = self.chunk_store.search(
                query=query,
                top_k=num_candidates,
                filter=filter,
                mode="content"
            )
            logger.debug(f"Content search returned {len(content_results)} results")
        except Exception as e:
            logger.error(f"Content search failed: {e}")
            content_results = []

        # B. Metadata-based semantic search
        logger.debug(f"Executing metadata search for: '{query}'")
        try:
            metadata_results = self.chunk_store.search(
                query=query,
                top_k=num_candidates,
                filter=filter,
                mode="metadata"
            )
            logger.debug(f"Metadata search returned {len(metadata_results)} results")
        except Exception as e:
            logger.error(f"Metadata search failed: {e}")
            metadata_results = []

        # C. Merge using Reciprocal Rank Fusion
        logger.debug("Merging results using RRF")
        merged_results = self._reciprocal_rank_fusion(
            content_results,
            metadata_results
        )

        # D. Sort by fused score (descending) and limit to top_k
        merged_results.sort(key=lambda x: x.score, reverse=True)
        final_results = merged_results[:top_k]

        logger.debug(
            f"Single pass complete: {len(final_results)} results "
            f"(from {len(content_results)} content + {len(metadata_results)} metadata)"
        )

        return final_results

    def _reciprocal_rank_fusion(
        self,
        content_results: List[SearchResult],
        metadata_results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Merges two result lists using Reciprocal Rank Fusion with custom weights.

        **Algorithm:**
        1. For each result list, assign RRF score: weight / (k + rank)
        2. Sum scores for chunks that appear in multiple lists
        3. Keep the chunk object from the content results (preferred) or metadata results

        **Key Point:** All chunks are already complete - no hydration needed!

        Args:
            content_results (List[SearchResult]): Results from content search.
            metadata_results (List[SearchResult]): Results from metadata search.

        Returns:
            List[SearchResult]: Merged results with RRF scores.
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, SearchResult] = {}
        retrieval_methods: Dict[str, List[str]] = defaultdict(list)

        # Process content results
        for result in content_results:
            chunk_id = result.chunk.chunk_id
            if not chunk_id:
                continue

            # RRF contribution from content search
            score_contribution = self.content_weight / (self.rrf_k + result.rank)
            rrf_scores[chunk_id] += score_contribution
            retrieval_methods[chunk_id].append(f"content(r={result.rank})")

            # Store chunk (prefer content source for complete data)
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = result

        # Process metadata results
        for result in metadata_results:
            chunk_id = result.chunk.chunk_id
            if not chunk_id:
                continue

            # RRF contribution from metadata search
            score_contribution = self.metadata_weight / (self.rrf_k + result.rank)
            rrf_scores[chunk_id] += score_contribution
            retrieval_methods[chunk_id].append(f"metadata(r={result.rank})")

            # Store chunk only if not already present from content search
            # (content results have priority since they're already optimized)
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = result

        # Build final merged results
        merged = []
        for chunk_id, rrf_score in rrf_scores.items():
            if chunk_id in chunk_map:
                original_result = chunk_map[chunk_id]

                # Create method description
                methods = retrieval_methods[chunk_id]
                method_str = f"rrf_ensemble[{'+'.join(methods)}]"

                # Create new SearchResult with RRF score
                merged.append(SearchResult(
                    chunk=original_result.chunk,  # Already complete!
                    score=rrf_score,
                    retrieval_method=method_str,
                    rank=None  # Will be assigned after sorting
                ))

        logger.debug(f"RRF fusion produced {len(merged)} unique chunks")

        return merged