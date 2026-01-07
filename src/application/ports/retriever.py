import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from src.domain.models.search_result import SearchResult
from src.domain.services.retrieval.query_expander import QueryExpander

logger = logging.getLogger(__name__)


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

        if query_expander:
            logger.info(
                f"{self.__class__.__name__} initialized with query expansion "
                f"(strategy: {query_expander.strategy.value})"
            )
        else:
            logger.info(f"{self.__class__.__name__} initialized without query expansion")

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
            logger.debug("No query expansion configured, using original query")
            return [query]

        # Log business decision to expand
        query_preview = self._truncate(query, 80)
        logger.info(
            f"Expanding query with {self.query_expander.strategy.value}: '{query_preview}'"
        )

        try:
            expanded = self.query_expander.generate(query)

            if isinstance(expanded, list):
                total_queries = 1 + len(expanded)
                logger.info(
                    f"Using {total_queries} query variations: 1 original + {len(expanded)} expanded"
                )

                # Log expanded queries at debug level for troubleshooting
                if logger.isEnabledFor(logging.DEBUG):
                    for idx, exp_query in enumerate(expanded, 1):
                        logger.debug(f"  Variation {idx+1}: '{self._truncate(exp_query, 100)}'")

                return [query] + expanded
            else:
                logger.info("Using 2 query variations: 1 original + 1 expanded")
                logger.debug(f"  Variation 2: '{self._truncate(expanded, 100)}'")
                return [query, expanded]

        except Exception as e:
            # Business decision: fallback to original query
            logger.warning(
                f"Query expansion failed ({self.query_expander.strategy.value}), "
                f"falling back to original query: {e}"
            )
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
        initial_count = len(results)

        if initial_count == 0:
            logger.debug("No results to deduplicate")
            return []

        logger.debug(f"Deduplicating {initial_count} results (top_k={top_k})")

        seen = {}
        duplicates_found = 0
        score_improvements = 0

        for result in results:
            cid = result.chunk.chunk_id

            if cid not in seen:
                seen[cid] = result
            else:
                duplicates_found += 1
                if result.score > seen[cid].score:
                    score_improvements += 1
                    old_score = seen[cid].score
                    seen[cid] = result
                    logger.debug(
                        f"Updated chunk '{cid}': score {old_score:.4f} → {result.score:.4f}"
                    )

        deduplicated = list(seen.values())
        deduplicated.sort(key=lambda x: x.score, reverse=True)
        final_results = deduplicated[:top_k]

        # Summary
        unique_count = len(deduplicated)
        returned_count = len(final_results)

        if duplicates_found > 0:
            logger.info(
                f"Deduplicated {initial_count} → {unique_count} unique "
                f"({duplicates_found} duplicates, {score_improvements} improved), "
                f"returning top {returned_count}"
            )
        else:
            logger.debug(f"No duplicates found, returning top {returned_count}")

        # Log top results at debug level
        if logger.isEnabledFor(logging.DEBUG) and final_results:
            logger.debug(f"Top results (scores: {final_results[-1].score:.4f} to {final_results[0].score:.4f}):")
            for idx, result in enumerate(final_results[:3], 1):
                content_preview = self._truncate(
                    getattr(result.chunk, 'content', 'N/A'),
                    60
                )
                logger.debug(
                    f"  [{idx}] {result.score:.4f} | {result.chunk.chunk_id} | {content_preview}"
                )

        return final_results

    @staticmethod
    def _truncate(text: str, max_length: int) -> str:
        """Helper to truncate text with ellipsis."""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."