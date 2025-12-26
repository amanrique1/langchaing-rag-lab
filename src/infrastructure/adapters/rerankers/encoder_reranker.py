import torch
import logging
from typing import List
from sentence_transformers import CrossEncoder

from src.application.ports.reranker import Reranker
from src.domain.models.search_result import SearchResult

logger = logging.getLogger(__name__)

class EncoderReranker(Reranker):
    """
    Local reranker using Sentence Transformers Cross-Encoder.
    This approach is faster and deterministic compared to LLM reranking,
    and runs entirely locally.
    """

    def __init__(
        self, 
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize the Encoder Reranker.

        Args:
            model_name: The HuggingFace model ID to use.
                        Default is a lightweight, high-performance model.
        """
        self.model_name = model_name
        self._device = self._detect_device()
        
        try:
            logger.info(f"Loading CrossEncoder model: {model_name} on {self._device}...")
            self.model = CrossEncoder(model_name, device=self._device)
            logger.info("CrossEncoder loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load CrossEncoder model {model_name}: {e}")
            raise RuntimeError(f"Could not initialize EncoderReranker: {e}")

    def _detect_device(self) -> str:
        """Helper to determine the best available hardware acceleration."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        return "cpu"

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Rerank search results using Cross-Encoder scores.

        Args:
            query: The search query string.
            results: List of SearchResult objects from the retriever.
            top_k: Number of results to return after reranking.

        Returns:
            List[SearchResult]: The top_k results sorted by relevance.
        """
        
        if not results:
            logger.warning("Reranker received empty results list.")
            return []
        
        if not query.strip():
            logger.warning("Reranker received empty query. Returning original order.")
            return results[:top_k]

        try:
            # 1. Filter valid results and create pairs in one pass
            # We filter out results with empty content to avoid model errors
            valid_results = [r for r in results if getattr(r.chunk, 'content', None)]
            
            if not valid_results:
                logger.warning("No valid content found in results to rerank.")
                return []

            # Create inputs for the model: List[List[str]] -> [[query, text], ...]
            pairs = [[query, res.chunk.content] for res in valid_results]

            # 2. Predict scores (Vectorized operation)
            scores = self.model.predict(pairs)

            # 3. Map scores back to objects using zip (Avoids index tracking)
            for result, score in zip(valid_results, scores):
                result.score = float(score)

            # 4. Sort and Slice
            valid_results.sort(key=lambda x: x.score, reverse=True)
            top_results = valid_results[:top_k]

            # 5. Update Rank
            for rank, result in enumerate(top_results, start=1):
                result.rank = rank

            return top_results

        except Exception as e:
            logger.error(f"Error during encoding reranking: {e}. Returning original top_k.")
            # Fallback strategy: return original top_k
            return results[:top_k]