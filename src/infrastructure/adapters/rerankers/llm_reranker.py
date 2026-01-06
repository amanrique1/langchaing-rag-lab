import json
from typing import List
from pathlib import Path
from src.application.ports.reranker import Reranker
from src.application.ports.language_model import LanguageModel
from src.domain.models.search_result import SearchResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser

try:
    reranking_query_template_path = Path("assets/templates/reranking_query_template.txt")

    if not reranking_query_template_path.exists():
        raise FileNotFoundError("The file assets/templates/reranking_query_template.txt does not exist")

    RERANKING_QUERY_TEMPLATE_CONTENT = reranking_query_template_path.read_text()
except FileNotFoundError as e:
    print(f"FATAL: Could not load reranking query template. {e}")
    raise

class LLMReranker(Reranker):
    """
    LLM-based reranker using Google Gemini to intelligently reorder search results.
    """

    def __init__(self, language_model: LanguageModel):
        """
        Initialize the LLM reranker.

        Args:
            language_model: Language model for reranking
        """
        self.language_model = language_model
        # Create the reranking chain using the template
        # Access the underlying model from the language_model adapter
        if hasattr(language_model, 'model'):
            self.chain: Runnable = (
                ChatPromptTemplate.from_template(RERANKING_QUERY_TEMPLATE_CONTENT)
                | language_model.model
                | StrOutputParser()
            )
        else:
            raise ValueError("Language model does not have a 'model' attribute")

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Rerank search results using LLM-based relevance assessment.

        Args:
            query: The search query
            results: List of search results to rerank
            top_k: Number of top results to return

        Returns:
            Reranked list of search results (top_k items)
        """
        if not results:
            return []

        # If we have fewer results than top_k, just return them sorted by score
        if len(results) <= top_k:
            return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]

        # Prepare passages for the template
        passages = []
        for i, result in enumerate(results, start=1):
            content = result.chunk.content
            passages.append(f"[{i}] {content}")

        passages_text = "\n\n".join(passages)

        # Get LLM response using the chain
        try:
            response = self._get_llm_ranking(query, passages_text)
            rankings = self._parse_ranking_response(response, len(results))
        except Exception as e:
            # Fallback to original scores if LLM fails
            print(f"LLM reranking failed: {e}. Falling back to original scores.")
            return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]

        # Reorder results based on LLM rankings
        reranked = self._apply_rankings(results, rankings)

        # Update ranks
        for i, result in enumerate(reranked[:top_k], start=1):
            result.rank = i

        return reranked[:top_k]

    def _get_llm_ranking(self, query: str, passages_text: str) -> str:
        """
        Get ranking from LLM using the chain.

        Args:
            query: The search query
            passages_text: Formatted passages text

        Returns:
            LLM response string
        """
        # The chain is invoked with a dictionary matching the variables in the template
        response = self.chain.invoke({
            "query": query,
            "passages_text": passages_text
        })

        return response

    def _parse_ranking_response(self, response: str, num_passages: int) -> List[int]:
        """
        Parse LLM response to extract ranking.

        Args:
            response: LLM response string
            num_passages: Expected number of passages

        Returns:
            List of passage indices in ranked order
        """
        try:
            # Try to find JSON array in response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON array found in response")

            json_str = response[start_idx:end_idx]
            rankings = json.loads(json_str)

            # Validate rankings
            if not isinstance(rankings, list):
                raise ValueError("Rankings is not a list")

            # Convert to 0-indexed and validate range
            rankings = [r - 1 for r in rankings if isinstance(r, int) and 1 <= r <= num_passages]

            # Add missing indices at the end
            all_indices = set(range(num_passages))
            ranked_indices = set(rankings)
            missing = all_indices - ranked_indices
            rankings.extend(sorted(missing))

            return rankings

        except Exception as e:
            # Fallback: return original order
            print(f"Failed to parse ranking response: {e}")
            return list(range(num_passages))

    def _apply_rankings(
        self,
        results: List[SearchResult],
        rankings: List[int]
    ) -> List[SearchResult]:
        """
        Apply rankings to reorder results.

        Args:
            results: Original search results
            rankings: List of indices in ranked order

        Returns:
            Reordered search results
        """
        reranked = []
        for idx in rankings:
            if 0 <= idx < len(results):
                reranked.append(results[idx])

        return reranked
