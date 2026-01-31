import logging
from typing import List, Optional, Any

from src.application.graphs.rag_graph import create_search_graph
from src.application.graphs.nodes.rag_nodes import RAGNodes
from src.application.ports.chunk_store import ChunkStore
from src.application.ports.reranker import Reranker
from src.domain.guardrails.input_guard import InputGuard
from src.domain.models.chunk import Chunk
from src.infrastructure.adapters.models.llama_guard_model import LlamaGuard


logger = logging.getLogger(__name__)


class SearchUseCase:
    """
    Orchestrates search pipeline using LangGraph: Validation → Retrieval → Reranking.

    This use case returns chunks only (no generation). For Q&A with generation,
    use TalkUseCase. For conversational Q&A, use ChatUseCase.
    """

    def __init__(
        self,
        chunk_store: ChunkStore,
        reranker: Optional[Reranker] = None,
        input_guard: Optional[InputGuard] = None,
        query_expander: Optional[Any] = None
    ):
        """
        Initialize Search Use Case with LangGraph.

        Args:
            chunk_store: Vector store for retrieval
            reranker: Optional reranker for refining results
            input_guard: Optional security gateway for validation
            query_expander: Optional query expansion strategy
        """
        self.chunk_store = chunk_store
        self.reranker = reranker
        self.input_guard = input_guard
        self.query_expander = query_expander

        # Initialize nodes
        self.nodes = RAGNodes(
            language_model=None,  # Not needed for search
            chunk_store=chunk_store,
            reranker=reranker,
            input_guard=input_guard or self._create_minimal_guard(),
            query_expander=query_expander
        )

        # Build search-only graph (no generation)
        self.graph = create_search_graph(nodes=self.nodes)

        logger.info("SearchUseCase initialized with LangGraph")

    def _create_minimal_guard(self) -> InputGuard:
        """
        Create a minimal input guard that only does basic validation.
        This is used when no guard is provided.
        """

        return InputGuard(LlamaGuard())

    def execute(
        self,
        query: str,
        top_k: int = 5,
        num_candidates: int = 20,
        use_reranking: bool = True,
        dual_collection: bool = True
    ) -> List[Chunk]:
        """
        Execute search pipeline through LangGraph.

        Pipeline:
        1. Fast validation (regex-based security)
        2. Semantic validation (AI-based security)
        3. Query expansion (optional)
        4. Parallel retrieval (content + metadata)
        5. Fusion (RRF)
        6. Reranking (cross-encoder)

        Args:
            query: Search query
            top_k: Number of final results
            num_candidates: Number of candidates before reranking
            use_reranking: Whether to apply reranker
            dual_collection: Whether to use both content and metadata search

        Returns:
            List of most relevant chunks (ordered by relevance)
        """
        try:
            # Prepare initial state
            initial_state = {
                "user_id": "search",
                "session_id": "stateless",
                "query": query,
                "top_k": top_k,
                "num_candidates": num_candidates,
                "use_reranking": use_reranking,
                "dual_collection": dual_collection,
                "memory_k": 0,  # No memory for search
                "is_safe_fast": True,
                "is_safe_semantic": True,
                "error": None,
                "answer": None,
                "expanded_queries": None,
                "content_results": None,
                "metadata_results": None,
                "candidates": None,
                "chunks": None,
                "context_text": None,
                "messages": []  # Empty - no conversation
            }

            # Invoke the graph
            result = self.graph.invoke(initial_state)

            # Extract chunks from result
            chunks = result.get("chunks", [])

            # Handle security blocks
            if result.get("error") and not result.get("is_safe_fast", True):
                logger.warning(f"Query blocked by fast validation: {result.get('error')}")
                return []

            if result.get("error") and not result.get("is_safe_semantic", True):
                logger.warning(f"Query blocked by semantic validation: {result.get('error')}")
                return []

            logger.info(f"Search completed, found {len(chunks)} chunks")
            return chunks

        except Exception as e:
            logger.error(f"Error in SearchUseCase: {e}", exc_info=True)
            return []