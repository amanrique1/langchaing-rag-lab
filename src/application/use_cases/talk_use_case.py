import logging
from typing import Optional, Any

from src.application.graphs.rag_graph import create_qa_graph
from src.application.graphs.nodes.rag_nodes import RAGNodes
from src.application.ports.language_model import LanguageModel
from src.application.ports.chunk_store import ChunkStore
from src.application.ports.reranker import Reranker
from src.domain.guardrails.input_guard import InputGuard

logger = logging.getLogger(__name__)


class TalkUseCase:
    """
    Orchestrates the 'Chat with your Data' workflow.

    This is a stateless, one-shot Q&A use case (no conversation memory).
    For multi-turn conversations, use ChatUseCase instead.
    """

    def __init__(
        self,
        language_model: LanguageModel,
        chunk_store: ChunkStore,
        reranker: Reranker,
        input_guard: InputGuard,
        query_expander: Optional[Any] = None
    ):
        """
        Initialize the Talk Use Case with LangGraph.

        Args:
            language_model: The generative model for answering
            chunk_store: Vector store for retrieval
            reranker: Cross-encoder for reranking
            input_guard: Security gateway for validation
            query_expander: Optional query expansion strategy
        """
        self.language_model = language_model
        self.chunk_store = chunk_store
        self.reranker = reranker
        self.input_guard = input_guard
        self.query_expander = query_expander

        # Initialize nodes
        self.nodes = RAGNodes(
            language_model=language_model,
            chunk_store=chunk_store,
            reranker=reranker,
            input_guard=input_guard,
            query_expander=query_expander
        )

        # Build the graph without checkpointer (stateless)
        self.graph = create_qa_graph(nodes=self.nodes)

        logger.info("TalkUseCase initialized with LangGraph")

    def execute(
        self,
        query: str,
        top_k: int = 5,
        num_candidates: int = 20,
        use_reranking: bool = True,
        dual_collection: bool = True
    ) -> str:
        """
        Executes the question-answering pipeline (stateless, single-shot).

        This method orchestrates the full RAG pipeline:
        1. Security validation (fast + semantic)
        2. Query expansion (optional)
        3. Retrieval (content + metadata)
        4. Fusion and reranking
        5. Answer generation

        Args:
            query: The user's natural language question
            top_k: Number of chunks to provide as context to the LLM
            num_candidates: Number of chunks to retrieve before reranking
            use_reranking: Whether to use cross-encoder reranking
            dual_collection: Whether to use both content and metadata search

        Returns:
            str: The generated answer or a fallback message
        """
        try:
            # Prepare initial state (no conversation history)
            initial_state = {
                "user_id": "stateless",
                "session_id": "one_shot",
                "query": query,
                "top_k": top_k,
                "num_candidates": num_candidates,
                "use_reranking": use_reranking,
                "dual_collection": dual_collection,
                "memory_k": 0,  # No memory for stateless mode
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
                "messages": []  # Empty - no conversation history
            }

            # Invoke the graph (no config needed without checkpointer)
            result = self.graph.invoke(initial_state)

            # Extract answer
            answer = result.get("answer", "An unexpected error occurred.")

            logger.info(f"Query processed successfully")
            return answer

        except Exception as e:
            logger.error(f"Error in TalkUseCase: {e}", exc_info=True)
            return "An unexpected error occurred while generating the answer."