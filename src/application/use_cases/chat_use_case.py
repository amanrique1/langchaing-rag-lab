import logging
from typing import Optional, Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from src.application.graphs.rag_graph import create_conversational_rag_graph
from src.application.graphs.nodes.rag_nodes import RAGNodes
from src.application.ports.language_model import LanguageModel
from src.application.ports.chunk_store import ChunkStore
from src.application.ports.reranker import Reranker
from src.domain.guardrails.input_guard import InputGuard

logger = logging.getLogger(__name__)


class ChatUseCase:
    """
    Orchestrates conversational 'Chat with your Data' workflow using LangGraph.

    This implementation manages conversation history in-memory without checkpointing,
    suitable for single-process applications or stateless deployments with external
    session management.
    """

    def __init__(
        self,
        language_model: LanguageModel,
        chunk_store: ChunkStore,
        reranker: Reranker,
        input_guard: InputGuard,
        query_expander: Optional[Any] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        memory_k: int = 5
    ):
        """
        Initialize Chat Use Case with LangGraph (no checkpointer).

        Args:
            language_model: The generative model for answering
            chunk_store: Vector store for retrieval
            reranker: Cross-encoder for reranking
            input_guard: Security gateway for validation
            query_expander: Optional query expansion strategy
            user_id: Optional user identifier
            session_id: Optional session identifier
            memory_k: Number of conversation exchanges to keep in context
        """
        self.user_id = user_id or "default_user"
        self.session_id = session_id or "default_session"
        self.memory_k = memory_k

        # Initialize nodes
        self.nodes = RAGNodes(
            language_model=language_model,
            chunk_store=chunk_store,
            reranker=reranker,
            input_guard=input_guard,
            query_expander=query_expander
        )

        # Build the graph WITHOUT checkpointer
        self.graph = create_conversational_rag_graph(
            nodes=self.nodes
        )

        # Manual in-memory conversation history
        self._conversation_history: List[BaseMessage] = []

        logger.info(
            f"ChatUseCase initialized for user={self.user_id}, "
            f"session={self.session_id} with memory_k={memory_k}"
        )

    def execute(
        self,
        query: str,
        top_k: int = 5,
        num_candidates: int = 20,
        use_reranking: bool = True,
        dual_collection: bool = True
    ) -> str:
        """
        Execute conversational Q&A with RAG.

        Args:
            query: User's question
            top_k: Number of final chunks to use for generation
            num_candidates: Number of candidates to retrieve before reranking
            use_reranking: Whether to use cross-encoder reranking
            dual_collection: Whether to use both content and metadata search

        Returns:
            Generated answer string
        """
        try:
            # Prepare initial state with conversation history
            initial_state = {
                "user_id": self.user_id,
                "session_id": self.session_id,
                "query": query,
                "top_k": top_k,
                "num_candidates": num_candidates,
                "use_reranking": use_reranking,
                "dual_collection": dual_collection,
                "memory_k": self.memory_k,
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
                "messages": self._conversation_history.copy()  # Pass existing history
            }

            # Invoke the graph
            result = self.graph.invoke(initial_state)

            # Extract answer
            answer = result.get("answer", "An unexpected error occurred.")

            # Update conversation history with new messages
            # The graph returns new messages appended via operator.add
            new_messages = result.get("messages", [])

            # Since operator.add appends, new_messages will contain old + new
            # We need to extract only the new ones (last 2: Human + AI)
            if len(new_messages) > len(self._conversation_history):
                self._conversation_history = new_messages

            logger.info(
                f"Query processed for user={self.user_id}, "
                f"session={self.session_id}, "
                f"total_messages={len(self._conversation_history)}"
            )

            return answer

        except Exception as e:
            logger.error(f"Error in ChatUseCase: {e}", exc_info=True)
            return "An unexpected error occurred while generating the answer."

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history for the current session.

        Returns:
            List of message dictionaries with 'role' and 'content'
        """
        return [
            {
                "role": "human" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content
            }
            for msg in self._conversation_history
        ]

    def clear_memory(self) -> None:
        """Clear conversation history for the current session."""
        self._conversation_history.clear()
        logger.info(f"Memory cleared for session={self.session_id}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about current memory usage."""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "total_messages": len(self._conversation_history),
            "exchanges": len(self._conversation_history) // 2,
            "memory_window_k": self.memory_k
        }