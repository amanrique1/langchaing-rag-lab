import logging
from typing import Dict, Any, Optional, List
from collections import defaultdict

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from src.application.graphs.states.rag_state import RAGState
from src.application.ports.language_model import LanguageModel
from src.application.ports.chunk_store import ChunkStore
from src.application.ports.reranker import Reranker
from src.domain.guardrails.input_guard import InputGuard
from src.domain.exceptions.security_violation_exception import SecurityViolationError
from src.domain.models.search_result import SearchResult

logger = logging.getLogger(__name__)


class RAGNodes:
    """
    Orchestrates the Retrieval-Augmented Generation (RAG) pipeline nodes,
    with conversational memory support.
    """

    def __init__(
        self,
        language_model: Optional[LanguageModel],
        chunk_store: ChunkStore,
        reranker: Reranker,
        input_guard: InputGuard,
        query_expander: Optional[Any] = None
    ):
        """
        Initialize RAG nodes.

        Args:
            language_model: Language model for generation (None for search-only mode)
            chunk_store: Vector store for retrieval
            reranker: Cross-encoder for reranking
            input_guard: Security validation
            query_expander: Optional query expansion strategy
        """
        self.language_model = language_model
        self.chunk_store = chunk_store
        self.reranker = reranker
        self.input_guard = input_guard
        self.query_expander = query_expander

    def fast_validation(self, state: RAGState) -> Dict[str, Any]:
        """Layer 1: Regex-based security checks."""
        logger.debug(f"Fast validation for: {state['query']}")
        try:
            self.input_guard._check_fast_rules(state["query"])
            return {"is_safe_fast": True, "error": None}
        except SecurityViolationError as e:
            logger.warning(f"Fast validation fail: {e}")
            return {
                "is_safe_fast": False,
                "error": str(e),
                "answer": "Blocked by security policy (Regex).",
                "messages": [
                    HumanMessage(content=state["query"]),
                    AIMessage(content="Blocked by security policy (Regex).")
                ]
            }

    def semantic_validation(self, state: RAGState) -> Dict[str, Any]:
        """Layer 2: AI-based security checks."""
        if not state.get("is_safe_fast", True):
            return {}

        logger.debug(f"Semantic validation for: {state['query']}")
        try:
            self.input_guard._check_semantic_intent(state["query"])
            return {"is_safe_semantic": True, "error": None}
        except SecurityViolationError as e:
            logger.warning(f"Semantic validation fail: {e}")
            return {
                "is_safe_semantic": False,
                "error": str(e),
                "answer": "Blocked by security policy (AI Guard).",
                "messages": [
                    HumanMessage(content=state["query"]),
                    AIMessage(content="Blocked by security policy (AI Guard).")
                ]
            }
        except Exception as e:
            logger.error(f"Semantic validation error: {e}")
            return {
                "is_safe_semantic": True,
                "error": f"Semantic validation skipped: {str(e)}"
            }

    def expand_query(self, state: RAGState) -> Dict[str, Any]:
        """Layer 3: Expands the query into multiple variations."""
        if not state.get("is_safe_semantic", True) or not self.query_expander:
            return {"expanded_queries": [state["query"]]}

        logger.info(f"Expanding query: {state['query']}")
        try:
            expanded = self.query_expander.expand(state["query"])
            if state["query"] not in expanded:
                expanded.insert(0, state["query"])
            return {"expanded_queries": expanded}
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return {"expanded_queries": [state["query"]]}

    def content_search(self, state: RAGState) -> Dict[str, Any]:
        """Layer 3a: Content-based (vector/semantic) retrieval."""
        if not state.get("is_safe_semantic", True):
            return {}

        queries = state.get("expanded_queries") or [state["query"]]
        num_candidates = state.get("num_candidates", 20)

        logger.debug(f"Executing content search for {len(queries)} queries")

        all_results = []
        seen_ids = set()

        try:
            for q in queries:
                results = self.chunk_store.search(
                    query=q,
                    top_k=num_candidates,
                    mode="content"
                )
                for r in results:
                    if r.chunk.chunk_id not in seen_ids:
                        all_results.append(r)
                        seen_ids.add(r.chunk.chunk_id)

            return {"content_results": all_results[:num_candidates]}
        except Exception as e:
            logger.error(f"Content search failed: {e}")
            return {"content_results": []}

    def metadata_search(self, state: RAGState) -> Dict[str, Any]:
        """Layer 3b: Metadata-based (keyword/filter) retrieval."""
        if not state.get("is_safe_semantic", True) or not state.get("dual_collection", True):
            return {"metadata_results": []}

        queries = state.get("expanded_queries") or [state["query"]]
        num_candidates = state.get("num_candidates", 20)

        logger.debug(f"Executing metadata search for {len(queries)} queries")

        all_results = []
        seen_ids = set()

        try:
            for q in queries:
                results = self.chunk_store.search(
                    query=q,
                    top_k=num_candidates,
                    mode="metadata"
                )
                for r in results:
                    if r.chunk.chunk_id not in seen_ids:
                        all_results.append(r)
                        seen_ids.add(r.chunk.chunk_id)

            return {"metadata_results": all_results[:num_candidates]}
        except Exception as e:
            logger.error(f"Metadata search failed: {e}")
            return {"metadata_results": []}

    def fusion(self, state: RAGState) -> Dict[str, Any]:
        """Layer 3c: Combines results using Reciprocal Rank Fusion (RRF)."""
        if state.get("answer"):
            return {}

        content_results = state.get("content_results") or []
        metadata_results = state.get("metadata_results") or []

        if not content_results and not metadata_results:
            no_results_msg = "I could not find any relevant information."
            return {
                "candidates": [],
                "answer": no_results_msg,
                "messages": [
                    HumanMessage(content=state["query"]),
                    AIMessage(content=no_results_msg)
                ]
            }

        logger.debug(f"Fusing {len(content_results)} content and {len(metadata_results)} metadata results")

        rrf_k = 60
        scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, SearchResult] = {}

        for rank, res in enumerate(content_results, 1):
            cid = res.chunk.chunk_id
            scores[cid] += 1.0 / (rrf_k + rank)
            chunk_map[cid] = res

        for rank, res in enumerate(metadata_results, 1):
            cid = res.chunk.chunk_id
            scores[cid] += 1.0 / (rrf_k + rank)
            if cid not in chunk_map:
                chunk_map[cid] = res

        fused = []
        for cid, score in scores.items():
            res = chunk_map[cid]
            fused.append(SearchResult(
                chunk=res.chunk,
                score=score,
                retrieval_method=f"rrf_fusion({res.retrieval_method})",
                rank=None
            ))

        fused.sort(key=lambda x: x.score, reverse=True)
        return {"candidates": fused}

    def reranking(self, state: RAGState) -> Dict[str, Any]:
        """Layer 4: Re-ranks candidates using Cross-Encoder."""
        if state.get("answer") or not state.get("candidates"):
            return {}

        logger.debug(f"Reranking {len(state['candidates'])} candidates")

        if state.get("use_reranking", True) and self.reranker:
            results = self.reranker.rerank(
                state["query"],
                state["candidates"],
                state.get("top_k", 5)
            )
        else:
            results = state["candidates"][:state.get("top_k", 5)]

        chunks = [r.chunk for r in results]
        context_text = "\n\n".join([
            f"[Document {i+1}]\n{chunk.content}"
            for i, chunk in enumerate(chunks)
        ])

        return {"chunks": chunks, "context_text": context_text}

    def generation(self, state: RAGState) -> Dict[str, Any]:
        """
        Layer 5: Generates the final answer using the LLM.

        Note: This node is skipped in search-only mode.
        """
        if state.get("answer"):
            return {}

        # Check if language model is available
        if self.language_model is None:
            logger.warning("Generation node called but no language model available")
            return {"answer": "Generation not available in search-only mode"}

        logger.debug("Generating answer with conversation context")

        context = state.get("context_text", "")
        query = state["query"]

        # Apply memory window to conversation history
        history_messages = self._apply_memory_window(
            state.get("messages", []),
            state.get("memory_k", 5)
        )

        # Build the complete prompt
        final_prompt = self._build_prompt_with_history(
            query=query,
            context=context,
            history_messages=history_messages
        )

        # Generate answer
        answer = self.language_model.get_answer(final_prompt)

        logger.info(f"Generated answer for session={state.get('session_id')}")

        return {
            "answer": answer,
            "messages": [
                HumanMessage(content=query),
                AIMessage(content=answer)
            ]
        }

    def _apply_memory_window(
        self,
        messages: List[BaseMessage],
        memory_k: int
    ) -> List[BaseMessage]:
        """Apply sliding window to conversation history."""
        if not messages:
            return []

        # memory_k is the number of exchanges (pairs)
        # So we need memory_k * 2 messages
        window_limit = memory_k * 2

        if len(messages) > window_limit:
            return messages[-window_limit:]
        return messages

    def _build_prompt_with_history(
        self,
        query: str,
        context: str,
        history_messages: List[BaseMessage]
    ) -> str:
        """Build comprehensive prompt with history and context."""
        prompt_parts = [
            "You are a helpful AI assistant answering questions based on provided documentation.\n"
        ]

        # Add conversation history if present
        if history_messages:
            prompt_parts.append("=== Recent Conversation History ===")
            for msg in history_messages:
                if isinstance(msg, HumanMessage):
                    prompt_parts.append(f"Human: {msg.content}")
                elif isinstance(msg, AIMessage):
                    prompt_parts.append(f"Assistant: {msg.content}")
            prompt_parts.append("=== End of History ===\n")

        # Add retrieved context
        if context:
            prompt_parts.append("=== Relevant Documentation ===")
            prompt_parts.append(context)
            prompt_parts.append("=== End of Documentation ===\n")

        # Add current query
        prompt_parts.append("=== Current Question ===")
        prompt_parts.append(query)
        prompt_parts.append("\nAnswer:")

        return "\n".join(prompt_parts)