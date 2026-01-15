from typing import Optional, List
import logging

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from src.application.ports.language_model import LanguageModel
from src.application.use_cases.search_use_case import SearchUseCase
from src.domain.guardrails.input_guard import InputGuard
from src.domain.exceptions.security_violation_exception import SecurityViolationError

logger = logging.getLogger(__name__)


class ChatUseCase:
    """
    Orchestrates conversational 'Chat with your Data' workflow using
    modern LangChain Core memory management.
    """

    def __init__(
        self,
        language_model: LanguageModel,
        search_use_case: SearchUseCase,
        input_guard: InputGuard,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        memory_k: int = 5
    ):
        """
        Initialize Chat Use Case.

        Args:
            language_model: The generative model for answering
            search_use_case: The retrieval use case
            input_guard: Security gateway for validation
            user_id: Optional user identifier
            session_id: Optional session identifier
            memory_k: Number of conversation exchanges to keep in context.
        """
        self.language_model = language_model
        self.search_use_case = search_use_case
        self.input_guard = input_guard
        self.user_id = user_id or "default_user"
        self.session_id = session_id or "default_session"
        self.memory_k = memory_k

        # --- UPDATED: Use InMemoryChatMessageHistory directly ---
        # This replaces ConversationBufferWindowMemory.
        # It lives in langchain_core, so it will always be available.
        self.chat_history = InMemoryChatMessageHistory()

        logger.info(
            f"ChatUseCase initialized for user={self.user_id}, "
            f"session={self.session_id} with Window Limit (k={memory_k})"
        )

    def execute(
        self,
        query: str,
        top_k: int = 5,
        num_candidates: int = 20,
        use_reranking: bool = True
    ) -> str:
        """Execute conversational Q&A with RAG."""
        try:
            # 1. Security validation
            if not self._validate_query(query):
                return "I cannot process this request as it may violate security policies."

            # 2. Retrieve document context using RAG
            chunks = self.search_use_case.execute(
                query=query,
                top_k=top_k,
                num_candidates=num_candidates,
                use_reranking=use_reranking
            )

            # 3. Handle no results
            if not chunks:
                response = self._handle_no_context()
                self._save_to_memory(query, response)
                return response

            # 4. Build context from retrieved chunks
            context_text = self._build_context_from_chunks(chunks)

            # 5. Generate answer with conversation history
            response = self._generate_answer_with_context(query, context_text)

            # 6. Save to memory
            self._save_to_memory(query, response)

            return response

        except SecurityViolationError as e:
            logger.warning(f"Security Alert - User: {self.user_id} - {str(e)}")
            return "I cannot fulfill this request as it violates our security policies."

        except Exception as e:
            logger.error(f"Error in ChatUseCase: {e}", exc_info=True)
            return "An unexpected error occurred while generating the answer."

    def _validate_query(self, query: str) -> bool:
        if not query or not query.strip():
            return False
        if len(query) > 10000:
            return False
        return True

    def _handle_no_context(self) -> str:
        return "I could not find relevant information in the documentation to answer your question."

    def _build_context_from_chunks(self, chunks) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Document {i}]\n{chunk.content}\n")
        return "\n".join(context_parts)

    def _generate_answer_with_context(self, query: str, context_text: str) -> str:
        """
        Generate answer using RAG context and manually windowed history.
        """
        # Build secure prompt
        safe_prompt = self.input_guard.build_safe_query(query, context_text)
        if safe_prompt is None:
            raise ValueError("Failed to build safe prompt")

        # --- UPDATED: Manual Window Logic ---
        # InMemoryChatMessageHistory stores everything. We must manually
        # slice the list to respect 'memory_k' (Window).
        all_messages = self.chat_history.messages

        # We multiply by 2 because 'k' usually implies pairs (Human + AI)
        window_limit = self.memory_k * 2

        if len(all_messages) > window_limit:
            # Take the last N messages
            history_messages = all_messages[-window_limit:]
        else:
            history_messages = all_messages

        # Build prompt with the sliced history
        full_prompt = self._build_context_aware_prompt(
            safe_prompt=safe_prompt,
            history_messages=history_messages,
            context_text=context_text
        )

        return self.language_model.get_answer(full_prompt)

    def _build_context_aware_prompt(
        self,
        safe_prompt: str,
        history_messages: List[BaseMessage],
        context_text: str
    ) -> str:
        """Build comprehensive prompt with history, context, and current query."""
        prompt_parts = [
            "You are a helpful AI assistant answering questions based on provided documentation.\n"
        ]

        if history_messages:
            prompt_parts.append("=== Recent Conversation History ===")
            for msg in history_messages:
                if isinstance(msg, HumanMessage):
                    prompt_parts.append(f"Human: {msg.content}")
                elif isinstance(msg, AIMessage):
                    prompt_parts.append(f"Assistant: {msg.content}")
            prompt_parts.append("=== End of History ===\n")

        prompt_parts.append("=== Relevant Documentation ===")
        prompt_parts.append(context_text)
        prompt_parts.append("=== End of Documentation ===\n")

        prompt_parts.append("=== Current Question ===")
        prompt_parts.append(safe_prompt)
        prompt_parts.append("\nAnswer:")

        return "\n".join(prompt_parts)

    def _save_to_memory(self, query: str, response: str) -> None:
        """Save interaction to conversation memory."""
        self.chat_history.add_user_message(query)
        self.chat_history.add_ai_message(response)

        logger.debug(f"Saved interaction to memory - Session: {self.session_id}")

    def get_conversation_history(self) -> list:
        """Get history as list of dicts."""
        return [
            {
                "role": "human" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content
            }
            for msg in self.chat_history.messages
        ]

    def clear_memory(self) -> None:
        """Clear conversation buffer."""
        self.chat_history.clear()
        logger.info(f"Memory cleared for session={self.session_id}")

    def get_memory_stats(self) -> dict:
        """Get statistics about current memory usage."""
        messages = self.chat_history.messages
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "total_messages": len(messages),
            "exchanges": len(messages) // 2,
            "memory_window_k": self.memory_k
        }