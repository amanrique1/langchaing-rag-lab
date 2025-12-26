import logging
from typing import List, Optional, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable

from src.application.ports.language_model import LanguageModel
from src.domain.guardrails.input_guard import InputGuard
from src.domain.models.chunk import Chunk
from src.domain.exceptions.security_violation_exception import SecurityViolationError

logger = logging.getLogger(__name__)

class GoogleGenAILanguageModel(LanguageModel):
    """
    An implementation of the LanguageModel port using Google's Gemini models via LangChain.

    This class handles the orchestration of:
    1. Aggregating context chunks.
    2. Invoking the InputGuard to validate inputs and construct the secure prompt.
    3. Executing the request against Google's Generative AI.
    4. Handling security violations and runtime errors gracefully.

    Attributes:
        guard (InputGuard): The guardrail instance used for validation and prompt construction.
        model (ChatGoogleGenerativeAI): The configured LangChain Google chat model.
        parser (StrOutputParser): parser to convert message objects to string.
    """

    def __init__(
        self, 
        guard: InputGuard, 
        model_name: str = "gemini-2.5-flash", 
        temperature: float = 0.0,
        **kwargs: Any
    ):
        """
        Initializes the Google GenAI Language Model wrapper.

        Args:
            guard (InputGuard): An instance of InputGuard configured with validation rules 
                                and the specific system prompt template.
            model_name (str, optional): The name of the Gemini model to use. 
                                        Defaults to "gemini-2.5-flash".
            temperature (float, optional): The temperature for generation (0.0 is deterministic). 
                                           Defaults to 0.0.
            **kwargs: Additional keyword arguments passed directly to the 
                      ChatGoogleGenerativeAI constructor (e.g., max_retries, timeout).
        """
        self.guard = guard
        
        # Initialize the model with configuration passthrough
        self.model = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=temperature,
            **kwargs
        )
        
        self.parser = StrOutputParser()
        
        # Note: We do not use a ChatPromptTemplate here because InputGuard.build_safe_query 
        # returns a fully formed string containing System Instructions, Context, and Question.
        self.chain: RunnableSerializable = self.model | self.parser

    def get_answer(self, question: str, context_chunks: List[Chunk]) -> str:
        """
        Synchronously generates an answer to a question based on provided context.

        This method aggregates the content from the provided Chunk objects, passes them
        to the InputGuard to generate a secure, formatted prompt, and then queries the model.

        Args:
            question (str): The user's inquiry.
            context_chunks (List[Chunk]): A list of domain-specific context chunks retrieved 
                                          from the vector store.

        Returns:
            str: The generated answer from the LLM, or a refusal message if a security 
                 violation occurs.

        Raises:
            Exception: Logs internal errors but returns a generic error message to the user 
                       to prevent information leakage.
        """
        try:
            safe_prompt = self._prepare_safe_prompt(question, context_chunks)
            return self.chain.invoke(safe_prompt)
        except Exception as e:
            return self._handle_exception(e)

    async def aget_answer(self, question: str, context_chunks: List[Chunk]) -> str:
        """
        Asynchronously generates an answer to a question based on provided context.

        Recommended for use in web applications (FastAPI/Django) to prevent blocking
        the event loop during LLM API calls.

        Args:
            question (str): The user's inquiry.
            context_chunks (List[Chunk]): A list of domain-specific context chunks.

        Returns:
            str: The generated answer or refusal message.
        """
        try:
            safe_prompt = self._prepare_safe_prompt(question, context_chunks)
            return await self.chain.ainvoke(safe_prompt)
        except Exception as e:
            return self._handle_exception(e)

    def _prepare_safe_prompt(self, question: str, context_chunks: List[Chunk]) -> str:
        """
        Internal helper to format context and invoke the InputGuard.

        Args:
            question (str): The user question.
            context_chunks (List[Chunk]): The context objects.

        Returns:
            str: The fully validated and formatted prompt string.
        
        Raises:
            SecurityViolationError: If the InputGuard detects malicious intent.
            ValueError: If the InputGuard fails to build the template (e.g. missing keys).
        """
        # 1. Format Context
        if not context_chunks:
            logger.warning("Processing query with empty context chunks.")
            context_text = "No technical context provided."
        else:
            context_text = "\n\n".join([chunk.content for chunk in context_chunks])

        # 2. Build Safe Query
        # This step performs validation (REGEX + LLAMA_GUARD) and Template formatting.
        # It raises SecurityViolationError if validation fails.
        safe_prompt = self.guard.build_safe_query(question, context_text)

        # 3. Handle Template Errors
        # InputGuard.build_safe_query returns None if a KeyError occurs in .format()
        if safe_prompt is None:
            raise ValueError("InputGuard failed to generate prompt due to template mismatch.")
            
        return safe_prompt

    def _handle_exception(self, e: Exception) -> str:
        """Shared Error Handling Logic."""
        if isinstance(e, SecurityViolationError):
            logger.warning(f"Security Alert [{e.violation_type}]: {str(e)}")
            return (
                "I cannot fulfill this request as it violates our security policies "
                "regarding safe and appropriate content."
            )
        
        if isinstance(e, ValueError):
            logger.error(f"Configuration Error: {e}")
            return "An internal configuration error occurred."

        # Catch-all for API errors, Network issues, etc.
        logger.error("Unexpected System Error during LLM generation", exc_info=True)
        return "An unexpected error occurred while processing your request."

    