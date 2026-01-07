import logging
from typing import Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.runnables import RunnableSerializable

from src.application.ports.language_model import LanguageModel

logger = logging.getLogger(__name__)

class GoogleGenAILanguageModel(LanguageModel):
    """
    An implementation of the LanguageModel port using Google's Gemini models via LangChain.

    This class handles the execution of a pre-validated "safe prompt"
    against Google's Generative AI. It no longer handles guardrails
    internally, as that responsibility has been moved to the Use Case layer.

    Attributes:
        model (ChatGoogleGenerativeAI): The configured LangChain Google chat model.
        parser (StrOutputParser): parser to convert message objects to string.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        **kwargs: Any
    ):
        """
        Initializes the Google GenAI Language Model wrapper.

        Args:
            model_name (str, optional): The name of the Gemini model to use.
                                        Defaults to "gemini-2.5-flash".
            temperature (float, optional): The temperature for generation (0.0 is deterministic).
                                           Defaults to 0.0.
            **kwargs: Additional keyword arguments passed directly to the
                      ChatGoogleGenerativeAI constructor (e.g., max_retries, timeout).
        """

        # Call parent constructor to set standard attributes
        super().__init__(model_name=model_name, temperature=temperature, **kwargs)

        # Initialize the LangChain model
        self.model = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            **self.config
        )

        # Default parser for string output
        self._default_parser = StrOutputParser()

        logger.info(
            f"GoogleGenAILanguageModel initialized: model={self.model_name}, "
            f"temperature={self.temperature}"
        )

    def get_answer(
        self,
        prompt: str,
        parser: Optional[BaseOutputParser] = None
    ) -> Any:
        """
        Synchronously generates an answer based on the provided prompt.

        Args:
            prompt (str): The fully formatted prompt.
            parser (Optional[BaseOutputParser]): Optional output parser.
                                                 If None, uses StrOutputParser.

        Returns:
            Any: The generated answer. Type depends on the parser used.

        Raises:
            Exception: Wrapped as user-friendly error message.
        """
        try:
            # Build chain with appropriate parser
            chain = self._build_chain(parser)
            return chain.invoke(prompt)
        except Exception as e:
            return self._handle_exception(e)

    async def aget_answer(
        self,
        prompt: str,
        parser: Optional[BaseOutputParser] = None
    ) -> Any:
        """
        Asynchronously generates an answer based on the provided prompt.

        Args:
            prompt (str): The fully formatted prompt.
            parser (Optional[BaseOutputParser]): Optional output parser.

        Returns:
            Any: The generated answer. Type depends on the parser used.

        Raises:
            Exception: Wrapped as user-friendly error message.
        """
        try:
            # Build chain with appropriate parser
            chain = self._build_chain(parser)
            return await chain.ainvoke(prompt)
        except Exception as e:
            return self._handle_exception(e)

    def _build_chain(self, parser: Optional[BaseOutputParser] = None) -> RunnableSerializable:
        """
        Builds a LangChain runnable with the specified parser.

        Args:
            parser (Optional[BaseOutputParser]): The output parser to use.
                                                 If None, uses default StrOutputParser.

        Returns:
            RunnableSerializable: The configured chain (model | parser).
        """
        # Use provided parser or fall back to default
        active_parser = parser if parser is not None else self._default_parser

        # Build and return the chain
        return self.model | active_parser

    def _handle_exception(self, e: Exception) -> str:
        """
        Centralized error handling for LLM operations.

        Args:
            e (Exception): The exception that was raised.

        Returns:
            str: User-friendly error message.
        """
        if isinstance(e, ValueError):
            logger.error(f"Configuration Error: {e}")
            return "An internal configuration error occurred."

        if isinstance(e, TimeoutError):
            logger.error(f"Timeout Error: {e}")
            return "The request timed out. Please try again."

        # Catch-all for API errors, network issues, etc.
        logger.error(
            f"Unexpected error during LLM generation with model={self.model_name}",
            exc_info=True
        )
        return "An unexpected error occurred while processing your request."