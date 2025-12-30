import logging
from typing import List, Optional, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable

from src.application.ports.language_model import LanguageModel
from src.domain.models.chunk import Chunk

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

    def get_answer(self, prompt: str) -> str:
        """
        Synchronously generates an answer based on the provided prompt.

        Args:
            prompt (str): The fully formatted prompt.

        Returns:
            str: The generated answer from the LLM.
        """
        try:
            return self.chain.invoke(prompt)
        except Exception as e:
            return self._handle_exception(e)

    async def aget_answer(self, prompt: str) -> str:
        """
        Asynchronously generates an answer based on the provided prompt.

        Args:
            prompt (str): The fully formatted prompt.

        Returns:
            str: The generated answer.
        """
        try:
            return await self.chain.ainvoke(prompt)
        except Exception as e:
            return self._handle_exception(e)

    def _handle_exception(self, e: Exception) -> str:
        """Shared Error Handling Logic."""
        if isinstance(e, ValueError):
            logger.error(f"Configuration Error: {e}")
            return "An internal configuration error occurred."

        # Catch-all for API errors, Network issues, etc.
        logger.error("Unexpected System Error during LLM generation", exc_info=True)
        return "An unexpected error occurred while processing your request."

    