from abc import ABC, abstractmethod
from typing import Optional, TypeVar, Generic

# Type variable for parser output
T = TypeVar('T')


class LanguageModel(ABC):
    """
    Abstract base class for language model implementations.

    All subclasses must support:
    - Configurable model_name and temperature
    - Optional output parsing (e.g., NumberedListOutputParser, PydanticOutputParser)
    - Both sync and async generation
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        **kwargs
    ):
        """
        Initialize the language model with standard configuration.

        Args:
            model_name (str): The name/identifier of the model to use.
            temperature (float): Sampling temperature (0.0 = deterministic, 1.0 = creative).
            **kwargs: Additional provider-specific configuration.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.config = kwargs

    @abstractmethod
    def get_answer(
        self,
        prompt: str,
        parser: Optional[any] = None
    ) -> any:
        """
        Generates an answer based on a pre-validated and formatted prompt.

        Args:
            prompt (str): The fully formatted prompt (Grounding instructions + Context + Question).
            parser (Optional[any]): Optional LangChain output parser (e.g., StrOutputParser,
                                   NumberedListOutputParser, PydanticOutputParser).
                                   If None, returns raw string output.

        Returns:
            any: The generated answer. Type depends on parser:
                 - str if parser is None or StrOutputParser
                 - List[str] if NumberedListOutputParser
                 - Pydantic model if PydanticOutputParser
                 - etc.
        """
        pass

    @abstractmethod
    async def aget_answer(
        self,
        prompt: str,
        parser: Optional[any] = None
    ) -> any:
        """
        Asynchronously generates an answer based on a pre-validated prompt.

        Args:
            prompt (str): The fully formatted prompt.
            parser (Optional[any]): Optional LangChain output parser.

        Returns:
            any: The generated answer (type depends on parser).
        """
        pass