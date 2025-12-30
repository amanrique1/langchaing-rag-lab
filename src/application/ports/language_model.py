from abc import ABC, abstractmethod
from typing import List

from src.domain.models.chunk import Chunk


class LanguageModel(ABC):
    """Language model interface for generating answers."""
    
    @abstractmethod
    def get_answer(self, prompt: str) -> str:
        """
        Generates an answer based on a pre-validated and formatted prompt.

        Args:
            prompt (str): The fully formatted prompt (Grounding instructions + Context + Question).

        Returns:
            str: The generated answer.
        """
        pass

    @abstractmethod
    async def aget_answer(self, prompt: str) -> str:
        """
        Asynchronously generates an answer based on a pre-validated prompt.

        Args:
            prompt (str): The fully formatted prompt.

        Returns:
            str: The generated answer.
        """
        pass
