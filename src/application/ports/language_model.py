from abc import ABC, abstractmethod
from typing import List

from src.domain.models.chunk import Chunk


class LanguageModel(ABC):
    @abstractmethod
    def get_answer(self, question: str, context_chunks: List[Chunk]) -> str:
        """
        Generates an answer to a question based on the provided context chunks.

        Args:
            question (str): The question to be answered.
            context_chunks (List[Chunk]): A list of context chunks to be used for answering the question.

        Returns:
            str: The generated answer.
        """
        pass
