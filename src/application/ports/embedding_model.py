from abc import ABC, abstractmethod
from typing import List


class EmbeddingModel(ABC):
    """Embedding model interface for generating embeddings."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Generates an embedding for the given text.

        Args:
            text (str): The text to be embedded.

        Returns:
            List[float]: The generated embedding.
        """
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of documents.

        Args:
            texts (List[str]): The list of texts to be embedded.

        Returns:
            List[List[float]]: A list of embeddings for the given texts.
        """
        pass
