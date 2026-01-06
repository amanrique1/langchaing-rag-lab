from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document


class DocumentLoader(ABC):
    """Document loader interface for reading, parsing and loading documents."""

    @abstractmethod
    def load(self, source: str) -> List[Document]:
        """
        Load documents from a source.

        Args:
            source: Source path of the documents

        Returns:
            List of Document objects
        """
        pass
