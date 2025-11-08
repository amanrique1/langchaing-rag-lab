from abc import ABC, abstractmethod
from typing import List
from domain.models.document import Document


class DocumentLoader(ABC):
    @abstractmethod
    def load(self, source: str) -> List[Document]:
        pass
