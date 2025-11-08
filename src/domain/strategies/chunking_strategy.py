from abc import ABC, abstractmethod
from typing import List
from domain.models.document import Document
from domain.models.chunk import Chunk


class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Chunk]:
        pass
