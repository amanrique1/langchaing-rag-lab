from abc import ABC, abstractmethod
from typing import List
from src.domain.models.chunk import Chunk


class ChunkStore(ABC):
    @abstractmethod
    def add(self, chunk: Chunk):
        pass

    @abstractmethod
    def get(self, chunk_id: str) -> Chunk | None:
        pass

    @abstractmethod
    def delete(self, chunk_id: str):
        pass

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        pass
