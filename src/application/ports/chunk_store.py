from abc import ABC, abstractmethod
from src.domain.models.chunk import Chunk


class ChunkStore(ABC):
    @abstractmethod
    def save(self, chunks: list[Chunk]):
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

    @abstractmethod
    def clear(self):
        pass
