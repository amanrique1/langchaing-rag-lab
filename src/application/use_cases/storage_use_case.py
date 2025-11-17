from typing import List
from src.application.ports.chunk_store import ChunkStore
from src.domain.models.chunk import Chunk


class StorageUseCase:
    def __init__(self, chunk_store: ChunkStore):
        self.chunk_store = chunk_store

    def save(self, chunks: List[Chunk]) -> None:
        self.chunk_store.save(chunks)

    def search(self, query: str, top_k: int = 5) -> List[Chunk]:

        # Retrieve relevant chunks
        relevant_chunks = self.chunk_store.search(query, top_k=top_k)

        return relevant_chunks

    def clear(self) -> None:
        self.chunk_store.clear()
