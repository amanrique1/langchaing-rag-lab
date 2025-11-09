from src.application.ports.chunk_store import ChunkStore
from src.domain.models.chunk import Chunk

# --- Test Implementation of the ABC for most tests ---
class ConcreteTestChunkStore(ChunkStore):
    """A concrete implementation for testing the functionality of a ChunkStore."""
    def __init__(self):
        self.chunks: dict[str, Chunk] = {}

    def save(self, chunks: list[Chunk]):
        for chunk in chunks:
            # Assuming metadata contains a unique 'id'
            chunk_id = chunk.metadata.get("id")
            if chunk_id:
                self.chunks[chunk_id] = chunk

    def get(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)

    def delete(self, chunk_id: str):
        if chunk_id in self.chunks:
            del self.chunks[chunk_id]

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        # This is a simplistic mock search, returning all chunks up to top_k
        return list(self.chunks.values())[:top_k]

    def clear(self):
        self.chunks.clear()