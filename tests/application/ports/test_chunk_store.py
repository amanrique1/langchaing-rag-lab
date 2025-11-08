import pytest
from src.application.ports.chunk_store import ChunkStore
from src.domain.models.chunk import Chunk

class ConcreteChunkStore(ChunkStore):
    def __init__(self):
        self.chunks = {}

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
        # This is a simplistic mock search, returning all chunks
        return list(self.chunks.values())[:top_k]

    def clear(self):
        self.chunks = {}

def test_chunk_store_raises_not_implemented_error():
    class AbstractChunkStore(ChunkStore):
        pass

    with pytest.raises(TypeError):
        AbstractChunkStore()

    class IncompleteChunkStore(ChunkStore):
        def some_other_method(self):
            pass

    with pytest.raises(TypeError):
        IncompleteChunkStore()

def test_concrete_chunk_store():
    store = ConcreteChunkStore()
    chunk1 = Chunk(metadata={"id": "1", "document_path": "path1"}, content="test1")
    chunk2 = Chunk(metadata={"id": "2", "document_path": "path2"}, content="test2")

    store.save([chunk1, chunk2])
    assert len(store.chunks) == 2
    
    retrieved_chunk = store.get("1")
    assert retrieved_chunk == chunk1

    search_results = store.search([0.1, 0.2])
    assert len(search_results) == 2

    store.delete("1")
    assert len(store.chunks) == 1
    assert store.get("1") is None

    store.clear()
    assert len(store.chunks) == 0