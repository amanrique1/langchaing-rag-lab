
import pytest
from src.domain.strategies.chunking_strategy import ChunkingStrategy
from src.domain.models.document import Document
from src.domain.models.chunk import Chunk

class ConcreteChunkingStrategy(ChunkingStrategy):
    def chunk(self, document: Document) -> list[Chunk]:
        return [Chunk(metadata={"document_path": document.metadata['path'], "id": "1"}, content=document.content)]

def test_chunking_strategy_raises_not_implemented_error():
    class AbstractChunkingStrategy(ChunkingStrategy):
        pass

    with pytest.raises(TypeError):
        AbstractChunkingStrategy()

    class IncompleteChunkingStrategy(ChunkingStrategy):
        def some_other_method(self):
            pass

    with pytest.raises(TypeError):
        IncompleteChunkingStrategy()

def test_concrete_chunking_strategy():
    strategy = ConcreteChunkingStrategy()
    document = Document(metadata={"path": "test_path"}, content="test content")
    chunks = strategy.chunk(document)
    assert len(chunks) == 1
    assert chunks[0].content == "test content"
    assert chunks[0].metadata["document_path"] == "test_path"
