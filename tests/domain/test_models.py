from src.domain.models.document import Document
from src.domain.models.chunk import Chunk


def test_document_creation():
    doc = Document(content="Test content", metadata={"source": "test.txt"})
    assert doc.content == "Test content"
    assert doc.metadata == {"source": "test.txt"}


def test_chunk_creation():
    chunk = Chunk(content="Test chunk", metadata={"doc_id": "123", "chunk_index": 0})
    assert chunk.content == "Test chunk"
    assert chunk.metadata == {"doc_id": "123", "chunk_index": 0}
