
import pytest
from unittest.mock import MagicMock
from src.domain.services.chunking_service import ChunkingService
from src.domain.models.document import Document
from src.domain.models.chunk import Chunk

@pytest.fixture
def sample_documents():
    return [
        Document(content="First document.", metadata={"source": "1.txt"}),
        Document(content="Second document.", metadata={"source": "2.txt"})
    ]

def test_chunking_service(sample_documents):
    mock_strategy = MagicMock()
    mock_strategy.chunk.return_value = [Chunk(content="chunk1", metadata={})]
    
    service = ChunkingService(chunking_strategy=mock_strategy)
    chunks = service.chunk_documents(sample_documents)
    
    mock_strategy.chunk.assert_called_once_with(sample_documents)
    assert len(chunks) == 1
    assert chunks[0].content == "chunk1"
