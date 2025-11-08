
import pytest
import tempfile
import shutil
from src.infrastructure.adapters.chunk_stores.chroma_chunk_store import ChromaChunkStore
from src.domain.models.chunk import Chunk

@pytest.fixture
def temp_chroma_db():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_chroma_chunk_store(temp_chroma_db):
    chunk_store = ChromaChunkStore(collection_name="test_collection", persist_directory=temp_chroma_db)
    
    chunk = Chunk(content="Test chunk", metadata={"source": "test.txt", "chunk_index": 0})
    chunk_id = f"{chunk.metadata.get('source', 'doc')}_{chunk.metadata.get('chunk_index', 0)}"
    
    chunk_store.add(chunk)
    
    retrieved_chunk = chunk_store.get(chunk_id)
    assert retrieved_chunk is not None
    assert retrieved_chunk.content == chunk.content
    
    chunk_store.delete(chunk_id)
    assert chunk_store.get(chunk_id) is None
