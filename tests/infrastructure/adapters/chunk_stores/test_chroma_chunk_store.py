import pytest
from unittest.mock import MagicMock, patch, call
from src.infrastructure.adapters.chunk_stores.chroma_chunk_store import ChromaChunkStore, DEFAULT_COLLECTION_NAME
from src.domain.models.chunk import Chunk
from src.domain.models.search_result import SearchResult

# --- Fixtures ---

@pytest.fixture
def mock_embedding_model():
    """Provides a MagicMock for the EmbeddingModel dependency."""
    mock = MagicMock()
    mock.embed_query.return_value = [0.1, 0.2, 0.3] # Dummy embedding
    return mock

@pytest.fixture
def mock_chroma_class():
    """Mocks the Chroma class where it's used to prevent actual disk I/O."""
    with patch('src.infrastructure.adapters.chunk_stores.chroma_chunk_store.Chroma') as mock_class:
        yield mock_class

@pytest.fixture
def chroma_chunk_store(mock_embedding_model, mock_chroma_class):
    """Creates a standard ChromaChunkStore instance for testing (dual collection by default)."""
    # We mock the return values of Chroma() to be distinct for content and metadata
    mock_content_store = MagicMock()
    mock_metadata_store = MagicMock()
    mock_chroma_class.side_effect = [mock_content_store, mock_metadata_store]
    
    store = ChromaChunkStore(
        collection_name="test_collection",
        embedding_model=mock_embedding_model
    )
    # Trigger lazy loading
    _ = store.content_collection
    _ = store.metadata_collection
    
    # Attach mocks to the store instance for easier assertion in tests
    store._content_store_mock = mock_content_store
    store._metadata_store_mock = mock_metadata_store
    
    return store

# --- Tests ---

def test_initialization_with_collection_name(mock_embedding_model):
    """Test that ChromaChunkStore initializes correctly with a given collection name."""
    store = ChromaChunkStore(collection_name="test_collection", embedding_model=mock_embedding_model)
    assert store.collection_name == "test_collection"
    assert store.persist_directory.endswith("chroma_db")
    assert store._content_vector_store is None
    assert store._metadata_vector_store is None

def test_initialization_with_default_collection_name(mock_embedding_model):
    """Test initialization uses the default collection name when None is provided."""
    store = ChromaChunkStore(collection_name=None, embedding_model=mock_embedding_model)
    assert store.collection_name == DEFAULT_COLLECTION_NAME

def test_vector_store_lazy_initialization(mock_embedding_model, mock_chroma_class):
    """Test that vector stores are lazily initialized."""
    store = ChromaChunkStore(collection_name="test_collection", embedding_model=mock_embedding_model)
    
    mock_chroma_class.assert_not_called()

    # Access content store
    _ = store.content_collection
    assert mock_chroma_class.call_count == 1
    
    # Access metadata store
    _ = store.metadata_collection
    assert mock_chroma_class.call_count == 2

def test_save_chunks(chroma_chunk_store, mock_embedding_model):
    """Test saving chunks involves both content and metadata stores."""
    # Ensure chunk_id is consistent for asserting IDs
    chunk = Chunk(metadata={"filename": "doc1.txt", "chunk_index": 0}, content="content1")
    chunk.chunk_id = "path1_0"
    chunks = [chunk]
    
    chroma_chunk_store.save(chunks)

    # Content store check
    chroma_chunk_store._content_store_mock.add_documents.assert_called_once()
    _, kwargs = chroma_chunk_store._content_store_mock.add_documents.call_args
    assert len(kwargs['documents']) == 1
    assert kwargs['ids'] == ["path1_0"]
    assert kwargs['documents'][0].page_content == "content1"

    # Metadata store check
    chroma_chunk_store._metadata_store_mock.add_documents.assert_called_once()
    _, kwargs = chroma_chunk_store._metadata_store_mock.add_documents.call_args
    assert len(kwargs['documents']) == 1
    assert kwargs['ids'] == ["path1_0_meta"]
    # Metadata summary content
    assert "File: doc1.txt" in kwargs['documents'][0].page_content

def test_delete_chunk(chroma_chunk_store):
    """Test deleting a chunk removes it from both stores."""
    chroma_chunk_store.delete("id_123")
    
    chroma_chunk_store._content_store_mock.delete.assert_called_once_with(
        ids=["id_123"], where=None, where_document=None
    )
    chroma_chunk_store._metadata_store_mock.delete.assert_called_once_with(
        ids=["id_123_meta"], where=None, where_document=None
    )

def test_search_content_mode(chroma_chunk_store):
    """Test search in content mode."""
    # Mock return from content store
    mock_doc = MagicMock()
    mock_doc.page_content = "searched_content"
    mock_doc.metadata = {"source": "searched_source"}
    # similarity_search_with_score returns list of (doc, score) tuples
    # 0.5 raw score -> 1/(1+0.5) = 0.666...
    chroma_chunk_store._content_store_mock.similarity_search_with_score.return_value = [(mock_doc, 0.5)]

    results = chroma_chunk_store.search(query="test query", mode="content")
    
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].chunk.content == "searched_content"
    assert results[0].score == pytest.approx(0.6666666666666666)
    
    chroma_chunk_store._content_store_mock.similarity_search_with_score.assert_called_once()
    chroma_chunk_store._metadata_store_mock.similarity_search_with_score.assert_not_called()

def test_search_metadata_mode(chroma_chunk_store):
    """Test search in metadata mode."""
    mock_doc = MagicMock()
    mock_doc.page_content = "source: searched_source | ..." 
    mock_doc.metadata = {"source": "searched_source", "original_content": "real content"} # Assuming storing original
    
    # Mock retrieval logic if needed, but for now checking routing
    chroma_chunk_store._metadata_store_mock.similarity_search_with_score.return_value = []

    results = chroma_chunk_store.search(query="test query", mode="metadata")
    
    chroma_chunk_store._metadata_store_mock.similarity_search_with_score.assert_called_once()
    chroma_chunk_store._content_store_mock.similarity_search_with_score.assert_not_called()

def test_clear_collection(chroma_chunk_store):
    """Test clearing calls delete_collection on both stores."""
    chroma_chunk_store.clear()
    
    chroma_chunk_store._content_store_mock.delete_collection.assert_called_once()
    chroma_chunk_store._metadata_store_mock.delete_collection.assert_called_once()
    
    assert chroma_chunk_store._content_vector_store is None
    assert chroma_chunk_store._metadata_vector_store is None

@patch('src.infrastructure.adapters.chunk_stores.chroma_chunk_store.shutil.rmtree')
@patch('src.infrastructure.adapters.chunk_stores.chroma_chunk_store.os.path.exists')
def test_clear_deletes_directory_when_collection_name_is_none(mock_exists, mock_rmtree, mock_embedding_model):
    """Test clear() deletes the directory when collection_name is manually set to None."""
    mock_exists.return_value = True
    
    store = ChromaChunkStore(embedding_model=mock_embedding_model)
    store.collection_name = None
    
    store.clear()
    
    mock_exists.assert_any_call(store.persist_directory)
    mock_rmtree.assert_called_once_with(store.persist_directory)