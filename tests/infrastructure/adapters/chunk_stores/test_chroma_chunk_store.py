import pytest
from unittest.mock import MagicMock, patch

from src.infrastructure.adapters.chunk_stores.chroma_chunk_store import ChromaChunkStore, DEFAULT_COLLECTION_NAME
from src.domain.models.chunk import Chunk


# --- Fixtures ---

@pytest.fixture
def mock_embedding_model():
    """Provides a MagicMock for the EmbeddingModel dependency."""
    return MagicMock()

@pytest.fixture
def mock_chroma_class():
    """Mocks the Chroma class where it's used to prevent actual disk I/O."""
    with patch('src.infrastructure.adapters.chunk_stores.chroma_chunk_store.Chroma') as mock_class:
        yield mock_class

@pytest.fixture
def mock_vector_store(mock_chroma_class):
    """Provides a mock for the Chroma instance (the vector store itself)."""
    return mock_chroma_class.return_value

@pytest.fixture
def chroma_chunk_store(mock_embedding_model):
    """Creates a standard ChromaChunkStore instance for testing."""
    return ChromaChunkStore(
        collection_name="test_collection",
        embedding_model=mock_embedding_model
    )


# --- Tests ---

def test_initialization_raises_error_without_embedding_model():
    """Test that __init__ raises a ValueError if no embedding model is provided."""
    with pytest.raises(ValueError, match="An embedding model must be provided."):
        ChromaChunkStore(collection_name="test_collection")

def test_initialization_with_collection_name(mock_embedding_model):
    """Test that ChromaChunkStore initializes correctly with a given collection name."""
    store = ChromaChunkStore(collection_name="test_collection", embedding_model=mock_embedding_model)
    assert store.collection_name == "test_collection"
    assert store.persist_directory == "./chroma_db"
    assert store._vector_store is None  # Check that it's not initialized yet

def test_initialization_with_default_collection_name(mock_embedding_model):
    """Test initialization uses the default collection name when None is provided."""
    store = ChromaChunkStore(collection_name=None, embedding_model=mock_embedding_model)
    assert store.collection_name == DEFAULT_COLLECTION_NAME
    assert store.persist_directory == "./chroma_db"

def test_vector_store_lazy_initialization(mock_embedding_model, mock_chroma_class):
    """Test that the vector_store is lazily initialized on first access."""
    store = ChromaChunkStore(collection_name="test_collection", embedding_model=mock_embedding_model)
    
    # Check that it hasn't been called yet
    mock_chroma_class.assert_not_called()

    # Access the property to trigger initialization
    _ = store.vector_store
    
    # Assert it was initialized once with correct parameters
    mock_chroma_class.assert_called_once_with(
        collection_name="test_collection",
        persist_directory="./chroma_db",
        embedding_function=mock_embedding_model,
    )
    
    # Access again and ensure it's not initialized a second time
    _ = store.vector_store
    mock_chroma_class.assert_called_once()


def test_save_chunks(chroma_chunk_store, mock_vector_store):
    """Test saving chunks to the store generates correct documents and IDs."""
    chunks = [
        Chunk(metadata={"source": "path1", "chunk_index": 0}, content="content1"),
        Chunk(metadata={"source": "path2", "chunk_index": 1}, content="content2"),
    ]
    chroma_chunk_store.save(chunks)

    mock_vector_store.add_documents.assert_called_once()
    _, kwargs = mock_vector_store.add_documents.call_args
    
    assert len(kwargs['documents']) == 2
    assert kwargs['ids'] == ["path1_0", "path2_1"]
    assert kwargs['documents'][0].page_content == "content1"
    assert kwargs['documents'][1].page_content == "content2"

def test_save_chunks_with_missing_metadata(chroma_chunk_store, mock_vector_store):
    """Test saving a chunk with missing metadata generates a default ID."""
    chunks = [Chunk(metadata={}, content="content without metadata")]
    chroma_chunk_store.save(chunks)

    mock_vector_store.add_documents.assert_called_once()
    _, kwargs = mock_vector_store.add_documents.call_args
    
    assert len(kwargs['documents']) == 1
    assert kwargs['ids'] == ["doc_0"]

def test_delete_chunk(chroma_chunk_store, mock_vector_store):
    """Test deleting a chunk by its ID."""
    chroma_chunk_store.delete("id_123")
    mock_vector_store.delete.assert_called_once_with(
        ids=["id_123"],
        where=None,
        where_document=None
    )

def test_delete_chunk_with_optional_params(chroma_chunk_store, mock_vector_store):
    """Test the delete method with optional 'where' parameters."""
    chroma_chunk_store.delete(
        "id_123",
        where={"source": "test"},
        where_document={"content": "test"}
    )
    mock_vector_store.delete.assert_called_once_with(
        ids=["id_123"],
        where={"source": "test"},
        where_document={"content": "test"}
    )

def test_search(chroma_chunk_store, mock_vector_store):
    """Test searching for similar chunks and converting results to Chunk objects."""
    mock_doc = MagicMock()
    mock_doc.page_content = "searched_content"
    mock_doc.metadata = {"source": "searched_source"}
    mock_vector_store.similarity_search.return_value = [mock_doc]

    results = chroma_chunk_store.search(query="test query")
    
    assert len(results) == 1
    assert isinstance(results[0], Chunk)
    assert results[0].content == "searched_content"
    assert results[0].metadata["source"] == "searched_source"
    mock_vector_store.similarity_search.assert_called_once_with(
        query="test query",
        k=5,
        filter=None
    )

def test_search_with_optional_params(chroma_chunk_store, mock_vector_store):
    """Test search method with optional 'top_k' and 'filter' parameters."""
    mock_vector_store.similarity_search.return_value = []

    chroma_chunk_store.search(
        query="test query",
        top_k=10,
        filter={"source": "test"}
    )
    
    mock_vector_store.similarity_search.assert_called_once_with(
        query="test query",
        k=10,  # Ensure top_k is mapped to k
        filter={"source": "test"}
    )

def test_clear_collection(chroma_chunk_store, mock_vector_store):
    """Test clearing a specific named collection."""
    chroma_chunk_store.clear()
    mock_vector_store.delete_collection.assert_called_once()
    assert chroma_chunk_store._vector_store is None # Ensure instance is invalidated

def test_clear_collection_handles_exception(chroma_chunk_store, mock_vector_store):
    """Test that clear() handles exceptions gracefully when a collection doesn't exist."""
    mock_vector_store.delete_collection.side_effect = Exception("Collection not found")
    
    try:
        chroma_chunk_store.clear()
    except Exception as e:
        pytest.fail(f"clear() raised an unexpected exception: {e}")
    
    mock_vector_store.delete_collection.assert_called_once()

@patch('src.infrastructure.adapters.chunk_stores.chroma_chunk_store.shutil.rmtree')
@patch('src.infrastructure.adapters.chunk_stores.chroma_chunk_store.os.path.exists')
def test_clear_deletes_directory_when_collection_name_is_none(mock_exists, mock_rmtree, mock_embedding_model):
    """Test clear() deletes the directory when collection_name is manually set to None."""
    mock_exists.return_value = True
    
    store = ChromaChunkStore(embedding_model=mock_embedding_model)
    # Manually set collection_name to None to bypass __init__ logic
    store.collection_name = None
    
    store.clear()
    
    mock_exists.assert_any_call("./chroma_db")
    mock_rmtree.assert_called_once_with("./chroma_db")
    assert store._vector_store is None

@patch('src.infrastructure.adapters.chunk_stores.chroma_chunk_store.shutil.rmtree')
@patch('src.infrastructure.adapters.chunk_stores.chroma_chunk_store.os.path.exists')
def test_clear_deletes_directory_when_collection_name_is_empty(mock_exists, mock_rmtree, mock_embedding_model):
    """Test clear() deletes the directory when collection_name is an empty string."""
    mock_exists.return_value = True

    store = ChromaChunkStore(embedding_model=mock_embedding_model)
    # Manually set collection_name to an empty string
    store.collection_name = ""

    store.clear()

    mock_exists.assert_any_call("./chroma_db")
    mock_rmtree.assert_called_once_with("./chroma_db")

@patch('src.infrastructure.adapters.chunk_stores.chroma_chunk_store.shutil.rmtree')
@patch('src.infrastructure.adapters.chunk_stores.chroma_chunk_store.os.path.exists')
def test_clear_does_nothing_if_directory_not_exists(mock_exists, mock_rmtree, mock_embedding_model):
    """Test clear() does not call rmtree if the directory doesn't exist."""
    mock_exists.return_value = False
    
    store = ChromaChunkStore(embedding_model=mock_embedding_model)
    store.collection_name = None # Set to None to test the directory removal path
    
    store.clear()
    
    mock_exists.assert_any_call("./chroma_db")
    mock_rmtree.assert_not_called()