from unittest.mock import MagicMock, patch, call
import pytest
import os
from src.domain.models.chunk import Chunk
from src.infrastructure.adapters.chunk_stores.chroma_chunk_store import ChromaChunkStore

@pytest.fixture
def mock_embedding_model():
    """Mock GoogleGenerativeAIEmbeddings to avoid requiring API token"""
    with patch('src.infrastructure.adapters.chunk_stores.chroma_chunk_store.GoogleGenerativeAIEmbeddings') as mock_embeddings:
        mock_instance = MagicMock()
        mock_embeddings.return_value = mock_instance
        yield mock_embeddings

@pytest.fixture
def mock_chroma():
    """Mock Chroma vector store"""
    with patch('src.infrastructure.adapters.chunk_stores.chroma_chunk_store.Chroma') as mock_chroma:
        mock_vector_store = MagicMock()
        mock_chroma.return_value = mock_vector_store
        yield mock_chroma

@pytest.fixture
def chroma_chunk_store(mock_embedding_model, mock_chroma, tmp_path):
    """Create a ChromaChunkStore with mocked dependencies"""
    store = ChromaChunkStore(collection_name="test_collection", persist_directory=str(tmp_path))
    yield store

def test_initialization(mock_embedding_model, mock_chroma, tmp_path):
    """Test that ChromaChunkStore initializes correctly"""
    store = ChromaChunkStore(collection_name="test_collection", persist_directory=str(tmp_path))
    
    assert store.collection_name == "test_collection"
    assert store.persist_directory == str(tmp_path)
    mock_embedding_model.assert_called_once_with(model="models/embedding-001")
    mock_chroma.assert_called_once_with(
        collection_name="test_collection",
        persist_directory=str(tmp_path),
        embedding_function=mock_embedding_model.return_value,
    )

def test_initialization_default_params(mock_embedding_model, mock_chroma):
    """Test initialization with default parameters"""
    store = ChromaChunkStore()
    
    assert store.collection_name == "rag_docs"
    assert store.persist_directory == "./chroma_db"

def test_save_chunks(chroma_chunk_store):
    """Test saving chunks to the store"""
    chunks = [
        Chunk(metadata={"source": "path1", "chunk_index": 0}, content="content1"),
        Chunk(metadata={"source": "path2", "chunk_index": 1}, content="content2"),
    ]
    chroma_chunk_store.save(chunks)

    chroma_chunk_store.vector_store.add_documents.assert_called_once()
    args, kwargs = chroma_chunk_store.vector_store.add_documents.call_args
    
    assert len(kwargs['documents']) == 2
    assert kwargs['ids'] == ["path1_0", "path2_1"]
    # Verify document content
    assert kwargs['documents'][0].page_content == "content1"
    assert kwargs['documents'][1].page_content == "content2"

def test_save_chunks_with_missing_metadata(chroma_chunk_store):
    """Test saving chunks with missing source and chunk_index in metadata"""
    chunks = [
        Chunk(metadata={}, content="content without metadata"),
    ]
    chroma_chunk_store.save(chunks)

    chroma_chunk_store.vector_store.add_documents.assert_called_once()
    args, kwargs = chroma_chunk_store.vector_store.add_documents.call_args
    
    assert len(kwargs['documents']) == 1
    assert kwargs['ids'] == ["doc_0"]

def test_get_chunk(chroma_chunk_store):
    """Test retrieving a chunk by ID"""
    chroma_chunk_store.vector_store.get.return_value = {
        'ids': ['1'],
        'documents': ['retrieved_content'],
        'metadatas': [{'document_path': 'retrieved_path', 'id': '1'}]
    }

    retrieved_chunk = chroma_chunk_store.get("1")

    assert retrieved_chunk is not None
    assert retrieved_chunk.content == "retrieved_content"
    assert retrieved_chunk.metadata['document_path'] == "retrieved_path"
    chroma_chunk_store.vector_store.get.assert_called_once_with(
        ids=["1"],
        where=None,
        limit=None,
        offset=None,
        where_document=None,
        include=None,
    )

def test_get_chunk_with_optional_params(chroma_chunk_store):
    """Test get method with all optional parameters"""
    chroma_chunk_store.vector_store.get.return_value = {
        'ids': ['1'],
        'documents': ['retrieved_content'],
        'metadatas': [{'source': 'test'}]
    }

    retrieved_chunk = chroma_chunk_store.get(
        "1",
        where={"source": "test"},
        limit=10,
        offset=5,
        where_document={"content": "test"},
        include=["metadatas", "documents"]
    )

    assert retrieved_chunk is not None
    chroma_chunk_store.vector_store.get.assert_called_once_with(
        ids=["1"],
        where={"source": "test"},
        limit=10,
        offset=5,
        where_document={"content": "test"},
        include=["metadatas", "documents"],
    )

def test_get_chunk_not_found(chroma_chunk_store):
    """Test retrieving a non-existent chunk"""
    chroma_chunk_store.vector_store.get.return_value = {'ids': [], 'documents': [], 'metadatas': []}
    retrieved_chunk = chroma_chunk_store.get("non_existent_id")
    assert retrieved_chunk is None

def test_get_chunk_empty_results(chroma_chunk_store):
    """Test retrieving when results exist but are empty"""
    chroma_chunk_store.vector_store.get.return_value = None
    retrieved_chunk = chroma_chunk_store.get("non_existent_id")
    assert retrieved_chunk is None

def test_delete_chunk(chroma_chunk_store):
    """Test deleting a chunk by ID"""
    chroma_chunk_store.delete("123")
    chroma_chunk_store.vector_store.delete.assert_called_once_with(
        ids=["123"],
        where=None,
        where_document=None
    )

def test_delete_chunk_with_optional_params(chroma_chunk_store):
    """Test delete method with optional parameters"""
    chroma_chunk_store.delete(
        "123",
        where={"source": "test"},
        where_document={"content": "test"}
    )
    chroma_chunk_store.vector_store.delete.assert_called_once_with(
        ids=["123"],
        where={"source": "test"},
        where_document={"content": "test"}
    )

def test_search(chroma_chunk_store):
    """Test searching for similar chunks"""
    # Mocking the Document object returned by similarity_search_by_vector
    mock_doc = MagicMock()
    mock_doc.page_content = "searched_content"
    mock_doc.metadata = {"source": "searched_source"}
    chroma_chunk_store.vector_store.similarity_search_by_vector.return_value = [mock_doc]

    results = chroma_chunk_store.search(query_embedding=[0.1, 0.2, 0.3])
    
    assert len(results) == 1
    assert results[0].content == "searched_content"
    assert results[0].metadata["source"] == "searched_source"
    chroma_chunk_store.vector_store.similarity_search_by_vector.assert_called_once_with(
        embedding=[0.1, 0.2, 0.3],
        k=5,
        where=None,
        where_document=None,
    )

def test_search_with_optional_params(chroma_chunk_store):
    """Test search method with all optional parameters"""
    mock_doc = MagicMock()
    mock_doc.page_content = "searched_content"
    mock_doc.metadata = {"source": "searched_source"}
    chroma_chunk_store.vector_store.similarity_search_by_vector.return_value = [mock_doc]

    results = chroma_chunk_store.search(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=10,
        where={"source": "test"},
        where_document={"content": "test"}
    )
    
    assert len(results) == 1
    chroma_chunk_store.vector_store.similarity_search_by_vector.assert_called_once_with(
        embedding=[0.1, 0.2, 0.3],
        k=10,
        where={"source": "test"},
        where_document={"content": "test"},
    )

def test_search_multiple_results(chroma_chunk_store):
    """Test search returning multiple results"""
    mock_docs = []
    for i in range(3):
        mock_doc = MagicMock()
        mock_doc.page_content = f"content_{i}"
        mock_doc.metadata = {"source": f"source_{i}"}
        mock_docs.append(mock_doc)
    
    chroma_chunk_store.vector_store.similarity_search_by_vector.return_value = mock_docs

    results = chroma_chunk_store.search(query_embedding=[0.1, 0.2, 0.3])
    
    assert len(results) == 3
    for i, result in enumerate(results):
        assert result.content == f"content_{i}"
        assert result.metadata["source"] == f"source_{i}"

def test_clear_collection(mock_embedding_model, mock_chroma, tmp_path):
    """Test clearing a collection successfully"""
    store = ChromaChunkStore(collection_name="test_collection", persist_directory=str(tmp_path))
    
    # Reset the mock to clear the initialization call
    mock_chroma.reset_mock()
    
    store.clear()
    
    # Should delete the collection
    store.vector_store.delete_collection.assert_called_once()
    
    # Should recreate the vector store
    mock_chroma.assert_called_once_with(
        collection_name="test_collection",
        persist_directory=str(tmp_path),
        embedding_function=store.embeddings,
    )

def test_clear_collection_handles_exception(mock_embedding_model, mock_chroma, tmp_path):
    """Test that clear handles exception when collection doesn't exist"""
    store = ChromaChunkStore(collection_name="test_collection", persist_directory=str(tmp_path))
    store.vector_store.delete_collection.side_effect = Exception("Collection not found")
    
    # Reset mock to clear the initialization call
    mock_chroma.reset_mock()
    
    # Should not raise an exception
    store.clear()
    
    # Ensure we tried to delete
    store.vector_store.delete_collection.assert_called_once()
    
    # Should NOT recreate the vector store when exception occurs (based on the except: pass logic)
    mock_chroma.assert_not_called()

@patch('src.infrastructure.adapters.chunk_stores.chroma_chunk_store.shutil.rmtree')
@patch('src.infrastructure.adapters.chunk_stores.chroma_chunk_store.os.path.exists')
def test_clear_no_collection_name(mock_exists, mock_rmtree, mock_embedding_model, mock_chroma, tmp_path):
    """Test clearing when collection_name is None/empty - deletes entire directory"""
    mock_exists.return_value = True
    
    store = ChromaChunkStore(collection_name="test_collection", persist_directory=str(tmp_path))
    
    # Manually set collection_name to None/empty to test the else branch
    store.collection_name = None
    
    store.clear()
    
    # Should check if directory exists and delete it
    mock_exists.assert_called_once_with(str(tmp_path))
    mock_rmtree.assert_called_once_with(str(tmp_path))
    
    # Should NOT try to delete collection
    store.vector_store.delete_collection.assert_not_called()

@patch('src.infrastructure.adapters.chunk_stores.chroma_chunk_store.shutil.rmtree')
@patch('src.infrastructure.adapters.chunk_stores.chroma_chunk_store.os.path.exists')
def test_clear_no_collection_directory_not_exists(mock_exists, mock_rmtree, mock_embedding_model, mock_chroma, tmp_path):
    """Test clearing when directory doesn't exist"""
    mock_exists.return_value = False
    
    store = ChromaChunkStore(collection_name="test_collection", persist_directory=str(tmp_path))
    store.collection_name = None
    
    # Should not raise an exception even if directory doesn't exist
    store.clear()
    
    # Should check if directory exists but not try to delete it
    mock_exists.assert_called_once_with(str(tmp_path))
    mock_rmtree.assert_not_called()

@patch('src.infrastructure.adapters.chunk_stores.chroma_chunk_store.shutil.rmtree')
@patch('src.infrastructure.adapters.chunk_stores.chroma_chunk_store.os.path.exists')
def test_clear_empty_collection_name(mock_exists, mock_rmtree, mock_embedding_model, mock_chroma, tmp_path):
    """Test clearing when collection_name is empty string"""
    mock_exists.return_value = True
    
    store = ChromaChunkStore(collection_name="test_collection", persist_directory=str(tmp_path))
    store.collection_name = ""
    
    store.clear()
    
    # Empty string is falsy, so should go to else branch
    mock_exists.assert_called_once_with(str(tmp_path))
    mock_rmtree.assert_called_once_with(str(tmp_path))