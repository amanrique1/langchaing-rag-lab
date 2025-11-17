from unittest.mock import MagicMock, patch
import pytest
from src.infrastructure.adapters.language_models.google_genai_embedding_model import GoogleGenAIEmbeddingModel


@pytest.fixture
def mock_google_embeddings():
    """Fixture to mock the GoogleGenerativeAIEmbeddings from LangChain."""
    with patch('src.infrastructure.adapters.language_models.google_genai_embedding_model.GoogleGenerativeAIEmbeddings') as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        yield mock_instance


def test_initialization(mock_google_embeddings):
    """Test that the adapter initializes the LangChain model correctly."""
    GoogleGenAIEmbeddingModel(model_name="models/custom-embedding")
    mock_google_embeddings.__init__.assert_called_once_with(model="models/custom-embedding")


def test_embed_query(mock_google_embeddings):
    """Test that embed_query calls the underlying LangChain method."""
    adapter = GoogleGenAIEmbeddingModel()
    mock_google_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]

    result = adapter.embed_query("test query")

    mock_google_embeddings.embed_query.assert_called_once_with("test query")
    assert result == [0.1, 0.2, 0.3]


def test_embed_documents(mock_google_embeddings):
    """Test that embed_documents calls the underlying LangChain method."""
    adapter = GoogleGenAIEmbeddingModel()
    documents = ["doc1", "doc2"]
    mock_google_embeddings.embed_documents.return_value = [[0.1], [0.2]]

    result = adapter.embed_documents(documents)

    mock_google_embeddings.embed_documents.assert_called_once_with(documents)
    assert result == [[0.1], [0.2]]
