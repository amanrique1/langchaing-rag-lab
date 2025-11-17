from unittest.mock import MagicMock, patch
import pytest

from src.infrastructure.adapters.language_models.google_genai_embedding_model import GoogleGenAIEmbeddingModel


@pytest.fixture
def mock_google_embeddings():
    """
    Fixture to mock GoogleGenerativeAIEmbeddings.
    It yields a tuple containing the mock class and the mock instance,
    allowing tests to verify both instantiation and method calls.
    """
    with patch('src.infrastructure.adapters.language_models.google_genai_embedding_model.GoogleGenerativeAIEmbeddings') as mock_class:
        # The instance that the mocked class will return upon instantiation
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        yield mock_class, mock_instance


def test_initialization_with_custom_model(mock_google_embeddings):
    """Test that the adapter initializes the LangChain model with a custom model name."""
    # We need the mock_class to check how the constructor was called.
    mock_class, _ = mock_google_embeddings
    
    # This line will trigger the __init__ of our adapter, which in turn
    # calls the constructor of the mocked GoogleGenerativeAIEmbeddings.
    GoogleGenAIEmbeddingModel(model_name="models/custom-embedding")

    # Assert that the mocked class was instantiated exactly once with the correct model.
    mock_class.assert_called_once_with(model="models/custom-embedding")


def test_initialization_with_default_model(mock_google_embeddings):
    """Test that the adapter initializes with the default model name to ensure 100% coverage."""
    mock_class, _ = mock_google_embeddings

    # Initialize the adapter without arguments to test the default parameter.
    GoogleGenAIEmbeddingModel()

    # Assert that the mocked class was instantiated with the default model name.
    mock_class.assert_called_once_with(model="models/embedding-001")


def test_embed_query(mock_google_embeddings):
    """Test that embed_query correctly calls the underlying LangChain method."""
    # We need the mock_instance to control and check method calls.
    _, mock_instance = mock_google_embeddings

    # The adapter's __init__ will use the mock_instance.
    adapter = GoogleGenAIEmbeddingModel()

    # Set up the return value for the method call on the mock instance.
    mock_instance.embed_query.return_value = [0.1, 0.2, 0.3]

    # Call the method on our adapter.
    result = adapter.embed_query("test query")

    # Assert that the adapter correctly forwarded the call to the instance.
    mock_instance.embed_query.assert_called_once_with("test query")
    assert result == [0.1, 0.2, 0.3]


def test_embed_documents(mock_google_embeddings):
    """Test that embed_documents correctly calls the underlying LangChain method."""
    _, mock_instance = mock_google_embeddings

    adapter = GoogleGenAIEmbeddingModel()
    documents = ["doc1", "doc2"]

    # Set up the return value.
    mock_instance.embed_documents.return_value = [[0.1], [0.2]]

    # Call the method.
    result = adapter.embed_documents(documents)

    # Assert the call was forwarded correctly.
    mock_instance.embed_documents.assert_called_once_with(documents)
    assert result == [[0.1], [0.2]]