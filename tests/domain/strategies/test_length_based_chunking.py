from unittest.mock import patch, MagicMock
import pytest

# Adjust these import paths to match your project's structure
from src.domain.models.document import Document
from src.domain.models.chunk import Chunk
from src.domain.models.enums import LengthBasedChunkingMode
from src.domain.services.strategies.length_based_chunking import LengthBasedChunkingStrategy
from langchain_core.documents import Document as LangchainDocument

# The full path to the modules AS THEY ARE IMPORTED in the file under test
CHAR_SPLITTER_PATH = "src.domain.services.strategies.length_based_chunking.CharacterTextSplitter"
TOKEN_SPLITTER_PATH = "src.domain.services.strategies.length_based_chunking.TokenTextSplitter"
CHUNK_MODEL_PATH = "src.domain.services.strategies.length_based_chunking.Chunk"
LANGCHAIN_DOC_PATH = "src.domain.services.strategies.length_based_chunking.LangchainDocument"


@pytest.fixture
def sample_documents() -> list[Document]:
    """Provides a list of sample Document objects for testing."""
    return [
        Document(content="This is the first document. It has some text.", metadata={"source": "doc1.txt"}),
        Document(content="This is the second document.", metadata={"source": "doc2.txt"}),
    ]


def test_chunking_with_character_mode(sample_documents):
    """
    Tests that the CHARACTER mode correctly initializes and uses CharacterTextSplitter.
    Covers: `if self.mode == LengthBasedChunkingMode.CHARACTER`
    """
    # 1. Setup
    strategy = LengthBasedChunkingStrategy(
        chunk_size=20, chunk_overlap=5, mode=LengthBasedChunkingMode.CHARACTER
    )
    
    # Create mock chunks that the splitter will "return"
    mock_lc_chunk1 = LangchainDocument(page_content="This is the first", metadata={"source": "doc1.txt"})
    mock_lc_chunk2 = LangchainDocument(page_content="first document.", metadata={"source": "doc1.txt"})

    # 2. Mock external dependencies
    with patch(CHAR_SPLITTER_PATH) as mock_splitter_class, \
         patch(TOKEN_SPLITTER_PATH) as mock_token_splitter, \
         patch(CHUNK_MODEL_PATH, new=Chunk):  # Use the test's Chunk model to avoid import issues
        
        # Configure the mock splitter instance to return our mock chunks
        mock_splitter_instance = mock_splitter_class.return_value
        mock_splitter_instance.split_documents.return_value = [mock_lc_chunk1, mock_lc_chunk2]

        # 3. Action
        result_chunks = strategy.chunk([sample_documents[0]]) # Use one document for a clear test

        # 4. Assertions
        # Ensure the correct splitter was instantiated with the right parameters
        mock_splitter_class.assert_called_once_with(chunk_size=20, chunk_overlap=5, separator="")
        mock_token_splitter.assert_not_called()
        
        # Ensure the splitter was called
        mock_splitter_instance.split_documents.assert_called_once()
        
        # Check the final output
        assert len(result_chunks) == 2
        assert result_chunks[0].content == "This is the first"
        assert result_chunks[1].content == "first document."
        
        # Verify metadata is correctly created
        assert result_chunks[0].metadata["source"] == "doc1.txt"
        assert result_chunks[0].metadata["chunk_index"] == 0
        assert result_chunks[0].metadata["total_chunks_in_doc"] == 2


def test_chunking_with_token_mode(sample_documents):
    """
    Tests that the TOKEN mode correctly initializes and uses TokenTextSplitter.
    Covers: `elif self.mode == LengthBasedChunkingMode.TOKEN`
    """
    # 1. Setup
    strategy = LengthBasedChunkingStrategy(
        chunk_size=10, chunk_overlap=2, mode=LengthBasedChunkingMode.TOKEN
    )
    mock_lc_chunk = LangchainDocument(page_content="This is the second document.", metadata={"source": "doc2.txt"})

    # 2. Mock
    with patch(TOKEN_SPLITTER_PATH) as mock_splitter_class, \
         patch(CHAR_SPLITTER_PATH) as mock_char_splitter, \
         patch(CHUNK_MODEL_PATH, new=Chunk):

        mock_splitter_instance = mock_splitter_class.return_value
        mock_splitter_instance.split_documents.return_value = [mock_lc_chunk]

        # 3. Action
        result_chunks = strategy.chunk([sample_documents[1]])

        # 4. Assertions
        mock_splitter_class.assert_called_once_with(chunk_size=10, chunk_overlap=2)
        mock_char_splitter.assert_not_called()
        mock_splitter_instance.split_documents.assert_called_once()
        
        assert len(result_chunks) == 1
        assert result_chunks[0].content == "This is the second document."
        assert result_chunks[0].metadata["source"] == "doc2.txt"
        assert result_chunks[0].metadata["chunk_index"] == 0
        assert result_chunks[0].metadata["total_chunks_in_doc"] == 1


def test_chunking_with_invalid_mode(sample_documents):
    """
    Tests that an invalid mode raises a ValueError.
    Covers: `else: raise ValueError(...)`
    """
    strategy = LengthBasedChunkingStrategy(
        chunk_size=10, chunk_overlap=2, mode="INVALID_MODE"
    )
    with pytest.raises(ValueError, match="Invalid mode: INVALID_MODE"):
        strategy.chunk(sample_documents)


def test_chunking_with_no_documents():
    """
    Tests that passing an empty list of documents returns an empty list.
    Covers: `for doc in documents:` loop with an empty list.
    """
    strategy = LengthBasedChunkingStrategy(chunk_size=100, chunk_overlap=10)
    result_chunks = strategy.chunk([])
    assert result_chunks == []


def test_langchain_document_is_created_correctly(sample_documents):
    """
    Verifies that our Document model is correctly converted to a LangchainDocument.
    Covers: `langchain_document = LangchainDocument(...)` line.
    """
    strategy = LengthBasedChunkingStrategy(chunk_size=100, chunk_overlap=10)
    test_doc = sample_documents[0]

    with patch(CHAR_SPLITTER_PATH) as mock_splitter_class, \
         patch(LANGCHAIN_DOC_PATH) as mock_langchain_doc_class:
        
        # We don't care about the output, only that the conversion happened correctly
        mock_splitter_class.return_value.split_documents.return_value = []
        
        strategy.chunk([test_doc])

        # Assert that LangchainDocument was instantiated with the correct data from our document
        mock_langchain_doc_class.assert_called_once_with(
            page_content=test_doc.content, 
            metadata=test_doc.metadata
        )