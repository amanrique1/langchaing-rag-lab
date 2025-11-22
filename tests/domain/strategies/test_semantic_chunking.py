import pytest
from unittest.mock import MagicMock, patch
from src.domain.strategies.semantic_chunking import SemanticChunkingStrategy
from src.domain.models.document import Document
from src.domain.models.enums import SemanticChunkingThresholdType


def test_chunk():
    """Tests that the chunk method works correctly."""
    mock_embedding_model = MagicMock()
    mock_embedding_model.embed_documents.return_value = [[0.1, 0.2, 0.3]] * 5
    strategy = SemanticChunkingStrategy(
        embedding_model=mock_embedding_model,
        breakpoint_threshold_type=SemanticChunkingThresholdType.PERCENTILE,
        breakpoint_threshold_amount=0.95,
    )
    document = Document(
        metadata={},
        content="This is a sentence. This is another sentence. This is a third sentence. This is a fourth sentence. This is a fifth sentence.",
    )
    chunks = strategy.chunk([document])
    assert len(chunks) > 0


def test_default_embedding_model_initialization():
    """Tests that the default embedding model is initialized when none is provided."""
    with patch.dict('os.environ', {'EMBEDDING_MODEL': 'models/embedding-001'}):
        with patch('src.domain.strategies.semantic_chunking.GoogleGenerativeAIEmbeddings') as mock_embeddings:
            strategy = SemanticChunkingStrategy()
            mock_embeddings.assert_called_once_with(model='models/embedding-001')


def test_standard_deviation_threshold():
    """Tests the STANDARD_DEVIATION threshold type."""
    mock_embedding_model = MagicMock()
    # Create embeddings that will result in varying similarities
    mock_embedding_model.embed_documents.return_value = [
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.8, 0.9, 1.0],
        [0.85, 0.95, 1.0],
    ]
    
    strategy = SemanticChunkingStrategy(
        embedding_model=mock_embedding_model,
        breakpoint_threshold_type=SemanticChunkingThresholdType.STANDARD_DEVIATION,
        breakpoint_threshold_amount=1.0,
    )
    
    document = Document(
        metadata={"source": "test.md"},
        content="First sentence. Second sentence. Third sentence. Fourth sentence.",
    )
    chunks = strategy.chunk([document])
    assert len(chunks) > 0


def test_interquartile_threshold():
    """Tests the INTERQUARTILE threshold type."""
    mock_embedding_model = MagicMock()
    mock_embedding_model.embed_documents.return_value = [
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5],
        [0.4, 0.5, 0.6],
    ]
    
    strategy = SemanticChunkingStrategy(
        embedding_model=mock_embedding_model,
        breakpoint_threshold_type=SemanticChunkingThresholdType.INTERQUARTILE,
        breakpoint_threshold_amount=1.5,
    )
    
    document = Document(
        metadata={"source": "test.md"},
        content="First sentence. Second sentence. Third sentence. Fourth sentence.",
    )
    chunks = strategy.chunk([document])
    assert len(chunks) > 0


def test_absolute_threshold():
    """Tests the ABSOLUTE threshold type."""
    mock_embedding_model = MagicMock()
    mock_embedding_model.embed_documents.return_value = [[0.1, 0.2, 0.3]] * 3
    
    strategy = SemanticChunkingStrategy(
        embedding_model=mock_embedding_model,
        breakpoint_threshold_type=SemanticChunkingThresholdType.ABSOLUTE,
        breakpoint_threshold_amount=0.5,
    )
    
    document = Document(
        metadata={"source": "test.md"},
        content="First sentence. Second sentence. Third sentence.",
    )
    chunks = strategy.chunk([document])
    assert len(chunks) > 0


def test_invalid_threshold_type():
    """Tests that an invalid threshold type raises ValueError."""
    mock_embedding_model = MagicMock()
    mock_embedding_model.embed_documents.return_value = [[0.1, 0.2, 0.3]] * 3
    
    strategy = SemanticChunkingStrategy(
        embedding_model=mock_embedding_model,
        breakpoint_threshold_type="INVALID_TYPE",  # Invalid type
        breakpoint_threshold_amount=0.5,
    )
    
    document = Document(
        metadata={},
        content="First sentence. Second sentence. Third sentence.",
    )
    
    with pytest.raises(ValueError, match="Unsupported threshold type"):
        chunks = strategy.chunk([document])


def test_empty_document():
    """Tests handling of documents with no sentences."""
    mock_embedding_model = MagicMock()
    
    strategy = SemanticChunkingStrategy(
        embedding_model=mock_embedding_model,
        breakpoint_threshold_type=SemanticChunkingThresholdType.PERCENTILE,
        breakpoint_threshold_amount=95.0,
    )
    
    document = Document(metadata={}, content="")
    chunks = strategy.chunk([document])
    assert len(chunks) == 0


def test_single_sentence_document():
    """Tests handling of documents with only one sentence."""
    mock_embedding_model = MagicMock()
    mock_embedding_model.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    
    strategy = SemanticChunkingStrategy(
        embedding_model=mock_embedding_model,
        breakpoint_threshold_type=SemanticChunkingThresholdType.PERCENTILE,
        breakpoint_threshold_amount=95.0,
    )
    
    document = Document(metadata={"source": "test.md"}, content="Single sentence.")
    chunks = strategy.chunk([document])
    assert len(chunks) == 1
    assert chunks[0].content == "Single sentence."


def test_max_chunk_size_enforcement():
    """Tests that max_chunk_size is enforced."""
    mock_embedding_model = MagicMock()
    # High similarity to avoid natural breaks
    mock_embedding_model.embed_documents.return_value = [
        [0.9, 0.9, 0.9],
        [0.91, 0.91, 0.91],
        [0.92, 0.92, 0.92],
        [0.93, 0.93, 0.93],
        [0.94, 0.94, 0.94],
    ]
    
    strategy = SemanticChunkingStrategy(
        embedding_model=mock_embedding_model,
        breakpoint_threshold_type=SemanticChunkingThresholdType.PERCENTILE,
        breakpoint_threshold_amount=95.0,
        max_chunk_size=2,  # Force chunks to break after 2 sentences
    )
    
    document = Document(
        metadata={"source": "test.md"},
        content="One. Two. Three. Four. Five.",
    )
    chunks = strategy.chunk([document])
    # Should create multiple chunks due to max_chunk_size
    assert len(chunks) > 1


def test_chunk_metadata_creation():
    """Tests that chunks are created with proper metadata."""
    mock_embedding_model = MagicMock()
    # Low similarity to force breaks
    mock_embedding_model.embed_documents.return_value = [
        [0.1, 0.1, 0.1],
        [0.9, 0.9, 0.9],
        [0.1, 0.1, 0.1],
    ]
    
    strategy = SemanticChunkingStrategy(
        embedding_model=mock_embedding_model,
        breakpoint_threshold_type=SemanticChunkingThresholdType.PERCENTILE,
        breakpoint_threshold_amount=50.0,
        min_chunk_size=1,
    )
    
    document = Document(
        metadata={"source": "test.md"},
        content="First sentence. Second sentence. Third sentence.",
    )
    chunks = strategy.chunk([document])
    
    # Verify metadata structure
    for chunk in chunks:
        assert "source" in chunk.metadata
        assert "doc_index" in chunk.metadata
        assert "chunk_index" in chunk.metadata
        assert "start_sentence_index" in chunk.metadata
        assert "end_sentence_index" in chunk.metadata
        assert chunk.metadata["source"] == "test.md"