import pytest
from unittest.mock import MagicMock
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