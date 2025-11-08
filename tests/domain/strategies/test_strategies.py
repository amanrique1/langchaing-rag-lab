import pytest
from unittest.mock import MagicMock
from src.domain.models.document import Document
from src.domain.strategies.length_based_chunking import LengthBasedChunkingStrategy
from src.domain.strategies.semantic_chunking import SemanticChunkingStrategy
from src.domain.strategies.structure_based_chunking import (
    StructureBasedChunkingStrategy,
)
from src.domain.models.enums import (
    LengthBasedChunkingMode,
    SemanticChunkingThresholdType,
)


@pytest.fixture
def sample_document():
    return Document(
        content="This is a test document. It has several sentences. The document is for testing purposes.",
        metadata={"source": "test.txt"},
    )


@pytest.fixture
def markdown_document():
    return Document(
        content="# Header 1\n\nThis is a paragraph under header 1. ## Header 2\n\nThis is a paragraph under header 2.",
        metadata={"source": "markdown.md"},
    )


def test_length_based_chunking_character_mode(sample_document):
    strategy = LengthBasedChunkingStrategy(
        chunk_size=50, chunk_overlap=10, mode=LengthBasedChunkingMode.CHARACTER
    )
    chunks = strategy.chunk([sample_document])
    assert len(chunks) > 0
    assert all(isinstance(chunk.content, str) for chunk in chunks)
    assert all(len(chunk.content) <= 50 for chunk in chunks)


def test_length_based_chunking_token_mode(sample_document):
    strategy = LengthBasedChunkingStrategy(
        chunk_size=10, chunk_overlap=2, mode=LengthBasedChunkingMode.TOKEN
    )
    chunks = strategy.chunk([sample_document])
    assert len(chunks) > 0
    assert all(isinstance(chunk.content, str) for chunk in chunks)


def test_structure_based_chunking(markdown_document):
    strategy = StructureBasedChunkingStrategy()
    chunks = strategy.chunk([markdown_document])
    assert len(chunks) > 0
    assert "Header 1" in chunks[0].metadata


def test_semantic_chunking(sample_document):
    mock_embedding_model = MagicMock()
    mock_embedding_model.embed_documents.return_value = [
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
        [0.7, 0.8],
    ]
    strategy = SemanticChunkingStrategy(
        embedding_model=mock_embedding_model,
        breakpoint_threshold_type=SemanticChunkingThresholdType.ABSOLUTE,
        breakpoint_threshold_amount=0.5,
    )
    chunks = strategy.chunk([sample_document])
    assert len(chunks) > 0
