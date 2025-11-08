
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from src.domain.models.document import Document
from src.domain.strategies.semantic_chunking import SemanticChunkingStrategy
from src.domain.models.enums import SemanticChunkingThresholdType

@pytest.fixture
def mock_embedding_model():
    model = MagicMock()
    # Simulate embeddings for 5 sentences
    model.embed_documents.return_value = [
        np.array([1.0, 0.0, 0.0]),  # Sentence 1
        np.array([0.9, 0.1, 0.0]),  # Sentence 2 - similar to 1
        np.array([0.0, 1.0, 0.0]),  # Sentence 3 - different
        np.array([0.1, 0.9, 0.0]),  # Sentence 4 - similar to 3
        np.array([0.0, 0.0, 1.0]),  # Sentence 5 - very different
    ]
    return model

@pytest.fixture
def semantic_chunking_strategy(mock_embedding_model):
    return SemanticChunkingStrategy(embedding_model=mock_embedding_model)

@patch('src.domain.strategies.semantic_chunking.sent_tokenize')
def test_semantic_chunking_with_percentile_threshold(mock_sent_tokenize, semantic_chunking_strategy):
    document_content = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
    mock_sent_tokenize.return_value = [
        "Sentence one.", "Sentence two.", "Sentence three.", "Sentence four.", "Sentence five."
    ]
    document = Document(metadata={"path": "test_path"}, content=document_content)

    chunks = semantic_chunking_strategy.chunk([document])

    assert len(chunks) > 0
    expected_chunks = [
        "Sentence one. Sentence two.",
        "Sentence three. Sentence four.",
        "Sentence five."
    ]
    for i, chunk in enumerate(chunks):
        assert chunk.content.strip() == expected_chunks[i]

@patch('src.domain.strategies.semantic_chunking.sent_tokenize')
def test_chunking_with_no_splits(mock_sent_tokenize, mock_embedding_model, semantic_chunking_strategy):
    # All sentences are similar
    mock_embedding_model.embed_documents.return_value = [np.array([1.0, 0.0])] * 5
    document_content = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
    mock_sent_tokenize.return_value = [
        "Sentence one.", "Sentence two.", "Sentence three.", "Sentence four.", "Sentence five."
    ]
    document = Document(metadata={"path": "test_path"}, content=document_content)

    chunks = semantic_chunking_strategy.chunk([document])
    assert len(chunks) == 1
    assert chunks[0].content.strip() == document_content

@patch('src.domain.strategies.semantic_chunking.sent_tokenize')
def test_single_sentence_document(mock_sent_tokenize, semantic_chunking_strategy):
    document_content = "This is a single sentence."
    mock_sent_tokenize.return_value = ["This is a single sentence."]
    # Mock embeddings for a single sentence
    semantic_chunking_strategy.embedding_model.embed_documents.return_value = [np.array([1.0, 0.0])]
    document = Document(metadata={"path": "test_path"}, content=document_content)
    chunks = semantic_chunking_strategy.chunk([document])
    assert len(chunks) == 1
    assert chunks[0].content.strip() == document_content

@patch('src.domain.strategies.semantic_chunking.sent_tokenize')
def test_empty_document(mock_sent_tokenize, semantic_chunking_strategy):
    document = Document(metadata={"path": "test_path"}, content="")
    mock_sent_tokenize.return_value = []
    chunks = semantic_chunking_strategy.chunk([document])
    assert len(chunks) == 0

@patch('src.domain.strategies.semantic_chunking.sent_tokenize')
def test_different_thresholds(mock_sent_tokenize):
    """
    Tests that a higher percentile threshold leads to more chunks,
    and a lower percentile threshold leads to fewer chunks.
    """
    # Mock the embedding model dependency
    mock_embedding_model = MagicMock()

    mock_sent_tokenize.return_value = [
        "Sentence one.", "Sentence two.", "Sentence three.", "Sentence four.", "Sentence five."
    ]

    # Mock embeddings to create a predictable set of similarities
    mock_embedding_model.embed_documents.return_value = [
        np.array([1.0, 0.0]),  # S1
        np.array([0.9, 0.1]), # S2 (similar to S1)
        np.array([0.1, 0.9]), # S3 (dissimilar to S2)
        np.array([0.11, 0.89]),# S4 (similar to S3)
        np.array([0.8, 0.2]), # S5 (dissimilar to S4)
    ]

    # This strategy uses a high percentile, which results in a high similarity
    # threshold. More sentence pairs will fall below this threshold, leading to MORE splits.
    strategy_high_threshold = SemanticChunkingStrategy(
        embedding_model=mock_embedding_model,
        breakpoint_threshold_type=SemanticChunkingThresholdType.PERCENTILE,
        breakpoint_threshold_amount=95
    )

    # This strategy uses a low percentile, resulting in a low similarity
    # threshold. Fewer sentence pairs will fall below this, leading to FEWER splits.
    strategy_low_threshold = SemanticChunkingStrategy(
        embedding_model=mock_embedding_model,
        breakpoint_threshold_type=SemanticChunkingThresholdType.PERCENTILE,
        breakpoint_threshold_amount=5
    )

    document_content = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
    # FIX: Added the required 'metadata' argument during Document instantiation.
    document = Document(content=document_content, metadata={"source": "test.txt"})

    chunks_high = strategy_high_threshold.chunk([document])
    chunks_low = strategy_low_threshold.chunk([document])

    # The high percentile threshold (95) should create more chunks than the low one (5).
    assert len(chunks_high) > len(chunks_low)