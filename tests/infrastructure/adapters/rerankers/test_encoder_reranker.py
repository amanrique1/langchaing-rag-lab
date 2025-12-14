import pytest
from unittest.mock import MagicMock
from src.infrastructure.adapters.rerankers.encoder_reranker import EncoderReranker
from src.domain.models.search_result import SearchResult
from src.domain.models.chunk import Chunk

class TestEncoderReranker:

    @pytest.fixture
    def mock_cross_encoder(self, mocker):
        mock_encoder = mocker.patch("src.infrastructure.adapters.rerankers.encoder_reranker.CrossEncoder")
        return mock_encoder.return_value

    def test_rerank_empty_results(self, mock_cross_encoder):
        reranker = EncoderReranker()
        results = reranker.rerank("query", [])
        assert results == []

    def test_rerank_reorders_results(self, mock_cross_encoder):
        # Setup
        mock_cross_encoder.predict.return_value = [0.1, 0.9, 0.5]
        
        chunk1 = Chunk(content="Content 1", metadata={})
        chunk2 = Chunk(content="Content 2", metadata={})
        chunk3 = Chunk(content="Content 3", metadata={})
        
        result1 = SearchResult(chunk=chunk1, score=0.5, rank=1, retrieval_method="test")
        result2 = SearchResult(chunk=chunk2, score=0.4, rank=2, retrieval_method="test")
        result3 = SearchResult(chunk=chunk3, score=0.3, rank=3, retrieval_method="test")
        
        results = [result1, result2, result3]
        
        reranker = EncoderReranker()
        reranked = reranker.rerank("query", results, top_k=3)
        
        # Verify call to encoder
        pairs = [("query", "Content 1"), ("query", "Content 2"), ("query", "Content 3")]
        # Note: Depending on how mocks work, we assume the constructor called CrossEncoder and we get the instance
        
        # Validation
        assert len(reranked) == 3
        assert reranked[0].chunk.content == "Content 2" # Score 0.9
        assert reranked[1].chunk.content == "Content 3" # Score 0.5
        assert reranked[2].chunk.content == "Content 1" # Score 0.1
        
        assert reranked[0].rank == 1
        assert reranked[1].rank == 2
        assert reranked[2].rank == 3
        
        assert reranked[0].score == 0.9
