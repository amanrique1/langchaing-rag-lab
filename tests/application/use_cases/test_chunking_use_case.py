import pytest
from unittest.mock import MagicMock
from src.application.use_cases.chunking_use_case import ChunkingUseCase
from src.domain.models.document import Document


@pytest.fixture
def mock_document_loader():
    return MagicMock()


@pytest.fixture
def mock_chunk_store():
    return MagicMock()


@pytest.fixture
def chunking_use_case(mock_document_loader):
    return ChunkingUseCase(document_loader=mock_document_loader)


def test_chunking_use_case_execute(chunking_use_case, mock_document_loader):
    mock_document_loader.load.return_value = [Document(content="test", metadata={})]

    strategy_name = "length_based"
    strategy_config = {"mode": "character", "chunk_size": 100, "chunk_overlap": 20}

    chunks = chunking_use_case.execute(
        source="dummy_source",
        strategy_name=strategy_name,
        strategy_config=strategy_config,
        loader_mode="single",
    )

    mock_document_loader.load.assert_called_once()
    assert len(chunks) > 0
