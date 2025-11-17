import pytest
from unittest.mock import MagicMock
from src.application.use_cases.storage_use_case import StorageUseCase
from src.domain.models.chunk import Chunk


def test_save():
    """Tests that the save method calls the chunk_store's save method."""
    mock_chunk_store = MagicMock()
    storage_use_case = StorageUseCase(mock_chunk_store)
    chunks = [Chunk(metadata={}, content="test")]
    storage_use_case.save(chunks)
    mock_chunk_store.save.assert_called_once_with(chunks)


def test_search():
    """Tests that the search method calls the chunk_store's search method."""
    mock_chunk_store = MagicMock()
    storage_use_case = StorageUseCase(mock_chunk_store)
    storage_use_case.search("test query")
    mock_chunk_store.search.assert_called_once_with("test query", top_k=5)


def test_clear():
    """Tests that the clear method calls the chunk_store's clear method."""
    mock_chunk_store = MagicMock()
    storage_use_case = StorageUseCase(mock_chunk_store)
    storage_use_case.clear()
    mock_chunk_store.clear.assert_called_once()
