import pytest
from unittest.mock import MagicMock
from src.application.use_cases.talk_use_case import TalkUseCase
from src.domain.models.chunk import Chunk


def test_execute():
    """Tests that the execute method calls the language_model's get_answer method."""
    mock_language_model = MagicMock()
    mock_chunk_store = MagicMock()
    talk_use_case = TalkUseCase(mock_language_model, mock_chunk_store)
    mock_chunk_store.search.return_value = [Chunk(metadata={}, content="test")]
    talk_use_case.execute("test query")
    mock_language_model.get_answer.assert_called_once()


def test_execute_no_chunks():
    """Tests that the execute method returns a default message when no chunks are found."""
    mock_language_model = MagicMock()
    mock_chunk_store = MagicMock()
    talk_use_case = TalkUseCase(mock_language_model, mock_chunk_store)
    mock_chunk_store.search.return_value = []
    response = talk_use_case.execute("test query")
    assert response == "No relevant information found to answer the query. Please try rephrasing your question."
