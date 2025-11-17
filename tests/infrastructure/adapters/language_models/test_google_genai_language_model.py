import pytest
from unittest.mock import MagicMock
from src.infrastructure.adapters.language_models.google_genai_language_model import (
    GoogleGenAILanguageModel,
)
from src.domain.models.chunk import Chunk


def test_get_answer():
    """Tests that the get_answer method works correctly."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "test answer"
    model = GoogleGenAILanguageModel()
    model.chain = mock_chain
    answer = model.get_answer(
        "test question", [Chunk(metadata={}, content="test context")]
    )
    assert answer == "test answer"
