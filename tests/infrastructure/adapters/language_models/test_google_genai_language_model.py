import pytest
from unittest.mock import MagicMock, patch
from src.infrastructure.adapters.language_models.google_genai_language_model import (
    GoogleGenAILanguageModel,
)
from src.domain.models.chunk import Chunk


@patch('src.infrastructure.adapters.language_models.google_genai_language_model.ChatGoogleGenerativeAI')
@patch('src.infrastructure.adapters.language_models.google_genai_language_model.ChatPromptTemplate')
def test_get_answer(mock_prompt_template, mock_chat_google):
    """Tests that the get_answer method works correctly without making real API calls."""
    # Mock the chain components to prevent real API initialization
    mock_template = MagicMock()
    mock_prompt_template.from_template.return_value = mock_template
    mock_llm = MagicMock()
    mock_chat_google.return_value = mock_llm
    
    # Create the model (now mocked, won't make real API calls)
    model = GoogleGenAILanguageModel()
    
    # Replace the chain with our test mock
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "test answer"
    model.chain = mock_chain
    
    answer = model.get_answer(
        "test question", [Chunk(metadata={}, content="test context")]
    )
    assert answer == "test answer"


@patch('src.infrastructure.adapters.language_models.google_genai_language_model.ChatGoogleGenerativeAI')
@patch('src.infrastructure.adapters.language_models.google_genai_language_model.ChatPromptTemplate')
def test_initialization_with_custom_parameters(mock_prompt_template, mock_chat_google):
    """Tests that the model can be initialized with custom parameters without making API calls."""
    # Mock the components to prevent real API calls
    mock_template = MagicMock()
    mock_prompt_template.from_template.return_value = mock_template
    mock_llm = MagicMock()
    mock_chat_google.return_value = mock_llm
    
    # Initialize with custom parameters (mocked, no real API calls)
    model = GoogleGenAILanguageModel(model_name="gemini-pro", temperature=0.5)
    
    # Verify the ChatGoogleGenerativeAI was called with correct parameters
    mock_chat_google.assert_called_once_with(model="gemini-pro", temperature=0.5)
    assert model.chain is not None


def test_query_template_file_not_found():
    """Tests that FileNotFoundError is raised when query template file doesn't exist."""
    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="The file assets/query_template.txt does not exist"):
            # Force re-import to trigger the module-level code
            import importlib
            from src.infrastructure.adapters.language_models import google_genai_language_model
            importlib.reload(google_genai_language_model)

