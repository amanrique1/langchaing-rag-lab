
import pytest
from unittest.mock import MagicMock, patch
import sys
import tempfile
from pathlib import Path
from src.infrastructure.cli.main import main

@pytest.fixture
def temp_markdown_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test.md"
        with open(file_path, "w") as f:
            f.write("# Test\n\nThis is a test.")
        yield temp_dir

def test_cli_length_based(temp_markdown_file, monkeypatch):
    mock_use_case = MagicMock()
    mock_use_case.execute.return_value = []
    
    with patch('src.infrastructure.cli.main.ChunkingUseCase', return_value=mock_use_case) as mock_chunking_use_case:
        monkeypatch.setattr(sys, 'argv', ['main.py', temp_markdown_file, 'length_based', '--local'])
        main()
        mock_chunking_use_case.assert_called_once()
        mock_use_case.execute.assert_called_once()

def test_cli_structure_based(temp_markdown_file, monkeypatch):
    mock_use_case = MagicMock()
    mock_use_case.execute.return_value = []
    
    with patch('src.infrastructure.cli.main.ChunkingUseCase', return_value=mock_use_case) as mock_chunking_use_case:
        monkeypatch.setattr(sys, 'argv', ['main.py', temp_markdown_file, 'structure_based', '--local'])
        main()
        mock_chunking_use_case.assert_called_once()
        mock_use_case.execute.assert_called_once()

def test_cli_semantic(temp_markdown_file, monkeypatch):
    mock_use_case = MagicMock()
    mock_use_case.execute.return_value = []
    
    with patch('src.infrastructure.cli.main.ChunkingUseCase', return_value=mock_use_case) as mock_chunking_use_case:
        monkeypatch.setattr(sys, 'argv', ['main.py', temp_markdown_file, 'semantic', '--local'])
        main()
        mock_chunking_use_case.assert_called_once()
        mock_use_case.execute.assert_called_once()
