
import json
import os
from unittest.mock import MagicMock, mock_open, patch
import pytest
from pathlib import Path
from src.domain.models.chunk import Chunk
from src.infrastructure.adapters.chunk_stores.file_system_chunk_store import FileSystemChunkStore

@pytest.fixture
def chunk_store(tmp_path):
    return FileSystemChunkStore(output_dir=str(tmp_path))

def test_save(chunk_store, tmp_path):
    chunks = [
        Chunk(metadata={"chunk_index": 0}, content="content1"),
        Chunk(metadata={"chunk_index": 1}, content="content2"),
    ]
    chunk_store.save(chunks)

    # Verify file content
    for chunk in chunks:
        file_path = tmp_path / f"chunk_{chunk.metadata['chunk_index']}.json"
        assert file_path.exists()
        with open(file_path) as f:
            data = json.load(f)
            assert data["content"] == chunk.content
            assert data["metadata"] == chunk.metadata

def test_get_is_stubbed(chunk_store):
    assert chunk_store.get("some_id") is None

def test_delete_is_stubbed(chunk_store):
    # The method is a no-op, so we just call it
    chunk_store.delete("some_id")

def test_search_is_stubbed(chunk_store):
    assert chunk_store.search([0.1, 0.2]) == []

def test_clear_is_stubbed(chunk_store):
    # The method is a no-op, so we just call it
    chunk_store.clear()

def test_initialization_creates_directory(tmp_path):
    new_dir = tmp_path / "new_output"
    assert not new_dir.exists()
    FileSystemChunkStore(output_dir=str(new_dir))
    assert new_dir.exists()

@patch("pathlib.Path.mkdir")
def test_initialization_handles_existing_directory(mock_mkdir):
    FileSystemChunkStore(output_dir="existing_dir")
    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

@patch("builtins.open", new_callable=mock_open)
@patch("json.dump")
def test_save_writes_correct_data(mock_json_dump, mock_file):
    store = FileSystemChunkStore(output_dir="test_dir")
    chunks = [Chunk(metadata={"chunk_index": "test_id"}, content="test_content")]
    store.save(chunks)

    mock_file.assert_called_once_with(Path("test_dir") / "chunk_test_id.json", "w", encoding="utf-8")
    mock_json_dump.assert_called_once()
    # More detailed assertions can be added here to check the content of what's being written
    args, kwargs = mock_json_dump.call_args
    assert args[0]['content'] == 'test_content'
