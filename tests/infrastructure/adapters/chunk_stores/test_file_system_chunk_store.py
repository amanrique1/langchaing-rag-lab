
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

def test_delete_existing_file(chunk_store, tmp_path):
    """Tests that delete removes an existing chunk file."""
    # Create a chunk file
    chunk = Chunk(metadata={"chunk_index": 0}, content="test content")
    chunk_store.save([chunk])
    
    file_path = tmp_path / "chunk_0.json"
    assert file_path.exists()
    
    # Delete the file
    chunk_store.delete("0")
    assert not file_path.exists()


def test_delete_nonexistent_file(chunk_store):
    """Tests that delete handles non-existent files gracefully."""
    # Should not raise an error
    chunk_store.delete("nonexistent_id")


def test_search(chunk_store, tmp_path):
    """Tests the keyword search functionality."""
    # Create some test files
    chunk1 = Chunk(metadata={"chunk_index": 0}, content="hello world")
    chunk2 = Chunk(metadata={"chunk_index": 1}, content="hello there")
    chunk3 = Chunk(metadata={"chunk_index": 2}, content="another chunk")
    chunk_store.save([chunk1, chunk2, chunk3])

    # Search for a keyword
    results = chunk_store.search("hello")
    assert len(results) == 2
    assert results[0].content in ["hello world", "hello there"]
    assert results[1].content in ["hello world", "hello there"]

def test_search_no_matches(chunk_store):
    """Tests that search returns an empty list when no files match."""
    results = chunk_store.search("nonexistent")
    assert len(results) == 0

def test_search_empty_directory(chunk_store):
    """Tests that search returns an empty list when the directory is empty."""
    results = chunk_store.search("any")
    assert len(results) == 0

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
